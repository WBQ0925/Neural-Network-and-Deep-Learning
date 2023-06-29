import torch
from torch import nn
from torch import functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from model import VisionTransformer
import numpy as np
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('GPU: ', torch.cuda.get_device_name(0))

else:
    device = torch.device("cpu")
    print('No GPU available')

alpha = 0.2
cutmix_prob = 0.1
lr = 1e-4
batch_size = 256
num_workers = 2
patch_size = 4
image_sz = 32
max_len = 100
embed_dim = 512
classes = 100
layers = 6
channels = 3
heads = 16
epochs = 100
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, dataloader, criterion, optimizer):
    running_loss = 0.0
    running_accuracy = 0.0

    for data, target in tqdm(dataloader):
        inputs = data.to(device)
        labels = target.to(device)
        r = np.random.rand(1)

        if alpha > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(alpha, alpha)
            rand_index = torch.randperm(inputs.size()[0])
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            output, _ = model(inputs)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            # compute output
            output,_ = model(inputs)
            loss = criterion(output, labels)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == labels).float().mean()
        running_accuracy += acc / len(dataloader)
        running_loss += loss.item() / len(dataloader)

    return running_loss, running_accuracy

def evaluation(model, dataloader, criterion):
    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        test_top5_accuracy = 0.0
        
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)
            
            output, _ = model(data)
            loss = criterion(output, target)
            
            acc = (output.argmax(dim=1) == target).float().mean()
            _, predicted = output.topk(5, 1)
            correct = predicted.eq(target.view(-1, 1).expand_as(predicted))
            top5_acc = correct[:, :5].any(dim=1).float().mean()
            
            test_accuracy += acc / len(dataloader)
            test_top5_accuracy += top5_acc / len(dataloader)
            test_loss += loss.item() / len(dataloader)

    return test_loss, test_accuracy, test_top5_accuracy


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='logs')
model = VisionTransformer(
    patch_size=patch_size,
    max_len=max_len,
    embed_dim=embed_dim,
    classes=classes,
    layers=layers,
    channels=channels,
    heads=heads).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

train_accs = []
train_losses = []
test_accs = []
test_top5_accs = []
test_losses = []
for epoch in range(epochs):
    running_loss, running_accuracy = train(model, train_dataloader, criterion, optimizer)
    writer.add_scalar(tag="accuracy", # 可以暂时理解为图像的名字
                    scalar_value=running_accuracy,  # 纵坐标的值
                    global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                    )
    writer.add_scalar(tag="loss", # 可以暂时理解为图像的名字
                scalar_value=running_loss,  # 纵坐标的值
                global_step=epoch  # 当前是第几次迭代，可以理解为横坐标的值
                )
    print(f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n")
    train_accs.append(running_accuracy)
    train_losses.append(running_loss)

    test_loss, test_accuracy, test_top5_accuracy = evaluation(model, test_dataloader, criterion)
    print(f"test acc: {test_accuracy:.4f} -top 5 acc: {test_top5_accuracy:.4f} - test loss : {test_loss:.4f}\n")
    test_accs.append(test_accuracy)
    test_top5_accs.append(test_top5_accs)
    test_losses.append(test_loss)

    if (epoch + 1) % 10 == 0:
        path = './'
        torch.save(model.state_dict(), path + 'model_epoch_{}.pt'.format(epoch + 1))