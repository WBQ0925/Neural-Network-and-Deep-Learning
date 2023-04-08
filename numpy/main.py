import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pickle
# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int32')

# 将标签转换为one-hot编码
num_labels = len(np.unique(y))
y = np.eye(num_labels)[y]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array(X_train)
y_train = np.array(y_train)

# 对训练集进行随机打乱
shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# 定义网络架构和超参数
input_size = X_train.shape[1]
hidden_size = 128
output_size = num_labels
batch_size = 128
learning_rate = 0.1
reg_lambda = 0.001

# 初始化权重
W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
b2 = np.zeros(output_size)

# 定义激活函数和其导数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

# 定义损失函数和其导数
def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_loss_deriv(y_true, y_pred):
    return y_pred - y_true

# 定义SGD优化器
def sgd_optimizer(learning_rate, params, grads):
    for param, grad in zip(params, grads):
        param -= learning_rate * grad

# 训练模型
train_loss_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(50):
    # 随机打乱训练集
    shuffle_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # 计算训练集上的损失和准确率
    train_loss = 0.0
    train_acc = 0.0
    for i in range(0, len(X_train), batch_size):
        # 前向传播
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        z1 = np.dot(batch_X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

        # 计算损失和准确率
        train_loss += cross_entropy_loss(batch_y, y_pred)
        train_acc += np.sum(np.argmax(batch_y, axis=1) == np.argmax(y_pred, axis=1))

        # 反向传播
        delta3 = cross_entropy_loss_deriv(batch_y, y_pred)
        delta2 = np.dot(delta3, W2.T) * sigmoid_deriv(a1)
        dW2 = np.dot(a1.T, delta3) / batch_size + reg_lambda * W2
        db2 = np.sum(delta3, axis=0) / batch_size
        dW1 = np.dot(batch_X.T, delta2) / batch_size + reg_lambda * W1
        db1 = np.sum(delta2, axis=0) / batch_size

        # 使用SGD更新权重和偏置
        sgd_optimizer(learning_rate, [W2, b2, W1, b1], [dW2, db2, dW1, db1])

    # 每个epoch结束时记录训练集的损失和准确率
    train_loss /= len(X_train)
    train_acc /= len(X_train)
    train_loss_history.append(train_loss)

    # 计算测试集上的损失和准确率
    test_loss = 0.0
    test_acc = 0.0
    for i in range(0, len(X_test), batch_size):
        # 前向传播
        batch_X = X_test[i:i + batch_size]
        batch_y = y_test[i:i + batch_size]
        z1 = np.dot(batch_X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

        # 计算损失和准确率
        test_loss += cross_entropy_loss(batch_y, y_pred)
        test_acc += np.sum(np.argmax(batch_y, axis=1) == np.argmax(y_pred, axis=1))

    # 每个epoch结束时记录测试集的损失和准确率
    test_loss /= len(X_test)
    test_acc /= len(X_test)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

    # 打印训练进度
    print('Epoch {}/{} - train_loss: {:.4f} - train_acc: {:.4f} - test_loss: {:.4f} - test_acc: {:.4f}'
          .format(epoch + 1, 50, train_loss, train_acc, test_loss, test_acc))
    if epoch == 49:
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
# 绘制训练和测试的损失曲线
plt.plot(train_loss_history, label='train loss')
plt.plot(test_loss_history, label='test loss')
plt.title('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()

# 绘制测试的准确率曲线
plt.plot(test_acc_history)
plt.title('Test Accuracy')
plt.savefig('Accuracy.png')
plt.show()
# 导入模型
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']

# 用经过参数查找后的模型进行测试
test_loss = 0.0
test_acc = 0.0
for i in range(0, len(X_test), batch_size):
    # 前向传播
    batch_X = X_test[i:i + batch_size]
    batch_y = y_test[i:i + batch_size]
    z1 = np.dot(batch_X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

    # 计算损失和准确率
    test_loss += cross_entropy_loss(batch_y, y_pred)
    test_acc += np.sum(np.argmax(batch_y, axis=1) == np.argmax(y_pred, axis=1))

# 输出分类精度
test_loss /= len(X_test)
test_acc /= len(X_test)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

import matplotlib.pyplot as plt

# 显示第一层的权重
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.title("W1")
plt.imshow(W1, cmap="gray")

# 显示第一层的偏置
plt.subplot(2, 2, 2)
plt.title("b1")
plt.bar(range(hidden_size), b1)

# 显示第二层的权重
plt.subplot(2, 2, 3)
plt.title("W2")
plt.imshow(W2, cmap="gray")

# 显示第二层的偏置
plt.subplot(2, 2, 4)
plt.title("b2")
plt.bar(range(output_size), b2)

plt.savefig('network_parameters.png')
plt.show()