#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np


# In[2]:


# 下载并加载 CIFAR-100 数据集
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# 切分数据集
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# 数据预处理
train_images = train_images / 255.0
val_images = val_images / 255.0


# In[3]:


# 定义网络结构
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=4096, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(units=100, activation='softmax'))

# 打印网络结构
model.summary()


# In[4]:


# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# 添加 TensorBoard 记录训练日志
tensorboard_callback = tf.keras.callbacks.TensorBoard("./logs")


# In[5]:


# 模型训练
history = model.fit(train_images, train_labels, batch_size=128, epochs=100, validation_data=(val_images, val_labels), callbacks=[tensorboard_callback])

# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy: ", test_acc)

# 可视化训练和测试的 loss 曲线和 accuracy 曲线
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[10]:


# cutmix, cutout, mixup 可视化
import copy
from skimage import transform

# 定义cutmix函数
def cutmix(image, label, prob=0.5):
    '''
    随机裁剪出一个矩形区域，将该区域与另一张随机选择的图像的对应区域进行拼接
    '''
    if np.random.rand() > prob:
        return image, label
     
    # 图像宽、高
    h, w = image.shape[0], image.shape[1]
    
    # 随机裁剪出一个矩形区域
    x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
    x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    cut_area = image[y_min:y_max, x_min:x_max]
    
    # 随机选择另一张图像
    rand_idx = np.random.randint(len(train_images))
    rand_image, rand_label = train_images[rand_idx], train_labels[rand_idx]
    
    # 将图像拼接起来
    new_image = np.copy(image)
    new_image[y_min:y_max, x_min:x_max] = rand_image[y_min:y_max, x_min:x_max]
    
    # 计算标签混合权重
    weight = (x_max - x_min) * (y_max - y_min) / (w * h)
    new_label = label * (1 - weight) + rand_label * weight

    return new_image, new_label


# 定义cutout函数
def cutout(image, label, prob=0.5, size=16):
    '''
    随机将一块区域涂黑
    '''
    if np.random.rand() > prob:
        return image, label
    
    # 图像宽、高
    h, w = image.shape[0], image.shape[1]
    
    # 随机涂黑一块区域
    x, y = np.random.randint(0, w), np.random.randint(0, h)
    x_min, y_min = max(0, x - size // 2), max(0, y - size // 2)
    x_max, y_max = min(w, x + size // 2), min(h, y + size // 2)
    
    image[y_min:y_max, x_min:x_max] = 0
    
    return image, label


# 定义mixup函数
def mixup(image1, label1, prob=0.5):
    '''
    随机选择另一张图像，将两张图像线性混合
    '''
    if np.random.rand() > prob:
        return image1, label1
    
    # 随机选择另一张图像
    rand_idx = np.random.randint(len(train_images))
    image2, label2 = train_images[rand_idx], train_labels[rand_idx]
    
    # 计算混合权重
    lam = np.random.beta(1, 1)
    
    # 线性混合
    new_image = lam * image1 + (1 - lam) * image2
    new_label = lam * label1 + (1 - lam) * label2
    
    return new_image, new_label


cutmix_images, _ = cutmix(train_images, train_labels)
cutout_images, _ = cutout(train_images, train_labels)
mixup_images, _ = mixup(train_images, train_labels)


# In[19]:


# 取出三张训练样本进行可视化
indices = [3, 11, 17]
for i, idx in enumerate(indices):
    image = train_images[idx]
    label = train_labels[idx]

    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    axs[0, 1].imshow(image)
    axs[0, 1].set_title('Original Image')

    # cutmix可视化
    cutmix_image, cutmix_label = cutmix(image, label)
    axs[1, 0].imshow(cutmix_image)
    axs[1, 0].set_title('CutMix Image')

    # cutout可视化
    cutout_image, cutout_label = cutout(image, label)
    axs[1, 1].imshow(cutout_image)
    axs[1, 1].set_title('Cutout Image')

    # Mixup可视化
    mixup_image, mixup_label = mixup(image, label)
    axs[1, 2].imshow(mixup_image)
    axs[1, 2].set_title('Mixup Image')

    axs[2, 0].axis('off')
    axs[2, 1].axis('off')
    axs[2, 2].axis('off')
    axs[2, 0].text(0, 0.5, 'Label: {}'.format(label), fontsize=12)
    axs[2, 1].text(0, 0.5, 'CutMix Label: {:.1f}'.format(cutmix_label[0]), fontsize=12)
    axs[2, 2].text(0, 0.5, 'Mixup Label: {:.1f}'.format(mixup_label[0]), fontsize=12)

    plt.suptitle('Image {}'.format(i+1), fontsize=16)
    plt.tight_layout()
    plt.show()


# In[20]:


# 保存训练好的模型
model.save('./cifar100_alexnet.h5')


# In[ ]:




