#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

# 定义AlexNet模型
def AlexNet(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Max2D((3, 3), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# 数据预处理
def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x /= 255.0
    y = tf.squeeze(y)
    y = tf.one_hot(y, depth=100)
    return x, y

# 加载CIFAR-100数据集
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# 数据预处理
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.map(preprocess)
train_dataset = train_dataset.batch(128)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.map(preprocess)
test_dataset = test_dataset.batch(128)

# 定义训练参数
epochs = 100
learning_rate = 0.001
num_classes = 100
input_shape = (32, 32, 3)

# 定义优化器和损失函数
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# 定义TensorBoard回调函数
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

# 定义训练函数
def train(model, train_dataset, test_dataset, optimizer, loss_fn, epochs):
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    test_mAP_results = []
    test_mIoU_results = []

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, logits)

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        test_loss_avg = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        test_mAP = tf.keras.metrics.MeanAveragePrecision()
        test_mIoU = tf.keras.metrics.MeanIoU(num_classes=num_classes)

        for x, y in test_dataset:
            logits = model(x, training=False)
            loss_value = loss_fn(y, logits)

            test_loss_avg.update_state(loss_value)
            test_accuracy.update_state(y, logits)
            test_mAP.update_state(y, logits)
            test_mIoU.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))

        test_loss_results.append(test_loss_avg.result())
        test_accuracy_results.append(test_accuracy.result())
        test_mAP_results.append(test_mAP.result())
        test_mIoU_results.append(test_mIoU.result())

        print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}, Test mAP: {:.3f}, Test mIoU: {:.3f}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result(), test_loss_avg.result(), test_accuracy.result(), test_mAP.result(), test_mIoU.result()))

    return train_loss_results, train_accuracy_results, test_loss_results, test_accuracy_results, test_mAP_results, test_mIoU_results

# 训练baseline模型
baseline_model = AlexNet(input_shape, num_classes)
baseline_train_loss, baseline_train_accuracy, baseline_test_loss, baseline_test_accuracy, baseline_test_mAP, baseline_test_mIoU = train(baseline_model, train_dataset, test_dataset, optimizer, loss_fn, epochs)

# 定义cutmix函数
def cutmix(x1 y1, x2, y2, alpha=1.0):
    batch_size = tf.shape(x1)[0]
    height, width, channels = tf.shape(x1)[1], tf.shape(x1)[2], tf.shape(x1)[3]
    lam = tf.random.uniform([], 0.0, 1.0)
    bbx1, bby1, bbx2, bby2 = rand_bbox(height, width, lam, alpha)
    x1_cutmix = x1
    x2_cutmix = tf.image.crop_to_bounding_box(x2, bbx1, bby1, bbx2 - bbx1, bby2 - bby1)
    y1_cutmix = y1
    y2_cutmix = y2
    x_cutmix = tf.concat([x1_cutmix, x2_cutmix], axis=0)
    y_cutmix = tf.concat([y1_cutmix, y2_cutmix], axis=0)
    return x_cutmix, y_cutmix

# 定义cutout函数
def cutout(x, length=16):
    batch_size = tf.shape(x)[0]
    height, width, channels = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    mask = tf.ones((batch_size, length, length, channels))
    row = tf.random.uniform([], 0, height, tf.int32)
    col = tf.random.uniform([], 0, width, tf.int32)
    cutout_area = tf.zeros((batch_size, height, width, channels))
    cutout_area = tf.tensor_scatter_nd_update(cutout_area, [[0, row, col, 0]], mask)
    return tf.where(tf.equal(cutout_area, 0), x, cutout_area)

# 定义mixup函数
def mixup(x1, y1, x2, y2, alpha=1.0):
    batch_size = tf.shape(x1)[0]
    lam = tf.random.uniform([], 0.0, 1.0)
    x_mixup = lam * x1 + (1 - lam) * x2
    y_mixup = lam * y1 + (1 - lam) * y2
    return x_mixup, y_mixup

# 定义可视化函数
def visualize(x1, y1, x2, y2, x_cutmix, y_cutmix, x_cutout, x_mixup, y_mixup):
    plt.figure(figsize=(10, 10))
    for i in range(3):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x1[i])
        plt.title("Class: {}".format(np.argmax(y1[i])))
        plt.axis('off')
    for i in range(3):
        plt.subplot(3, 3, i + 4)
        plt.imshow(x2[i])
        plt.title("Class: {}".format(np.argmax(y2[i])))
        plt.axis('off')
    for i in range(3):
        plt.subplot(3, 3, i + 7)
        plt.imshow(x_cutmix[i])
        plt.title("Class: {}".format(np.argmax(y_cutmix[i])))
        plt.axis('off')
    plt.show()

# 定义训练函数（使用cutmix）
def train_cutmix(model, train_dataset, test_dataset, optimizer, loss_fn, epochs, alpha=1.0):
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    test_mAP_results = []
    test_mIoU_results = []

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x1, y1 in train_dataset:
            x2, y2 = next(iter(train_dataset))
            x_cutmix, y_cutmix = cutmix(x1, y1, x2, y2, alpha)
            with tf.GradientTape() as tape:
                logits = model(x_cutmix, training=True)
                loss_value = loss_fn(y_cutmix, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y_cutmix, logits)

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        test_loss_avg = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        test_mAP = tf.keras.metrics.MeanAveragePrecision()
        test_mIoU = tf.keras.metrics.MeanIoU(num_classes=num_classes)

        for x, y in test_dataset:
            logits = model(x, training=False)
            loss_value = loss_fn(y, logits)

            test_loss_avg.update_state(loss_value)
            test_accuracy.update_state(y, logits)
            test_mAP.update_state(y, logits)
            test_mIoU.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))

        test_loss_results.append(test_loss_avg.result())
        test_accuracy_results.append(test_accuracy.result())
        test_mAP_results.append(test_mAP.result())
        test_mIoU_results.append(test_mIoU.result())

        print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}, Test mAP: {:.3f}, Test mIoU: {:.3f}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result(), test_loss_avg.result(), test_accuracy.result(), test_mAP.result(), test_mIoU.result()))

    return train_loss_results, train_accuracy_results, test_loss_results, test_accuracy_results, test_mAP_results, test_mIoU_results

# 训练cutmix模型
cutmix_model = AlexNet(input_shape, num_classes)
cutmix_train_loss, cutmix_train_accuracy, cutmix_test_loss, cutmix_test_accuracy, cutmix_test_mAP, cutmix_test_mIoU = train_cutmix(cutmix_model, train_dataset, test_dataset, optimizer, loss_fn, epochs)

# 定义训练函数（使用cutout）
def train_cutout(model, train_dataset, test_dataset, optimizer, loss_fn, epochs, length=16):
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    test_mAP_results = []
    test_mIoU_results = []

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x, y in train_dataset:
            x_cutout = cutout(x, length)
            with tf.GradientTape() as tape:
                logits = model(x_cutout, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, logits)

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        test_loss_avg = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        test_mAP = tf.keras.metrics.MeanAveragePrecision()
        test_mIoU = tf.keras.metrics.MeanIoU(num_classes=num_classes)

        for x, y in test_dataset:
            logits = model(x, training=False)
            loss_value = loss_fn(y, logits)

            test_loss_avg.update_state(loss_value)
            test_accuracy.update_state(y, logits)
            test_mAP.update_state(y, logits)
            test_mIoU.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))

        test_loss_results.append(test_loss_avg.result())
        test_accuracy_results.append(test_accuracy.result())
        test_mAP_results.append(test_mAP.result())
        test_mIoU_results.append(test_mIoU.result())

        print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}, Test mAP: {:.3f}, Test mIoU: {:.3f}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result(), test_loss_avg.result(), test_accuracy.result(), test_mAP.result(), test_mIoU.result()))
    return train_loss_results, train_accuracy_results, test_loss_results, test_accuracy_results, test_mAP_results, test_mIoU_results
# 训练cutout模型
cutout_model = AlexNet(input_shape, num_classes)
cutout_train_loss, cutout_train_accuracy, cutout_test_loss, cutout_test_accuracy, cutout_test_mAP, cutout_test_mIoU = train_cutout(cutout_model, train_dataset, test_dataset, optimizer, loss_fn, epochs)

# 定义训练函数（使用mixup）
def train_mixup(model, train_dataset, test_dataset, optimizer, loss_fn, epochs, alpha=1.0):
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    test_mAP_results = []
    test_mIoU_results = []

    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for x1, y1 in train_dataset:
            x2, y2 = next(iter(train_dataset))
            x_mixup, y_mixup = mixup(x1, y1, x2, y2, alpha)
            with tf.GradientTape() as tape:
                logits = model(x_mixup, training=True)
                loss_value = loss_fn(y_mixup, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y_mixup, logits)

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        test_loss_avg = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        test_mAP = tf.keras.metrics.MeanAveragePrecision()
        test_mIoU = tf.keras.metrics.MeanIoU(num_classes=num_classes)

        for x, y in test_dataset:
            logits = model(x, training=False)
            loss_value = loss_fn(y, logits)

            test_loss_avg.update_state(loss_value)
            test_accuracy.update_state(y, logits)
            test_mAP.update_state(y, logits)
            test_mIoU.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))

        test_loss_results.append(test_loss_avg.result())
        test_accuracy_results.append(test_accuracy.result())
        test_mAP_results.append(test_mAP.result())
        test_mIoU_results.append(test_mIoU.result())

        print("Epoch {:03d}: Train Loss: {:.3f}, Train Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}, Test mAP: {:.3f}, Test mIoU: {:.3f}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result(), test_loss_avg.result(), test_accuracy.result(), test_mAP.result(), test_mIoU.result()))

    return train_loss_results, train_accuracy_results, test_loss_results, test_accuracy_results, test_mAP_results, test_mIoU_results

# 训练mixup模型
mixup_model = AlexNet(input_shape, num_classes)
mixup_train_loss, mixup_train_accuracy, mixup_test_loss, mixup_test_accuracy, mixup_test_mAP, mixup_test_mIoU = train_mixup(mixup_model, train_dataset, test_dataset, optimizer, loss_fn, epochs)

