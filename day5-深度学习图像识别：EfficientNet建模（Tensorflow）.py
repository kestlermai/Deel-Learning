# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 17:11:54 2023

@author: maihuanzhuo
"""

#深度学习图像识别：EfficientNet建模（Tensorflow）

'''
Efficientnet

EfficientNet是Google在2019年提出的一种新的卷积神经网络架构，主要目标是提高模型的效率，即在保持或提高模型性能的同时，尽可能地降低模型的复杂性和计算需求。

EfficientNet的关键思想是均衡网络的深度（层数）、宽度（每层的通道数）和分辨率（输入的图像尺寸）。
这是通过一种称为复合缩放（Compound Scaling）的方法实现的。在复合缩放中，深度、宽度和分辨率的缩放是同时进行的，而不是分别进行。
这种方法能够有效地利用模型的参数，提高模型的性能。

EfficientNet包括多个变体，从EfficientNet-B0到EfficientNet-B7，
其中"B"后面的数字越大，网络的深度和宽度越大，需要的计算资源也越多，但同时能够达到更高的性能。

其实就是在传统的CNN上增加了网络的深度、宽度以及图像的分辨率，但与传统的不一样的是，分别增加这三者的准确率会趋于饱和
而Efficientnet同时调整三者的参数，使得准确率还能有更大的提升
'''

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, Reshape, Softmax, GlobalAveragePooling2D, BatchNormalization
#dense全连接层，flatten（扁平化）将多维数据转换成一维向量，将数据展平；
#conv2D卷积层，MaxPool2D池化层，Dropout去除一定比例的神经元，避免过拟合
#Activation激活函数, Reshape数据转换（维度）, Softmax实现分类问题，输出概率值
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.python.keras import Sequential#模型容器，各个层按照顺序堆叠在一起，每一层都是前一层的输出
from tensorflow.python.keras import Model#
from tensorflow.python.keras.optimizers import adam_v2#模型优化器，根据梯度信息来更新模型的权重，以最小化训练过程中的损失函数
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, image_dataset_from_directory
#ImageDataGenerator于进行图像数据增强和批量数据生成的类，image_dataset_from_directory用于从图像文件夹创建 TensorFlow 数据集的函数
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import RandomFlip, RandomRotation, RandomContrast, RandomZoom, RandomTranslation
#RandomFlip图像随机翻转层，用于进行随机水平或垂直翻转图像。这有助于增加训练数据的多样性，从而提高模型的泛化能力。
#RandomRotation图像随机旋转层，用于进行随机旋转图像。这有助于增加数据多样性，减少模型对于特定角度的依赖性。
import os,PIL,pathlib
import warnings

import os
os.chdir('C:/Users/maihuanzhuo/Desktop/python-test/Deep Learning') ##修改路径

#设置GPU
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    tf.config.set_visible_devices([gpu0],"GPU")
    
warnings.filterwarnings("ignore") #忽略警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

#1.导入数据
data_dir = "./MTB"
#image_count = len(list(data_dir.glob('*/*')))检查是否有权限访问文件夹，True为有权限
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*')))
print("图片总数为：",image_count)

batch_size = 32#每次同时处理多少批次图片
img_height = 100#图片高度
img_width  = 100#图片宽度
#长宽影响图片放进模型中的分辨率

#关于image_dataset_from_directory()的详细介绍可以参考文章：https://mtyjkh.blog.csdn.net/article/details/117018789
#加载训练集数据
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,#数据比较少，选择0.2为验证集
    subset="training",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size)
#Using 7020 files for training.

#加载验证集
val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size)
#Using 3008 files for validation.

class_names = train_ds.class_names#获取分类名称
print(class_names)

#2.检查数据
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
#查看第一个图像批次和标签批次的形状
#(16, 150, 150, 3)#即第一批次有16张图，每张图片为150*150，3个通道：红、绿、蓝，代表这是彩色图片

#3.配置数据
AUTOTUNE = tf.data.AUTOTUNE#用于自动调整数据加载和预处理操作的并行程度，以提高性能

#对每个图像数据进行归一化，每个像素值缩放到0-255之间
def train_preprocessing(image,label):
    return (image/255.0,label)

#配置训练集以更好训练模型
train_ds = (
    train_ds.cache()#cache()函数将数据集添加到缓存当中，提高性能
    #.shuffle(1000)#每次迭代数据都被打乱，缓冲区中包含1000个样本；
    .shuffle(800)#这里如果样本量为800，那么应该取800，这里训练集为1280，既可以选1000也可以选800，都可以试试
    .map(train_preprocessing)    #map函数对图像数据进行归一化（上一步定义了train_preprocessing函数）
    .prefetch(buffer_size=AUTOTUNE)#预加载下一批数据
)

val_ds = (
    val_ds.cache()
    #.shuffle(1000)#然而，在验证数据中进行乱序实际上并不需要，因为在评估模型性能时，数据的顺序并不影响最终结果。
    #这并不是一个错误，只是一种不必要的操作。
    .map(train_preprocessing)    
    .prefetch(buffer_size=AUTOTUNE)
)

#4. 数据可视化
plt.figure(figsize=(10, 8))  
plt.suptitle("数据展示")
class_names = ["Tuberculosis","Normal"]
for images, labels in train_ds.take(1):#take(1)取一个批次
    for i in range(15):
        plt.subplot(4, 5, i + 1)#划分为4行5列，子图编号从1开始...
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]-1])

plt.show()

#数据增强
data_augmentation = Sequential([
  RandomFlip("horizontal_and_vertical"),
  RandomRotation(0.2),#旋转角度
  RandomContrast(1.0),#对比度
  RandomZoom(0.5,0.2),#图像缩放
  RandomTranslation(0.3,0.5),#图像平移，设置平移范围
])

#定义函数，对训练集进行图像增强
def prepare(ds):
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds

train_ds = prepare(train_ds)

#获取预训练模型对输入的预处理方法
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0#用的是B0版本，效果不好，换B2看看
from tensorflow.python.keras.applications.efficientnet import EfficientNetB2#B2也不行
from tensorflow.python.keras.applications.efficientnet import EfficientNetB4
from tensorflow.python.keras import Input, regularizers
IMG_SIZE = (img_height, img_width, 3)

base_model = EfficientNetB4(include_top=False, #是否包含顶层的全连接层
                         weights='imagenet')

inputs = Input(shape=IMG_SIZE)
#模型
x = base_model(inputs, training=False) #参数不变化
#全局池化
x = GlobalAveragePooling2D()(x)
#BatchNormalization
x = BatchNormalization()(x)
#Dropout
x = Dropout(0.5)(x)
#Dense
#x = Dense(128, kernel_regularizer=regularizers.l1(0.01))(x) #加个L1正则化
x = Dense(128, kernel_regularizer=regularizers.l2(0.01))(x)  # 添加 L2 正则化
#BatchNormalization
x = BatchNormalization()(x)
#激活函数
x = Activation('relu')(x)
#输出层
#x = Dense(2, kernel_regularizer=regularizers.l1(0.01))(x) #加个L1正则化
outputs = Dense(2, kernel_regularizer=regularizers.l2(0.01))(x)  # 添加 L2 正则化
#BatchNormalization
outputs = BatchNormalization()(outputs)
#激活函数
outputs = Activation('sigmoid')(outputs)
#整体封装
model = Model(inputs, outputs)
#打印模型结构
print(model.summary())

# Model: "model_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_6 (InputLayer)         [(None, 100, 100, 3)]     0         
# _________________________________________________________________
# efficientnetb4 (Functional)  (None, None, None, 1792)  17673823  
# _________________________________________________________________
# global_average_pooling2d_2 ( (None, 1792)              0         
# _________________________________________________________________
# batch_normalization_6 (Batch (None, 1792)              7168      
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 1792)              0         
# _________________________________________________________________
# dense_4 (Dense)              (None, 128)               229504    
# _________________________________________________________________
# batch_normalization_7 (Batch (None, 128)               512       
# _________________________________________________________________
# activation_4 (Activation)    (None, 128)               0         
# _________________________________________________________________
# dense_5 (Dense)              (None, 2)                 258       
# _________________________________________________________________
# batch_normalization_8 (Batch (None, 2)                 8         
# _________________________________________________________________
# activation_5 (Activation)    (None, 2)                 0         
# =================================================================
# Total params: 17,911,273
# Trainable params: 17,782,222
# Non-trainable params: 129,051
# _________________________________________________________________
# None
#参数多了很多

#编译模型
#定义优化器
from tensorflow.python.keras.optimizers import adam_v2, rmsprop_v2
#from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
optimizer = adam_v2.Adam()
#optimizer = SGD(learning_rate=0.001)
#optimizer = rmsprop_v2.RMSprop()
#编译模型
model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#训练模型
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

NO_EPOCHS = 30
PATIENCE  = 10
VERBOSE   = 1

# 设置动态学习率
annealer = LearningRateScheduler(lambda x: 1e-4 * 0.99 ** (x+NO_EPOCHS))#0.0001

#性能不提升时，减少学习率
#reduce = ReduceLROnPlateau(monitor='val_accuracy', 
#                           patience=PATIENCE,
#                           verbose=1,
#                           factor=0.8,
#                           min_lr=1e-6)

# 设置早停
earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)

# 设置模型检查点
checkpointer = ModelCheckpoint('./day5-TensorFlow_Efficientnet_MTB/mtb_best_model_EfficientNetB4.h5',
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True,
                                mode='max')
#训练模型
train_model  = model.fit(train_ds,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=val_ds,
                  callbacks=[earlystopper, checkpointer, annealer])
#B0、B2训练过程惨不忍睹，差的不行

#保存模型
model.save('./day5-TensorFlow_Efficientnet_MTB/mtb_best_model_EfficientNetB0.h5')

#Accuracy和Loss可视化
import matplotlib.pyplot as plt

loss = train_model.history['loss']
acc = train_model.history['accuracy']
val_loss = train_model.history['val_loss']
val_acc = train_model.history['val_accuracy']
epoch = range(1, len(loss)+1)

fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(epoch, loss, label='Train loss')
ax[0].plot(epoch, val_loss, label='Validation loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(epoch, acc, label='Train acc')
ax[1].plot(epoch, val_acc, label='Validation acc')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
plt.show()
#准确率一直随缘波动

#混淆矩阵可视化
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import math
# 定义一个绘制混淆矩阵图的函数
def plot_cm(labels, predictions):
    # 生成混淆矩阵
    conf_numpy = confusion_matrix(labels, predictions)
    # 将矩阵转化为 DataFrame
    conf_df = pd.DataFrame(conf_numpy, index=class_names ,columns=class_names)  
    plt.figure(figsize=(8,7))
    sns.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu")
    plt.title('混淆矩阵',fontsize=15)
    plt.ylabel('真实值',fontsize=14)
    plt.xlabel('预测值',fontsize=14)

val_pre   = []
val_label = []

for images, labels in val_ds:#这里可以取部分验证数据（.take(1)）生成混淆矩阵
    for image, label in zip(images, labels):
        # 需要给图片增加一个维度
        img_array = tf.expand_dims(image, 0) 
        # 使用模型预测图片中的人物
        prediction = model.predict(img_array)
        val_pre.append(np.argmax(prediction))
        val_label.append(label)

plot_cm(val_label, val_pre)

cm_val = confusion_matrix(val_label, val_pre)    
a_val = cm_val[0,0]
b_val = cm_val[0,1]
c_val = cm_val[1,0]
d_val = cm_val[1,1]
acc_val = (a_val+d_val)/(a_val+b_val+c_val+d_val) #准确率：就是被分对的样本数除以所有的样本数
error_rate_val = 1 - acc_val #错误率：与准确率相反，描述被分类器错分的比例
sen_val = d_val/(d_val+c_val) #灵敏度：表示的是所有正例中被分对的比例，衡量了分类器对正例的识别能力
sep_val = a_val/(a_val+b_val) #特异度：表示的是所有负例中被分对的比例，衡量了分类器对负例的识别能力
precision_val = d_val/(b_val+d_val) #精确度：表示被分为正例的示例中实际为正例的比例
F1_val = (2*precision_val*sen_val)/(precision_val+sen_val) #F1值：P和R指标有时候会出现的矛盾的情况，这样就需要综合考虑他们，最常见的方法就是F-Measure（又称为F-Score）
MCC_val = (d_val*a_val-b_val*c_val) / (math.sqrt((d_val+b_val)*(d_val+c_val)*(a_val+b_val)*(a_val+c_val))) #马修斯相关系数（Matthews correlation coefficient）：当两个类别具有非常不同的大小时，可以使用MCC
print("验证集的灵敏度为：",sen_val, 
      "验证集的特异度为：",sep_val,
      "验证集的准确率为：",acc_val, 
      "验证集的错误率为：",error_rate_val,
      "验证集的精确度为：",precision_val, 
      "验证集的F1为：",F1_val,
      "验证集的MCC为：",MCC_val)
# 验证集的灵敏度为： 0.7361111111111112 
# 验证集的特异度为： 1.0 
# 验证集的准确率为： 0.88125 
# 验证集的错误率为： 0.11875000000000002 
# 验证集的精确度为： 1.0 
# 验证集的F1为： 0.8480000000000001 
# 验证集的MCC为： 0.7780744130966396

train_pre   = []
train_label = []

for images, labels in train_ds:#这里可以取部分验证数据（.take(1)）生成混淆矩阵
    for image, label in zip(images, labels):
        # 需要给图片增加一个维度
        img_array = tf.expand_dims(image, 0)
        # 使用模型预测图片中的人物
        prediction = model.predict(img_array)
        train_pre.append(np.argmax(prediction))
        train_label.append(label)
        
plot_cm(train_label, train_pre)

cm_train = confusion_matrix(train_label, train_pre)  
a_train = cm_train[0,0]
b_train = cm_train[0,1]
c_train = cm_train[1,0]
d_train = cm_train[1,1]
acc_train = (a_train+d_train)/(a_train+b_train+c_train+d_train)
error_rate_train = 1 - acc_train
sen_train = d_train/(d_train+c_train)
sep_train = a_train/(a_train+b_train)
precision_train = d_train/(b_train+d_train)
F1_train = (2*precision_train*sen_train)/(precision_train+sen_train)
MCC_train = (d_train*a_train-b_train*c_train) / (math.sqrt((d_train+b_train)*(d_train+c_train)*(a_train+b_train)*(a_train+c_train))) 
print("训练集的灵敏度为：",sen_train, 
      "训练集的特异度为：",sep_train,
      "训练集的准确率为：",acc_train, 
      "训练集的错误率为：",error_rate_train,
      "训练集的精确度为：",precision_train, 
      "训练集的F1为：",F1_train,
      "训练集的MCC为：",MCC_train)
# 训练集的灵敏度为： 0.8597122302158273 
# 训练集的特异度为： 0.8632596685082873 
# 训练集的准确率为： 0.86171875 
# 训练集的错误率为： 0.13828125000000002 
# 训练集的精确度为： 0.82842287694974 
# 训练集的F1为： 0.8437775816416593 
# 训练集的MCC为： 0.7202156045594401

#AUC曲线绘制
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import math

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = metrics.roc_curve(labels, predictions)
    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.xlabel('False positives rate')
    plt.ylabel('True positives rate')
    ax = plt.gca()
    ax.set_aspect('equal')


val_pre_auc   = []
val_label_auc = []

#取验证集数据生成roc曲线
for images, labels in val_ds:
    for image, label in zip(images, labels):      
        img_array = tf.expand_dims(image, 0) 
        prediction_auc = model.predict(img_array)
        val_pre_auc.append((prediction_auc)[:,1])
        val_label_auc.append(label)

auc_score_val = metrics.roc_auc_score(val_label_auc, val_pre_auc)


train_pre_auc   = []
train_label_auc = []

#取训练集数据生成roc曲线
for images, labels in train_ds:
    for image, label in zip(images, labels):
        img_array_train = tf.expand_dims(image, 0) 
        prediction_auc = model.predict(img_array_train)
        train_pre_auc.append((prediction_auc)[:,1])#输出概率而不是标签！
        train_label_auc.append(label)

auc_score_train = metrics.roc_auc_score(train_label_auc, train_pre_auc)

plot_roc('validation AUC: {0:.4f}'.format(auc_score_val), val_label_auc , val_pre_auc , color="red", linestyle='--')
plot_roc('training AUC: {0:.4f}'.format(auc_score_train), train_label_auc, train_pre_auc, color="blue", linestyle='--')
plt.legend(loc='lower right')
#plt.savefig("roc.pdf", dpi=300,format="pdf")

print("训练集的AUC值为：",auc_score_train, "验证集的AUC值为：",auc_score_val)
# 训练集的AUC值为： 0.95611163003299 验证集的AUC值为： 0.9998421717171717