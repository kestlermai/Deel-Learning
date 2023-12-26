# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 22:27:44 2023

@author: maihuanzhuo
"""

#深度学习图像识别：Inception V3建模（Tensorflow）
'''
# Inception V1
# Inception是一种深度学习模型，也被称为GoogLeNet，因为它是由Google的研究人员开发的。
# Inception模型的主要特点是它的“网络中的网络”结构，也就是说，它在一个大网络中嵌入了很多小网络。
# Inception模型中的每个小网络都有自己的任务，它们可以处理不同尺度的特征。然后，这些小网络的输出被合并在一起，形成模型的最终输出。
# 这种结构使得Inception模型能够更有效地处理复杂的图像识别任务。

# Inception V2和V3
# 这两个版本引入了两个重要的概念：分解（Factorization）和批标准化（Batch Normalization）。
# 分解是指将大的卷积核分解成几个小的卷积核，这样可以减少模型的复杂度，提高计算效率。
# 批标准化是一种技术，可以使模型的训练更稳定，加快训练速度。
'''

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, Reshape, Softmax, GlobalAveragePooling2D
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
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import RandomFlip, RandomRotation
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
data_dir = "./cat_dog"
#image_count = len(list(data_dir.glob('*/*')))检查是否有权限访问文件夹，True为有权限
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*')))
print("图片总数为：",image_count)

batch_size = 32#每次同时处理多少批次图片
img_height = 150#图片高度
img_width  = 150#图片宽度
#长宽影响图片放进模型中的分辨率

#加载训练集数据
train_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    seed=1234,
    image_size=(img_height, img_width),
    batch_size=batch_size)
#Using 7020 files for training.

#加载验证集
val_ds = image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
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
    .shuffle(1000)#每次迭代数据都被打乱，缓冲区中包含1000个样本
    .map(train_preprocessing)    #map函数对图像数据进行归一化（上一步定义了train_preprocessing函数）
    .prefetch(buffer_size=AUTOTUNE)#预加载下一批数据
)

val_ds = (
    val_ds.cache()
    .shuffle(1000)
    .map(train_preprocessing)    
    .prefetch(buffer_size=AUTOTUNE)
)

#4. 数据可视化
plt.figure(figsize=(10, 8))  
plt.suptitle("数据展示")
class_names = ["Dog","Cat"]
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
  RandomRotation(0.2)#旋转角度
  #RandomContrast(1.0),#图像对比度
  #RandomZoom(0.5,0.2),#图像缩放
  #RandomTranslation(0.3,0.5),#图像平移
  #RandomCrop(,),#裁剪指定大小的图像
  #RandomHeight(,),#调整图像高度
  #RandomWidth(,)#调整图像宽度
  ])

#定义函数，对训练集进行图像增强
def prepare(ds):
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)#指定并行处理的线程数量
    return ds

train_ds = prepare(train_ds)

#导入Inception V3
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras import Input
from tensorflow.keras.layers import BatchNormalization
IMG_SIZE = (img_height, img_width, 3)
base_model = InceptionV3(include_top=False, #是否包含顶层的全连接层
                         weights='imagenet')#weights参数设置为'imagenet'，表示使用在ImageNet数据集上预训练的权重。

#迁移学习主流程代码，开始利用预训练的VGG19创建模型
inputs = Input(shape=IMG_SIZE)#(150,150,3)
#模型
x = base_model(inputs, training=False) #参数不变化
#全局池化
x = GlobalAveragePooling2D()(x)
#BatchNormalization
x = BatchNormalization()(x)
#Dropout
x = Dropout(0.3)(x)
#Dense
x = Dense(512)(x)
#BatchNormalization
x = BatchNormalization()(x)
#激活函数
x = Activation('relu')(x)
#输出层
outputs = Dense(2)(x)
#BatchNormalization
outputs = BatchNormalization()(outputs)
#激活函数
outputs = Activation('sigmoid')(outputs)
#整体封装
model = Model(inputs, outputs)
#打印模型结构
print(model.summary())

# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_3 (InputLayer)         [(None, 150, 150, 3)]     0         
# _________________________________________________________________
# inception_v3 (Functional)    (None, None, None, 2048)  21802784  使用InceptionV3原始参数
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 2048)              0         全局平均池化降低维度 
# _________________________________________________________________
# batch_normalization_188 (Bat (None, 2048)              8192      数据归一化
# _________________________________________________________________
# dropout (Dropout)            (None, 2048)              0         
# _________________________________________________________________
# dense (Dense)                (None, 512)               1049088   
# _________________________________________________________________
# batch_normalization_189 (Bat (None, 512)               2048      
# _________________________________________________________________
# activation_188 (Activation)  (None, 512)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 1026      
# _________________________________________________________________
# batch_normalization_190 (Bat (None, 2)                 8         
# _________________________________________________________________
# activation_189 (Activation)  (None, 2)                 0         
# =================================================================
# Total params: 22,863,146
# Trainable params: 22,823,590
# Non-trainable params: 39,556
# _________________________________________________________________
# None

#定义优化器
from tensorflow.python.keras.optimizers import adam_v2, rmsprop_v2
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
optimizer = adam_v2.Adam()
#optimizer = SGD(learning_rate=0.001)
#optimizer = rmsprop_v2.RMSprop()
#编译模型
model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#训练模型
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

NO_EPOCHS = 20
PATIENCE  = 10
VERBOSE   = 1

# 设置动态学习率
annealer = LearningRateScheduler(lambda x: 1e-4 * 0.99 ** (x+NO_EPOCHS))#学习率太高会影响模型无法收敛，我改成0.0001开始

# 设置早停
earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)

# 设置模型检查点
checkpointer = ModelCheckpoint('./day3-TensorFlow_Inception_V3_cat_dog/cat_dog_best_model_inception_v3.h5',
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

#训练模型
train_model  = model.fit(train_ds,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=val_ds,
                  callbacks=[earlystopper, checkpointer, annealer])

# # 加载权重
# model.load_weights('./TensorFlow_Inception_V3_cat_dog/cat_dog_best_model_inception_v3.h5')

# #保存模型
# model.save('./TensorFlow_Inception_V3_cat_dog/cat_dog_best_model_inception_v3.h5')
# print("The trained model has been saved.")

# from tensorflow.python.keras.models import load_model
# train_model=load_model('./TensorFlow_Inception_V3_cat_dog/cat_dog_best_model_inception_v3.h5')

#保存模型
model.save('./day3-TensorFlow_Inception_V3_cat_dog/cat_dog_best_model_inception_v3.h5')

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
#看着不错

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
# 验证集的灵敏度为： 0.9654485049833887 
# 验证集的特异度为： 0.9620758483033932
# 验证集的准确率为： 0.9637632978723404 
# 验证集的错误率为： 0.036236702127659615 
# 验证集的精确度为： 0.9622516556291391 
# 验证集的F1为： 0.9638474295190713 
# 验证集的MCC为： 0.9275315291298019

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
# 训练集的灵敏度为： 0.9917425968109339 
# 训练集的特异度为： 0.9911630558722919 
# 训练集的准确率为： 0.9914529914529915 
# 训练集的错误率为： 0.008547008547008517 
# 训练集的精确度为： 0.9911781445645987 
# 训练集的F1为： 0.9914602903501282 
# 训练集的MCC为： 0.9829061313676634

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
# 训练集的AUC值为： 0.9991978958605517 验证集的AUC值为： 0.9912865299301727


#测试模型
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import img_to_array
from PIL import Image
import os, shutil, pathlib

img_height = 150#图片高度
img_width  = 150#图片宽度

label = np.array(["Cat","Dog"])#0、1赋值给标签

#载入模型
model = load_model('./day3-TensorFlow_Inception_V3_cat_dog/cat_dog_best_model_inception_v3.h5')

#导入图片
image = image.load_img('C:/Users/maihuanzhuo/Desktop/python-test/Deep Learning/maomao1.jpg')#手动修改路径，删除隐藏字符
plt.imshow(image)
plt.show()

#调整图片尺寸
image = image.resize((img_width,img_height))
image = img_to_array(image)
image = image/255#数值归一化，转为0-1
image = np.expand_dims(image,0)
print(image.shape)

# 使用模型进行预测
predictions = model.predict(image)
predicted_class = np.argmax(predictions)#np.argmax返回最大值的索引

# 打印预测的类别
print(label[predicted_class])
#好咯能识别出来