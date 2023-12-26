# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 23:00:47 2023

@author: maihuanzhuo
"""
'''
# 积神经网络（Convolutional Neural Networks，简称CNN）
# CNN的基本构建块包括卷积层、池化层和全连接层

# 1.卷积层是用来提取图像中的局部特征
# 卷积层主要通过卷积核来提取特征。
# 举个例子，我们把一张图片想象成一座山，那么卷积层就像是一只鹰在飞翔，卷积核就好比鹰的眼睛，它通过眼睛扫视着山。但是每个时间点仅仅扫视山的一小部分，
# 并抓取那里的信息（实际上在进行一个数学运算，这个运算会将图片的一小块区域转化为一个数字，而这个数字就代表了那个区域的特征）
# 卷积核可以设计为很多种，侧重于提取的特征也不一致。有些专门提取竖线，有些专门提取横线，有些专门提取毛发等等。
# 因此，CNN一次可以使用多个卷积核进行扫描，得到了CNN的第一个卷积层（包含了N个原图的特征图），也就是仅保留了部分重要的图片信息

# 2.池化层是用来降低数据的维度
# 经过第一轮扫描得到了第一个卷积层，仅仅保留了重要信息，存在很多冗余的数据。
# 因此，需要把多余的部分删了，这个过程就是池化。最终目的在于，缩小图片尺寸，同时保留基本特征。

# 3.全连接层则是用来将学习到的各种特征进行组合，进行最后的分类或回归。
# 一个CNN模型，一般这么设计：卷积（提取特征）——>池化（缩小图片）——>再卷积（再提取特征）——>再池化（再缩小图片）
# ——>循环卷积和池化——>全连接层做分类（类似之前介绍的基础分类模型）
'''

from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, Reshape, Softmax
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


#搭建一个卷积网络
#Sequential层=构建一个序列化的container（容器），可以把想要在神经网络中添加的操作都放进去，按顺序进行执行。
# 设置层
#在这里设置了5个卷积层和4个池化层
#filters滤波器，也叫做卷积核：
#每次卷积之后，图像大小都变小了，那么就无法卷积多次，所以我们在每次卷积之后都给图像进行padding
#给图像周围都补充一圈空白，跟卷积前一样大，而且保证了图像边缘被计算在内。
#这种padding方式称为'same'方式，而不留任何填补称为'valid'
# activation:池化层、全连接层等通过线性运算构建的是线性模型，该模型的表达能力有限；
# 激活函数能够加入非线性因素，从而使模型的表达能力得到极大提升。包括Sigmoid、tanh、ReLU、LeakyReLU、pReLU、ELU、maxout等函数。
# MaxPooling和AveragePooling，采取2*2窗口取最大值或者是平均值，strides设置步长（滑动窗口）
model = Sequential([
                    Conv2D(filters=32, kernel_size=(3,3), padding='same',
                            input_shape=(img_width,img_height,3),  activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),  
                    Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)), 
                    Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
                    Flatten(),
                    Dense(512, activation='relu'),
                    Dense(2,activation='sigmoid')#二分类
                  ])
#打印模型结构
print(model.summary())
#Total params: 11,596,866

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d (Conv2D)              (None, 150, 150, 32)      896       卷积
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 75, 75, 32)        0         池化
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 75, 75, 64)        18496     再卷积
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 37, 37, 64)        0         再池化
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 37, 37, 128)       73856     
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 18, 18, 128)       0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 18, 18, 256)       295168    
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 9, 9, 256)         0         
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 9, 9, 256)         590080    
# _________________________________________________________________
# flatten (Flatten)            (None, 20736)             0         数据维度转换，扁平层
# _________________________________________________________________
# dense (Dense)                (None, 512)               10617344  全连接层
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 1026      最终二分类结局：cat or dog
# =================================================================
# Total params: 11,596,866
# Trainable params: 11,596,866
# Non-trainable params: 0
# _________________________________________________________________
# None

#编译模型
#负责优化模型的权重以减小损失函数的值
#定义优化器（选择其中一个即可）
from tensorflow.python.keras.optimizers import adam_v2, rmsprop_v2
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
optimizer = adam_v2.Adam()#Adam是一种自适应学习率的优化算法，它结合了动量和自适应学习率调整。
optimizer = SGD(learning_rate=0.001)#随机梯度下降是最基本的优化器之一，它在每次迭代中随机选择一个小批量样本进行梯度计算和参数更新。
optimizer = rmsprop_v2.RMSprop()#RMSprop也是一种自适应学习率算法，它通过计算梯度的平方根的移动平均值来自适应地调整学习率。

#编译模型
model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',#交叉熵，用于分类任务，通常用于测量分类模型输出的概率分布与真实标签之间的差异。
                metrics=['accuracy'])

#训练模型
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

NO_EPOCHS = 20#迭代20次
PATIENCE  = 10
VERBOSE   = 1#输出进程

# 设置动态学习率
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+NO_EPOCHS))
#LearningRateScheduler回调函数，在每个epoch结束时调用，更改学习率
#初始学习率为1e-3，之后每个epoch后都乘以0.99，使学习率逐渐减小，模型收敛加速

# 设置早停（在模型达到条件就终止训练，避免过拟合）
earlystopper = EarlyStopping(monitor='loss', #监测性能指标用的是损失函数
                             patience=PATIENCE, #在模型没有显示改进的情况下要等待的训练周期数，如果连续指定的周期数内没有损失的改进，训练将被终止
                             verbose=VERBOSE)


# 设置模型检查点
#模型检查点是一种训练策略，它用于在训练过程中定期保存模型的权重，以便在训练中途出现问题时或训练结束后能够恢复模型的最佳状态。
checkpointer = ModelCheckpoint('./day1-TensorFlow_CNN_cat_dog/cat_dog_best_model_cnn.h5',#保存模型
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

#训练模型
train_model  = model.fit(train_ds,
                  epochs=NO_EPOCHS,#迭代次数
                  verbose=1,
                  validation_data=val_ds,
                  callbacks=[earlystopper, checkpointer, annealer])#callbacks回调函数列表，用于在训练过程中执行额外的操作


#Accuracy和Loss可视化

import matplotlib.pyplot as plt

loss = train_model.history['loss']
acc = train_model.history['accuracy']
val_loss = train_model.history['val_loss']
val_acc = train_model.history['val_accuracy']
epoch = range(1, len(loss)+1)#创建一个包含迭代周期（epoch）数量的范围

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
#明显过拟合

#混淆矩阵可视化以及模型参数

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
        # 需要给图片增加一个维度，等于将单个图像变成一个批次
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
# 验证集的灵敏度为： 0.7335548172757476 
# 验证集的特异度为： 0.8669328010645376 
# 验证集的准确率为： 0.8001994680851063 
# 验证集的错误率为： 0.19980053191489366 
# 验证集的精确度为： 0.8466257668711656 
# 验证集的F1为： 0.7860448558205767 
# 验证集的MCC为： 0.605868266800025
    
train_pre   = []
train_label = []
for images, labels in train_ds:#这里可以取部分训练数据（.take(1)）生成混淆矩阵
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
# 训练集的灵敏度为： 0.9162870159453302 
# 训练集的特异度为： 0.9883124287343216 
# 训练集的准确率为： 0.9522792022792023 
# 训练集的错误率为： 0.047720797720797736 
# 训练集的精确度为： 0.9874194538201903 
# 训练集的F1为： 0.9505242947865898 
# 训练集的MCC为： 0.9069211202915147

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
#训练集的AUC值为： 0.9941917660381867 验证集的AUC值为： 0.8806599425733251
#没做数据增强，模型过拟合了

#保存模型
model.save('./day1-TensorFlow_CNN_cat_dog/cat_dog_best_model_cnn.h5')

#测试模型
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import img_to_array
from PIL import Image
import os, shutil, pathlib
label=np.array(["Dog","Cat"])#0、1赋值给标签

#载入模型
model = load_model('./day1-TensorFlow_CNN_cat_dog/cat_dog_best_model_cnn.h5')
#导入图片
image = image.load_img('C:/Users/maihuanzhuo/Desktop/python-test/Deep Learning/maomao.jpg')#手动修改路径，删除隐藏字符
plt.imshow(image)
plt.show()
image = image.resize((img_width,img_height))
image = img_to_array(image)
image = image/255#数值归一化，转为0-1
image = np.expand_dims(image,0)
print(image.shape)
# 使用模型进行预测
predictions = model.predict(image)
predicted_class = np.argmax(predictions)
# 打印预测的类别
print(label[predicted_class])
#明明是猫，但是预测是狗，模型失败
#需要做数据增强，避免模型过拟合，提高模型的泛化能力


from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, Reshape, Softmax
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
    RandomFlip("horizontal_and_vertical"),#随机地对图像进行水平垂直翻转和旋转
    RandomRotation(0.2),#旋转角度
    ])
def prepare(ds):
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds

train_ds = prepare(train_ds)

#构建cnn网络
model = Sequential([
                    Conv2D(filters=32, kernel_size=(3,3), padding='same',
                            input_shape=(img_width,img_height,3),  activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)),  
                    Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'),
                    MaxPooling2D(pool_size=(2, 2)), 
                    Conv2D(filters  =256, kernel_size=(3,3), padding='same', activation='relu'),
                    Flatten(),
                    Dense(512, activation='relu'),
                    Dense(2,activation='sigmoid')#二分类
                  ])
#打印模型结构
print(model.summary())

#编译模型
#负责优化模型的权重以减小损失函数的值
#定义优化器（选择其中一个即可）
from tensorflow.python.keras.optimizers import adam_v2, rmsprop_v2
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
optimizer = adam_v2.Adam()#Adam是一种自适应学习率的优化算法，它结合了动量和自适应学习率调整。

#编译模型
model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',#交叉熵，用于分类任务，通常用于测量分类模型输出的概率分布与真实标签之间的差异。
                metrics=['accuracy'])

#训练模型
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

NO_EPOCHS = 20#迭代20次
PATIENCE  = 10
VERBOSE   = 1#输出进程

# 设置动态学习率
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+NO_EPOCHS))
#LearningRateScheduler回调函数，在每个epoch结束时调用，更改学习率
#初始学习率为1e-3，之后每个epoch后都乘以0.99，使学习率逐渐减小，模型收敛加速

# 设置早停（在模型达到条件就终止训练，避免过拟合）
earlystopper = EarlyStopping(monitor='loss', #监测性能指标用的是损失函数
                             patience=PATIENCE, #在模型没有显示改进的情况下要等待的训练周期数，如果连续指定的周期数内没有损失的改进，训练将被终止
                             verbose=VERBOSE)


# 设置模型检查点
#模型检查点是一种训练策略，它用于在训练过程中定期保存模型的权重，以便在训练中途出现问题时或训练结束后能够恢复模型的最佳状态。
checkpointer = ModelCheckpoint('./day1-TensorFlow_CNN_cat_dog/cat_dog_best_model_cnn_data_augmentation.h5',#保存模型
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

#训练模型
train_model  = model.fit(train_ds,
                  epochs=NO_EPOCHS,#迭代次数
                  verbose=1,
                  validation_data=val_ds,
                  callbacks=[earlystopper, checkpointer, annealer])#callbacks回调函数列表，用于在训练过程中执行额外的操作


#Accuracy和Loss可视化
import matplotlib.pyplot as plt
loss = train_model.history['loss']
acc = train_model.history['accuracy']
val_loss = train_model.history['val_loss']
val_acc = train_model.history['val_accuracy']
epoch = range(1, len(loss)+1)#创建一个包含迭代周期（epoch）数量的范围

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
#没有过拟合了

#混淆矩阵可视化以及模型参数
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
        # 需要给图片增加一个维度，等于将单个图像变成一个批次
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
# 验证集的灵敏度为： 0.9049833887043189 
# 验证集的特异度为： 0.5968063872255489 特异度很差
# 验证集的准确率为： 0.7509973404255319 
# 验证集的错误率为： 0.2490026595744681 
# 验证集的精确度为： 0.6920731707317073 
# 验证集的F1为： 0.7843363086668587 
# 验证集的MCC为： 0.5275217029451771
    
train_pre   = []
train_label = []
for images, labels in train_ds:#这里可以取部分训练数据（.take(1)）生成混淆矩阵
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
# 训练集的灵敏度为： 0.8898063781321185 
# 训练集的特异度为： 0.6679019384264538 同样地，特异度差
# 训练集的准确率为： 0.7789173789173789 
# 训练集的错误率为： 0.2210826210826211 
# 训练集的精确度为： 0.7284382284382285
# 训练集的F1为： 0.8010766470135862 
# 训练集的MCC为： 0.572010758403984
#都不咋地，看来训练周期20太少了

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
#训练集的AUC值为： 0.8815505577229268 验证集的AUC值为： 0.8719840496194764

#保存模型
model.save('./day1-TensorFlow_CNN_cat_dog/cat_dog_best_model_cnn_data_augmentation.h5')

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
model = load_model('./day1-TensorFlow_CNN_cat_dog/cat_dog_best_model_cnn_data_augmentation.h5')
#导入图片
image = image.load_img('C:/Users/maihuanzhuo/Desktop/python-test/Deep Learning/maomao4.jpg')#手动修改路径，删除隐藏字符
plt.imshow(image)
plt.show()

image = image.resize((img_width,img_height))
image = img_to_array(image)
image = image/255#数值归一化，转为0-1
image = np.expand_dims(image,0)
print(image.shape)
# 使用模型进行预测
predictions = model.predict(image)
predicted_class = np.argmax(predictions)
# 打印预测的类别
print(label[predicted_class])
#预测出来是猫猫了，但第二张又预测为dog了，一半一半机率，看来模型泛化能力不咋地
