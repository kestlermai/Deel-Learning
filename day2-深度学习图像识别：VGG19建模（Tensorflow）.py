# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:36:10 2023

@author: maihuanzhuo
"""
'''
深度学习图像识别：VGG19建模（Tensorflow）
预训练模型：
预训练模型就像是一个精心制作的省力工具，它是在大量的数据上进行训练，然后将学习到的模型参数保存下来。
然后，我们可以直接使用这些参数，而不需要从头开始训练模型。这样可以节省大量的时间和计算资源。
迁移学习：
迁移学习就是将在一个任务上训练好的模型应用到另一个任务上。

VGG19 是一个深度卷积神经网络，由牛津大学的 Visual Geometry Group 开发并命名。
这个名字中的 "19" 代表着它的网络深度——总共有 19 层，这包括卷积层和全连接层。
这个模型在一系列的图像处理任务中表现得非常出色，包括图像分类、物体检测和语义分割等。VGG19 的一个显著特点是其结构的简洁和统一。
它主要由一系列的卷积层和池化层堆叠而成，其中每个卷积层都使用相同大小的滤波器，并且每两个卷积层之间都插入了一个池化层。
这样的设计使得 VGG19 既可以有效地提取图像特征，又保持了结构的简洁性。

VGG19 还有一个预训练的版本，这意味着我们可以直接使用在大量图像数据上训练好的模型，而不需要从头开始训练。
这大大节省了训练时间，同时也使得 VGG19 能够在数据较少的任务中表现得很好。
'''


from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, Reshape, Softmax, GlobalAveragePooling2D
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import adam_v2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, image_dataset_from_directory
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import RandomFlip, RandomRotation, RandomContrast, RandomZoom, RandomTranslation
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
#['cats', 'dogs']#这里的顺序要对上测试模型的label的顺序


#2.检查数据
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

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
  RandomRotation(0.2)
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

#导入VGG19
#获取预训练模型对输入的预处理方法
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras import Input
IMG_SIZE = (img_height, img_width, 3)
base_model = vgg19.VGG19(include_top=False, #是否包含顶层的全连接层
                         weights='imagenet')#weights参数设置为'imagenet'，表示使用在ImageNet数据集上预训练的权重。

#迁移学习主流程代码，开始利用预训练的VGG19创建模型
inputs = Input(shape=IMG_SIZE)#(150,150,3)
#模型
x = base_model(inputs, training=False) #参数不变化，将输入传递给预训练的VGG19模型，得到一个新的输出x
#全局池化
x = GlobalAveragePooling2D()(x)
#Dropout
x = Dropout(0.3)(x)#避免过拟合，dropout率为0.3
#Dense
x = Dense(512, activation='relu')(x)#dense层输出维度为512，relu增加非线性
#输出层
outputs = Dense(2,activation='sigmoid')(x)#输出二分类
#整体封装
model = Model(inputs, outputs)
#打印模型结构
print(model.summary())

# Model: "model"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_2 (InputLayer)         [(None, 150, 150, 3)]     0         
# _________________________________________________________________
# vgg19 (Functional)           (None, None, None, 512)   20024384  
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 512)               0         
# _________________________________________________________________
# dense_8 (Dense)              (None, 512)               262656    
# _________________________________________________________________
# dense_9 (Dense)              (None, 2)                 1026      
# =================================================================
# Total params: 20,288,066
# Trainable params: 20,288,066
# Non-trainable params: 0
# _________________________________________________________________
# None
####这个性能悲剧，对VGG19结构更改性能

#获取预训练模型对输入的预处理方法
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras import Input
from tensorflow.keras.layers import BatchNormalization
IMG_SIZE = (img_height, img_width, 3)
base_model = vgg19.VGG19(include_top=False, #是否包含顶层的全连接层
                         weights='imagenet')

#迁移学习主流程代码，开始利用预训练的VGG19创建模型
inputs = Input(shape=IMG_SIZE)
#模型
x = base_model(inputs, training=False) #参数不变化
#全局池化
x = GlobalAveragePooling2D()(x)
#BatchNormalization，添加批量归一化，有一定的正则化效果
x = BatchNormalization()(x)
#Dropout
x = Dropout(0.3)(x)#dropout率一般选0.5效果最好，即训练都会有50%的神经元被随机删除，以减少过拟合，当然处理过拟合方法也可以在dense层添加L1和L2正则化
#Dense
x = Dense(512)(x)#原本在这个全连接层这输出
#BatchNormalization
x = BatchNormalization()(x)#在全连接层再一次归一化
#激活函数
x = Activation('relu')(x)
#输出层
outputs = Dense(2)(x)
#BatchNormalization
outputs = BatchNormalization()(outputs)#在激活函数relu处理后再对数据归一化
#激活函数
outputs = Activation('sigmoid')(outputs)#变为二分类0和1结局
#整体封装
model = Model(inputs, outputs)
#打印模型结构
print(model.summary())

# Model: "model_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_6 (InputLayer)         [(None, 150, 150, 3)]     0         
# _________________________________________________________________
# vgg19 (Functional)           (None, None, None, 512)   20024384  
# _________________________________________________________________
# global_average_pooling2d_3 ( (None, 512)               0         
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 512)               2048      池化后归一化
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 512)               0         
# _________________________________________________________________
# dense_10 (Dense)             (None, 512)               262656    
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 512)               2048      全连接层归一化
# _________________________________________________________________
# activation (Activation)      (None, 512)               0         
# _________________________________________________________________
# dense_11 (Dense)             (None, 2)                 1026      
# _________________________________________________________________
# batch_normalization_3 (Batch (None, 2)                 8         relu非线性函数后归一化
# _________________________________________________________________
# activation_1 (Activation)    (None, 2)                 0         
# =================================================================
# Total params: 20,292,170
# Trainable params: 20,290,118
# Non-trainable params: 2,052
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

NO_EPOCHS = 30
PATIENCE  = 10
VERBOSE   = 1

# 设置动态学习率
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.99 ** (x+NO_EPOCHS))

# 设置早停
earlystopper = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)

# 设置模型检查点
checkpointer = ModelCheckpoint('./day2-TensorFlow_VGG19_cat_dog/cat_dog_best_model_vgg19.h5',
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
#过拟合

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
# 验证集的灵敏度为： 0.9215946843853821 
# 验证集的特异度为： 0.8749168330006654 
# 验证集的准确率为： 0.8982712765957447 
# 验证集的错误率为： 0.10172872340425532 
# 验证集的精确度为： 0.8806349206349207 
# 验证集的F1为： 0.9006493506493506 
# 验证集的MCC为： 0.7974003574086063

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
# 训练集的灵敏度为： 0.8932232346241458 
# 训练集的特异度为： 0.9315849486887116 
# 训练集的准确率为： 0.9123931623931624 
# 训练集的错误率为： 0.08760683760683763 
# 训练集的精确度为： 0.9289310038495706 
# 训练集的F1为： 0.9107272463347366 
# 训练集的MCC为： 0.8254008108187617

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
# 训练集的AUC值为： 0.973523217676226 验证集的AUC值为： 0.9659339129050869
#保存模型
model.save('./day2-TensorFlow_VGG19_cat_dog/cat_dog_best_model_vgg19.h5')

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
model = load_model('./day2-TensorFlow_VGG19_cat_dog/cat_dog_best_model_vgg19.h5')

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