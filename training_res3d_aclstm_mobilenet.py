'''
Descripttion: 
version: 
Author: @Yeqiwei
Date: 2020-09-03 20:29:35
LastEditors: @Yeqiwei
LastEditTime: 2020-10-12 17:12:08
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
sys.path.append("./networks")
import numpy as np
import tensorflow as tf
keras=tf.contrib.keras
l2=keras.regularizers.l2
K=tf.contrib.keras.backend
import inputs as data
from res3d_aclstm_mobilenet import res3d_aclstm_mobilenet
from callbacks import LearningRateScheduler 
from datagen import isoTrainImageGenerator, isoTestImageGenerator
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Used ConvLSTM Type
ATTENTIONX = 0
ATTENTIONI = 1
ATTENTIONO = 2

# Modality
RGB = 0
Depth = 1 
Flow = 2

cfg_type = ATTENTIONX
cfg_modality = RGB
if cfg_modality==RGB:
  str_modality = 'rgb'
elif cfg_modality==Depth:
  str_modality = 'depth'
elif cfg_modality==Flow:
  str_modality = 'flow'


nb_epoch = 1                   #训练次数
init_epoch = 0                #从0开始
seq_len = 32                  #代表每一个batch里的每一个数据包含32个，相当于一次取32个数据 
batch_size = 8                #批处理大小为8
num_classes = 5               #分类为5类
training_datalist = ".\\data.txt"      #数据来源的txt文件
testing_datalist = ".\\data.txt"
# './dataset_splits/IsoGD/valid_%s_list.txt'%str_modality
weight_decay = 0.00005        #防止过拟合（模型过复杂）的抑制参数，损失函数中加入了一项的系数，控制正则化项在损失函数中所占权重的
model_prefix = './models/'

dataset_name = 'isogr_%s'%str_modality    
weights_file = '%s/%s_weights.{epoch:02d}-{val_loss:.2f}.h5'%(model_prefix,dataset_name)     #训练数据保存在这个文件下
  
_,train_labels = data.load_iso_video_list(training_datalist)        #读取数据
train_steps = len(train_labels)/batch_size                          #训练次数
_,test_labels = data.load_iso_video_list(testing_datalist)
test_steps = len(test_labels)/batch_size
print('nb_epoch: %d - seq_len: %d - batch_size: %d - weight_decay: %.6f' %(nb_epoch, seq_len, batch_size, weight_decay))

def lr_polynomial_decay(global_step):           #更新模型学习率的函数
  learning_rate = 0.001
  end_learning_rate=0.000001
  decay_steps=train_steps*nb_epoch
  power = 0.9
  p = float(global_step)/float(decay_steps)
  lr = (learning_rate - end_learning_rate)*np.power(1-p, power)+end_learning_rate
  return lr
  # shape=(seq_len, 112, 112, 3),
inputs = keras.layers.Input(batch_shape=(batch_size, seq_len, 112, 112, 3))      #开始定义模型的输入
feature = res3d_aclstm_mobilenet(inputs, seq_len, weight_decay, cfg_type)       #模型的中间层即设计的核心
flatten = keras.layers.Flatten(name='Flatten')(feature)                         #Flatten层用来将输入“压平”，一维化
classes = keras.layers.Dense(num_classes, activation='linear', kernel_initializer='he_normal',
                    kernel_regularizer=l2(weight_decay), name='Classes')(flatten)     #全连接层
outputs = keras.layers.Activation('softmax', name='Output')(classes)    #将输入根据大小映射到01之间作为一个概率输出

model = keras.models.Model(inputs=inputs, outputs=outputs)    #模型的定义

for i in range(len(model.trainable_weights)):
  print(model.trainable_weights[i])              #模型的信息

optimizer = keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)    #随机梯度下降优化器，每次迭代使用一个样本来对参数进行更新。
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])      #定义损失函数 交叉熵

lr_reducer = LearningRateScheduler(lr_polynomial_decay,train_steps)     #调用函数调整学习率
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", verbose=1,
                   save_best_only=False,save_weights_only=True,mode='auto')    #，每一个epoch后将保存模型到weight_file
callbacks = [lr_reducer, model_checkpoint]

model.fit_generator(isoTrainImageGenerator(training_datalist, batch_size, seq_len, num_classes, cfg_modality),    #开始训练
          steps_per_epoch=train_steps,
          epochs=nb_epoch,
          verbose=1,
          callbacks=callbacks,
          validation_data=isoTestImageGenerator(testing_datalist, batch_size, seq_len, num_classes, cfg_modality),
          validation_steps=test_steps,
          initial_epoch=init_epoch,
          )
