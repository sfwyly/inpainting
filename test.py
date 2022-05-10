#encoding:utf-8

# 局部样式+局部对抗损失
import sys,os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers 
import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import math
import IPython.display as display
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
# import functools

class ChannelAttention(tf.keras.layers.Layer):
  def __init__(self,channels, ratio = 16,name="ChannelAttention",**kwargs):
    super(ChannelAttention, self).__init__(name=name,**kwargs)
    self.channels = channels

    self.avg_pool = layers.GlobalAveragePooling2D()
  
    self.max_pool = layers.GlobalMaxPooling2D()
  
    self.MLP = models.Sequential([
        layers.Dense(channels // ratio,activation="relu", use_bias=False),
        layers.Dense(channels, use_bias=False)
    ])
    
    self.sigmoid = tf.keras.activations.sigmoid

  def build(self, input_shape):
    # self.kernel = self.add_weight("kernel",shape=[int(input_shape[-1]),self.num_outputs])
    pass
  def call(self, input):
    avgout = self.MLP(self.avg_pool(input))
    maxout = self.MLP(self.max_pool(input))
    return self.sigmoid(avgout+maxout)
  def get_config(self):
    config = {"channels":self.channels}
    base_config = super(ChannelAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
class PatialConv(tf.keras.layers.Layer):
    def __init__(self, kernel=3,dilation_rate=1 ,stride=2,channels = 32,activation="relu",name="PatialConv",**kwargs):
        super(PatialConv,self).__init__(name=name,**kwargs)     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)
        self.slide_window = kernel**2
        # self.mask = mask #[1,h,w,1]
        self.kernel = kernel
        self.stride = stride
        self.channels = channels
        self.dilation_rate = dilation_rate
        self.dense1 = tf.keras.layers.Conv2D(filters=self.channels,kernel_size=self.kernel,dilation_rate = self.dilation_rate, kernel_initializer=tf.keras.initializers.Ones(),strides=self.stride, padding="same", use_bias=False, trainable=False)
        self.dense2 = tf.keras.layers.Conv2D(self.channels,kernel_size=self.kernel,dilation_rate = self.dilation_rate,kernel_initializer="he_normal",strides=self.stride, padding="same", use_bias=False, trainable=True)#
        self.bias = self.add_weight(shape=(self.channels,),initializer=tf.constant_initializer(0.0),trainable=True)
        self.relu = tf.keras.activations.relu
    def call(self, input,mask):
        update_mask = self.dense1(mask)
        mask_ratio = self.slide_window / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        output = self.dense2(input)#-self.bias
        output = output*mask_ratio
        # output = self.relu(output*mask_ratio+self.bias)
        
        return output , update_mask
    def get_config(self):
       config = {"kernel":self.kernel,"stride":self.stride,"channels":self.channels,"dilation_rate":self.dilation_rate}
       base_config = super(PatialConv, self).get_config()
       return dict(list(base_config.items()) + list(config.items()))


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self,shape=(32,32,256), kernel_size=3,name="SpatialAttention",**kwargs):
        super(SpatialAttention, self).__init__(name=name,**kwargs)
        # assert kernel_size in (3,7), "kernel size must be 3 or 7"
        # padding = 3 if kernel_size == 7 else 1
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(shape[2],kernel_size,padding="same",use_bias=False)
        self.sigmoid = tf.keras.activations.sigmoid
        self.bias = self.add_weight(shape=[shape[0],shape[1],shape[2]],initializer=tf.random_normal_initializer(),trainable = True)
        self.softmax_scale  = 10.
    def build(self, input_shape):
        # self.kernel = self.add_weight("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        pass
    def call(self, input,mask):#batch 32 32 128
        i_s = input.get_shape().as_list()

        # y = self.contextualAttention(x,mask)#batch 32 32 2
        #inpainting = tf.concat([y[...,0][...,tf.newaxis] for _ in range(i_s[3])],axis=-1)+self.bias
        
        #重构这里的注意力机制，为了 门控或者partial
        
        y = self.contextualAttention(input,mask)
        attention = self.sigmoid(self.conv(y)+self.bias)
        
        #a = self.sigmoid(input)/(self.sigmoid(input)+self.sigmoid(y))
        #new_feature = (input*mask + ((1-a)*y+a*input)*(1-mask))

        #avgout = tf.reduce_mean(new_feature, axis=-1, keepdims=True)
        #maxout= tf.reduce_max(new_feature, axis=-1, keepdims=True)
        #mask : 1 h w 1
        #x = layers.concatenate([avgout, maxout], axis=-1)
        #x = self.conv(x)
        #attention  = self.sigmoid(x)#注意力
        new_mask = tf.ones_like(mask) #门控机制 后续掩码直接为1
        return attention*y,new_mask
    def get_config(self):
        config = {"kernel_size":self.kernel_size}
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def contextualAttention(self,x,mask,kernel=3,stride=1,rate=2,fuse = True):
      x_s = x.get_shape().as_list()
    
      raw_x_patches = tf.image.extract_patches(x, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
      raw_x_patches = tf.reshape(raw_x_patches, [x_s[0], -1, kernel, kernel, x_s[3]]) 
      raw_x_patches = tf.transpose(raw_x_patches, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
      #缩放因子缩放前景，背景与mask
      f = tf.image.resize(x, [int(x_s[1]/rate),int(x_s[2]/rate)])#tf.image.resize_nearest_neighbor
      b = tf.image.resize(x, [int(x_s[1]/rate), int(x_s[2]/rate)])
      f_s = tf.shape(f)
      b_s = tf.shape(b)
    
      #mask = resize(mask, scale=1./self.rate, func=tf.image.resize_nearest_neighbor)
      mask = tf.image.resize(mask,[int(mask.shape[1]/rate),int(mask.shape[2]/rate)])
      m_s = mask.get_shape().as_list()
    
      x_patches = tf.image.extract_patches(b, [1,kernel,kernel,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
      x_patches = tf.reshape(x_patches, [b_s[0], -1, kernel, kernel, b_s[3]]) 
      x_patches = tf.transpose(x_patches, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        
      mask_patches = tf.image.extract_patches(mask, [1,kernel,kernel,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
      #mask_patches = tf.reshape(mask_patches, [1, -1, kernel, kernel, 1])
      mask_patches = tf.reshape(mask_patches, [m_s[0], -1, kernel, kernel, m_s[3]])
      mask_patches = tf.transpose(mask_patches, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw

      mm = tf.cast(tf.equal(tf.reduce_mean(mask_patches[0], axis=[0,1,2], keepdims=True), 1.), tf.float32)

      k = 3#fuse_k
      fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
      y = []
      for xi, wi,raw_wi, m in zip(tf.split(f, x_s[0], axis=0),tf.split(x_patches, x_s[0], axis=0),tf.split(raw_x_patches, x_s[0], axis=0),tf.split(mask_patches, x_s[0], axis=0)):
        mm = tf.cast(tf.equal(tf.reduce_mean(m[0], axis=[0,1,2], keepdims=True), 1.), tf.float32)
        wi = wi[0]
        xi = tf.cast(xi,tf.float32)
        wi = tf.cast(wi,tf.float32)
        
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")#1 h/rate w/rate (h/rate)*(w/rate)
        if fuse:
            yi = tf.reshape(yi, [1, b_s[1]*b_s[2], b_s[1]*b_s[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, b_s[1], b_s[2], b_s[1], b_s[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, b_s[1]*b_s[2], b_s[1]*b_s[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, b_s[2], b_s[1], b_s[2], b_s[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, b_s[1], b_s[2], b_s[1]*b_s[2]])
        
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*self.softmax_scale, 3)
        yi *=  mm  # mask
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], x_s[1:]], axis=0), strides=[1,rate,rate,1]) / rate**2
        y.append(yi)
      
      y = tf.concat(y, axis=0)
      #print("y ",y.shape)
      return y
_custom_objects = {
    "ChannelAttention" :  ChannelAttention,
   "SpatialAttention" : SpatialAttention,
   "PatialConv" :PatialConv
}

#generator = models.load_model("/usr/jjj/inpainting/generator_partial_attention.h5",custom_objects = _custom_objects)
#改 基于部分卷积的channel-wise注意力

class Partial_UNet():
    def __init__(self):
        print ('build Partial_UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        # print(target.shape,refer.shape)
#         print(target.get_shape(), refer.get_shape())
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1])
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self,train_bn=True):

        concat_axis = 3
        inputs = layers.Input(batch_shape = (6,256,256,3),name = "img_input")
        mask = layers.Input(batch_shape=(6,256,256,3),name="mask_input")
        
        def encoder_layer(img_in, mask_in, filters, kernel=3,strides= 1,rate=1,name="PatialConv", bn=True):
            conv, mask = PatialConv(kernel=kernel,dilation_rate=rate,stride=strides,channels = filters,name=name,trainable = True)(img_in, mask_in)
            #conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = layers.BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = layers.ReLU()(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0

        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel=3,strides= 1,rate=1,name="PatialConv", bn=True):
            up_img = layers.UpSampling2D(size=(2,2))(img_in)
            up_mask = layers.UpSampling2D(size=(2,2))(mask_in)
            concat_img = layers.Concatenate(axis=3)([e_conv,up_img])
            concat_mask = layers.Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PatialConv(kernel=kernel,dilation_rate=rate,stride=strides,channels = filters,name=name,trainable=True)(concat_img, concat_mask)
            #conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bn:
                conv = layers.BatchNormalization()(conv)
            conv = layers.LeakyReLU(alpha=0.2)(conv)
            return conv, mask
        
        
        conv1,mask1 = encoder_layer(inputs,mask,filters=64,kernel=3,strides=1,rate=1,name="PatialConv1",bn=False)
        conv1_2,mask1_2 = encoder_layer(conv1,mask1,filters=64,kernel=3,strides=2,rate=1,name="PatialConv1_2")
        
        conv2,mask2 = encoder_layer(conv1_2,mask1_2,filters=128,kernel=3,strides=1,rate=1,name="PatialConv2")
        conv2_2,mask2_2 = encoder_layer(conv2,mask2,filters=128,kernel=3,strides=2,rate=1,name="PatialConv2_2")

        conv3,mask3 = encoder_layer(conv2_2,mask2_2,filters=256,kernel=3,strides=1,rate=1,name="PatialConv3")
        conv3_2,mask3_2 = encoder_layer(conv3,mask3,filters=256,kernel=3,strides=2,rate=1,name="PatialConv3_2")
        
        conv3_3,mask3_3 = SpatialAttention(shape=(32,32,256),kernel_size=3,name="SpatialAttention")(conv3_2,mask3_2)
        
        conv4,mask4 = encoder_layer(conv3_3,mask3_3,filters=512,kernel=3,strides=1,rate=1,name="PatialConv4")
        conv4_2,mask4_2 = encoder_layer(conv4,mask4,filters=512,kernel=3,strides=2,rate=1,name="PatialConv4_2")

        conv5,mask5 = encoder_layer(conv4_2,mask4_2,filters=512,kernel=3,strides=1,rate=1,name="PatialConv5")
        conv5_2,mask5_2 = encoder_layer(conv5,mask5,filters=512,kernel=3,strides=2,rate=1,name="PatialConv5_2")
        
        conv6,mask6 = encoder_layer(conv5_2,mask5_2,filters=512,kernel=3,strides=1,rate=1,name="PatialConv6")
        conv6_2,mask6_2 = encoder_layer(conv6,mask6,filters=512,kernel=3,strides=2,rate=1,name="PatialConv6_2")
        
        
        #中间映射  全局处理 TODO 可能有效 可能无效
        conv7,mask7 = encoder_layer(conv6_2,mask6_2,filters=512,kernel=3,strides=1,rate=1,name="PatialConv7")
        

        
        conv11,mask11 = decoder_layer(conv7,mask7,conv6,mask6,filters=512,kernel=3,strides=1,rate=1,name="PatialConv11")
        conv11_2,mask11_2 = encoder_layer(conv11,mask11,filters=512,kernel=3,strides=1,rate=1,name="PatialConv11_2")

        conv12,mask12 = decoder_layer(conv11_2,mask11_2,conv5,mask5,filters=512,kernel=3,strides=1,rate=1,name="PatialConv12")
        conv12_2,mask12_2 = encoder_layer(conv12,mask12,filters=512,kernel=3,strides=1,rate=1,name="PatialConv12_2")
        
        conv13,mask13 = decoder_layer(conv12_2,mask12_2,conv4,mask4,filters=512,kernel=3,strides=1,rate=1,name="PatialConv13")
        conv13_2,mask13_2 = encoder_layer(conv13,mask13,filters=512,kernel=3,strides=1,rate=1,name="PatialConv13_2")
        
        #注意力机制 CBAM
        attention = ChannelAttention(channels=512,ratio=16,name="ChannelAttention")(conv13_2)
        conv13_3 = tf.transpose(tf.transpose(conv13_2,(1,2,0,3))*attention,(2,0,1,3))
        
        conv14,mask14 = decoder_layer(conv13_3,mask13_2,conv3,mask3,filters=256,kernel=3,strides=1,rate=1,name="PatialConv14")
        conv14_2,mask14_2 = encoder_layer(conv14,mask14,filters=256,kernel=3,strides=1,rate=1,name="PatialConv14_2")
        
        conv15,mask15 = decoder_layer(conv14_2,mask14_2,conv2,mask2,filters=128,kernel=3,strides=1,rate=1,name="PatialConv15")
        conv15_2,mask15_2 = encoder_layer(conv15,mask15,filters=128,kernel=3,strides=1,rate=1,name="PatialConv15_2")
        
        conv16,mask16 = decoder_layer(conv15_2,mask15_2,conv1,mask1,filters=64,kernel=3,strides=1,rate=1,name="PatialConv16")
        conv16_2,mask16_2 = encoder_layer(conv16,mask16,filters=64,kernel=3,strides=1,rate=1,name="PatialConv16_2",bn=False)

        outputs = layers.Conv2D(3, (1, 1),activation="sigmoid",name='img_outputs')(conv16_2)

        model = models.Model(inputs=[inputs,mask], outputs=outputs)#[conv1,conv2,conv3,conv4,conv5,up6,up7,up8,up9,

        return model
generator = Partial_UNet().create_model()
# generator.trainable = True
# generator.summary()
import os
import sys
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Dropout, LeakyReLU, BatchNormalization, Activation, Lambda,Concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
def build_pconv_unet(train_bn=True):      

        # INPUTS
        inputs_img = Input((256,256, 3), name='inputs_img')
        inputs_mask = Input((256, 256, 3), name='inputs_mask')
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size,name="PatialConv", bn=True):
            conv, mask = PatialConv(kernel=kernel_size,dilation_rate=1,stride=2,channels = filters,name=name)(img_in, mask_in)
            #conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
            if bn:
                conv = BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            conv = Activation('relu')(conv)
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        e_conv1, e_mask1 = encoder_layer(inputs_img, inputs_mask, 64, 7,name="PatialConv1", bn=False)
        e_conv2, e_mask2 = encoder_layer(e_conv1, e_mask1, 128, 5,name="PatialConv2")
        e_conv3, e_mask3 = encoder_layer(e_conv2, e_mask2, 256, 5,name="PatialConv3")
        e_conv4, e_mask4 = encoder_layer(e_conv3, e_mask3, 512, 3,name="PatialConv4")
        e_conv5, e_mask5 = encoder_layer(e_conv4, e_mask4, 512, 3,name="PatialConv5")
        e_conv6, e_mask6 = encoder_layer(e_conv5, e_mask5, 512, 3,name="PatialConv6")
        e_conv7, e_mask7 = encoder_layer(e_conv6, e_mask6, 512, 3,name="PatialConv7")
#         e_conv8, e_mask8 = encoder_layer(e_conv7, e_mask7, 512, 3,name="PatialConv8")
        
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size,name="PatialConv", bn=True):
            up_img = UpSampling2D(size=(2,2))(img_in)
            up_mask = UpSampling2D(size=(2,2))(mask_in)
            concat_img = Concatenate(axis=3)([e_conv,up_img])
            concat_mask = Concatenate(axis=3)([e_mask,up_mask])
            conv, mask = PatialConv(kernel=kernel_size,dilation_rate=1,stride=1,channels = filters,name=name)(concat_img, concat_mask)
            #conv, mask = PConv2D(filters, kernel_size, padding='same')([concat_img, concat_mask])
            if bn:
                conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.2)(conv)
            return conv, mask
            
#         d_conv9, d_mask9 = decoder_layer(e_conv8, e_mask8, e_conv7, e_mask7, 512, 3,name="PatialConv_up_1")
        d_conv10, d_mask10 = decoder_layer(e_conv7, e_mask7, e_conv6, e_mask6, 512, 3,name="PatialConv_up_2")
        d_conv11, d_mask11 = decoder_layer(d_conv10, d_mask10, e_conv5, e_mask5, 512, 3,name="PatialConv_up_3")
        d_conv12, d_mask12 = decoder_layer(d_conv11, d_mask11, e_conv4, e_mask4, 512, 3,name="PatialConv_up_4")
        d_conv13, d_mask13 = decoder_layer(d_conv12, d_mask12, e_conv3, e_mask3, 256, 3,name="PatialConv_up_5")
        d_conv14, d_mask14 = decoder_layer(d_conv13, d_mask13, e_conv2, e_mask2, 128, 3,name="PatialConv_up_6")
        d_conv15, d_mask15 = decoder_layer(d_conv14, d_mask14, e_conv1, e_mask1, 64, 3,name="PatialConv_up_7")
        d_conv16, d_mask16 = decoder_layer(d_conv15, d_mask15, inputs_img, inputs_mask, 3, 3,name="PatialConv_up_8", bn=False)
        outputs = Conv2D(3, 1, activation = 'sigmoid', name='outputs_img')(d_conv16)
        
        # Setup the model inputs / outputs
        model = Model(inputs=[inputs_img, inputs_mask], outputs=outputs)

        return model, inputs_mask
generator,inputs_mask = build_pconv_unet()


def getHoles(x, num = 1):
  direction = [[0,1],[0,-1],[1,0],[-1,0]]
  height,width = x[0] , x[1]
  #初始化所有数据
  result = []
  all = height * width
  for _ in range(num):
    t = 0 #孔洞数
    mask = np.ones((height,width),dtype=np.float32)
    x = np.random.randint(height/4,height/4*3)
    y = np.random.randint(width/4,width/4*3)
    for __ in range(all):
      forward = np.random.randint(4)
      
      while(x + direction[forward][0]<height/8 or x + direction[forward][0]>height/8*7 or y + direction[forward][1]<width/8 or y + direction[forward][1]>width/8*7):
        forward = np.random.randint(4)
        
      x = x + direction[forward][0]
      y = y + direction[forward][1]

      if(x < 0):
        x = 0
      elif(x > height-1):
        x = height-1
      
      if(y < 0):
        y = 0
      elif(y > width-1):
        y = width -1
      if(mask[x][y]==1):
        t = t+1
        mask[x][y] = 0.
      
    result.append((mask,t*1.0/all))
  return np.array(result)
def getRectHoles(x, num = 1):
  height,width = x[0] , x[1]
  #初始化所有数据
  result = []
  all = height * width
  for _ in range(num):
    t = 0 #孔洞数
    mask = np.ones((height,width),dtype=np.float32)

    x = np.random.randint(height/2)
    y = np.random.randint(width/2)

    up = np.random.randint(height/4,height/2)
    right = np.random.randint(width/4,width/2)
      
    mask[x:x+up,y:y+right] = 0.
    result.append((mask,up*right*1.0/all))
  return np.array(result)


def getHoles(image_shape,num):

    imageHeight,imageWidth = image_shape[0],image_shape[1]

    maxVertex = 10
    maxAngle = 30
    maxLength = 100
    maxBrushWidth = 20


    result = []

    for _ in range(num):

        mask = np.ones((imageHeight,imageWidth),dtype = np.float32)
        numVertex =5+ np.random.randint(maxVertex)

        

        for i in range(numVertex):

            startX = np.random.randint(imageHeight//4,imageHeight//4*3)
            startY = np.random.randint(imageWidth//4,imageWidth//4*3)

            for j in range(1+np.random.randint(5)):

                angle = 0.01 + np.random.randint(maxAngle)
                if(i%2==0):
                    angle = 2*np.pi - angle
                length = 5+np.random.randint(maxLength)
                brushWidth =5+ np.random.randint(maxBrushWidth)

                endX = (startX + length * np.sin(angle)).astype(np.int32)
                endY = (startY + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask,(startY,startX),(endY,endX),0.0,brushWidth)

                startX,startY = endX,endY
        result.append(mask)

    return np.array(result)
def getAllImagePath(name):
  path_root = pathlib.Path(name)
  result = list(path_root.glob("*.jpg"))
  
  return result

def getDataset(all_image_paths):

  train = []
  # labels = []
  for path in all_image_paths:
    path = str(path)
    #添加图片
    image = Image.open(path)
    # image = np.array(image)
    # image = image[21:189,10:179,:]
    # image = Image.fromarray(image)
    # image = image.resize((256,256),Image.BILINEAR)
    # image = image.resize((256,256*218//178),Image.BILINEAR)
    # image = np.array(image)[24:280,...]
    # train.append(image) #
    image = image.resize((256,256),Image.BILINEAR)
    image = np.array(image)
    if(len(image.shape)==3):
        train.append(image)
  return np.array(train)


#TODO  不要加随机误差
def multimask(image):
  # x,y,_ = image.shape
  return np.transpose(np.transpose(image,(2,0,1)) *mask  ,(1,2,0))

#保存读取mask
def saveMask(mask,name,prefix="/content/drive/My Drive/data/inpainting/mask/"):
  mask = mask[:,:,np.newaxis]
  mask = np.concatenate([mask,mask,mask],axis=-1)*255
  image = Image.fromarray(np.uint8(mask))
  image.save(prefix+name+".jpg")
def readMask(name,prefix ="/content/drive/My Drive/data/inpainting/mask/"):
  image = Image.open(prefix+name+".jpg")
  image = np.array(image)/255
  # print(image.shape)
  return image[:,:,0]
def getMaskList(prefix,files_num):
  #处理 几位数 添0
  file_list = [str(_) for _ in range(1,files_num+1)]
  mask_list = []
  for _ in file_list:
    mask = readMask(_,prefix)

    mask = np.array(mask,np.float32)
    for i in range(256):
      for j in range(256):
        if(mask[i,j]<0.5):
          mask[i,j] = 0
        else:
          mask[i,j] =1
    mask_list.append(mask)
    
  return np.array(mask_list)
def getPSNR(image , true_image):
  height,width,_ = image.shape
  channel_mse = np.sum((image-true_image)**2,axis = (0,1))/(height*width)
  mse = np.mean(channel_mse)
  Max = 1.0 #最大图像
  PSNR = 10.*np.log10(Max**2/mse) #峰值信噪比

  return PSNR
def l1_loss(y_pred,y_true,mask_list):
  # print("l1")
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true = tf.cast(y_true,dtype=tf.float32)
  return 100.*tf.reduce_mean(tf.abs(y_pred - y_true))+6.*tf.reduce_mean(tf.abs(y_pred*(1-mask_list) - y_true*(1-mask_list)))

def getMaskListPaths(name):
  path_root = pathlib.Path(name)
  mask_paths = list(path_root.glob("*.png"))
 
    
  return np.array(mask_paths)
def getMaskList(mask_paths):
  mask_list = []
  for path in mask_paths:
    path = str(path)
    #添加图片
    image = Image.open(path)
    image = image.resize((256,256),Image.BILINEAR)
    image  =np.array(image)/255
    for i in range(256):
      for j in range(256):
        if(image[i,j]<0.5):
          image[i,j]=0
        else:
          image[i,j]=1
    mask = (1-np.array(image))[...,np.newaxis]
    mask = np.concatenate([mask,mask,mask],axis=-1)
    mask_list.append(mask)
  return np.array(mask_list)
if(__name__=="__main__"):

  all_image_paths = getAllImagePath("F:/place2/data_256/z/zen_garden/")#F:/place2/data_256/z/zen_garden F:/celeA HQ/img_align_celeba/
  # np.random.shuffle(all_image_paths)
  print(len(all_image_paths))

  dataset = getDataset(all_image_paths[100:110]) /255.
  # print(dataset.shape)
  # plt.imshow(dataset[0])
  # plt.show()
  mask_paths = getMaskListPaths("F:/mask/testing_mask_dataset/")
  mask_list = getMaskList(mask_paths[9990:10000])[:10]

  print(mask_list.shape)

  # mask_list = getHoles((256,256),1)[...,np.newaxis]
  x = dataset*mask_list#mask[...,np.newaxis]#+(1-mask[...,np.newaxis])


  # 加载模型
  # generator.load_weights("E:/cnn/generator_partial_attention_place2.h5")
  # print((generator.get_layer('PatialConv1').output)[1])
  # model = models.Model(generator.input,generator.get_layer('PatialConv1').output)
  # model.summary()
  i=0
  result =generator.predict([x[i:i+6,...],mask_list[:6]])#mask[np.newaxis,...,np.newaxis]
  # for j in range(6):
  #   print(np.mean(np.abs(dataset[i+j]-result[j])))
  #   # print(getPSNR(dataset[i+j],result[j]))
  # # print(np.mean(np.abs(dataset[:6]-result)))

  # plt.figure(1)
  # plt.subplot(321)
  # plt.imshow(result[0][0])
  # plt.subplot(322)
  # plt.imshow(result[1][0] )
  # plt.subplot(323)
  # plt.imshow(result[2][0])
  # plt.subplot(324)
  # plt.imshow(result[3][0])
  # plt.subplot(325)
  # plt.imshow(result[4][0])
  # plt.subplot(326)
  # plt.imshow(result[5][0]) #*0.5 +0.5
  # plt.show()


  # print(l1_loss(result[:6],dataset[i:i+6],mask_list))
  plt.figure(1)
  plt.subplot(321)
  plt.imshow(x[i+0])
  plt.subplot(322)
  plt.imshow(result[0] )
  plt.subplot(323)
  plt.imshow(x[i+1])
  plt.subplot(324)
  plt.imshow(result[1])
  plt.subplot(325)
  plt.imshow(x[i+2])
  plt.subplot(326)
  plt.imshow(result[2]) #*0.5 +0.5
  plt.show()
  # plt.imshow(generator.get_layer("PatialConv1")[1][0])
  # plt.show()