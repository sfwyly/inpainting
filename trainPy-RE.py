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
# import matplotlib.pyplot as plt

import math
# import IPython.display as display
import cv2

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
# import functools
def getHoles(image_shape,num):

    imageHeight,imageWidth = image_shape[0],image_shape[1]

    maxVertex = 20
    maxAngle = 30
    maxLength = 100
    maxBrushWidth = 20


    result = []

    for _ in range(num):

        mask = np.ones((imageHeight,imageWidth),dtype = np.float32)
        numVertex =1+ np.random.randint(maxVertex)

        

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
  result = list(path_root.glob("*"))
  
  return result

def getDataset(all_image_paths):
    train = []
    # labels = []
    for path in all_image_paths:
        path = str(path)
        # 添加图片
        image = Image.open(path)
        # image = image.resize((256, 256), Image.BILINEAR)  # *218//178
        image = np.array(image)
        if(image.shape[-1] !=3):
            continue
        h, w, c = image.shape
        if (True):
            if (h < w):
                offset = int((w - h) / 2)
                image = image[:, offset:h + offset, :]
            else:
                offset = int((h - w) / 2)
                image = image[offset:w + offset, :, :]
            image = Image.fromarray(np.uint8(image))
            image = np.array(image.resize((256,256), Image.BILINEAR))
        train.append(image)  # [24:280,...])
    return np.array(train)
def getDataset(all_image_paths): #celebA

  train = []
  # labels = []
  for path in all_image_paths:
    path = str(path)
    #添加图片
    image = Image.open(path)
    image = image.resize((256,256),Image.BILINEAR)#*218//178
    image = np.array(image)
    if(len(image.shape) != 3):
        continue
    train.append(image)#[24:280,...])
  return np.array(train)
# def getDataset(all_image_paths):#paris

#   train = []
#   # labels = []
#   for path in all_image_paths:
#     path = str(path)
#     #添加图片
#     image = Image.open(path)
#     #image = image.resize((256,256*218//178),Image.BILINEAR)
#     #image = image.resize((256,256),Image.BILINEAR)
#     image = np.array(image)
#     train.append(image[140:140+256,340:340+256])
#   return np.array(train)
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self,shape=(32,32,256), kernel_size=3,name="SpatialAttention",**kwargs):
        super(SpatialAttention, self).__init__(name=name,**kwargs)
        # assert kernel_size in (3,7), "kernel size must be 3 or 7"
        # padding = 3 if kernel_size == 7 else 1
        self.kernel_size = kernel_size
        #多尺度门控策略
        self.conv = layers.Conv2D(shape[2],kernel_size=3,strides =1,dilation_rate=1,padding="same")
        self.conv1 = layers.Conv2D(shape[2],1,strides =1,dilation_rate=1,padding="same")
        self.conv2 = layers.Conv2D(shape[2],3,strides =1,dilation_rate=1,padding="same")
        self.sigmoid = tf.keras.activations.sigmoid
        self.bias = self.add_weight(shape=(shape[0],shape[1],shape[2]),initializer= tf.random_normal_initializer(),trainable = True)
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
        #连接后卷积
        y = tf.concat([y,input],axis=-1)
        
        gate1 = self.sigmoid(self.conv1(y))
        gate2 = self.sigmoid(self.conv2(y))
        gate = (gate1+gate2)/2
        
        new_mask = tf.ones_like(mask) #门控机制 后续掩码直接为1
        return gate*self.conv(y),new_mask
    def get_config(self):
        config = {"kernel_size":self.kernel_size}
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def contextualAttention(self,x,mask,kernel=3,stride=1,rate=1,fuse = True):
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
class Bottleneck(layers.Layer):
    
    def __init__(self, output_dim,strides = 1,training = True, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv1 = layers.Conv2D(output_dim//4,kernel_size = 1,padding = "same",use_bias =False) 
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_dim//4,kernel_size = 3,strides = strides,padding = "same",use_bias = False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(output_dim,kernel_size = 1,padding = "same",use_bias = False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.training = training
    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)  
    
    def call(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out,training = self.training)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out,training = self.training)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out,training = self.training)
        
        out =  out + residual
        out = self.relu(out)
        
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#循环模块叠加两层
# class RFRModule(layers.Layer):
    
#     def __init__(self, layer_size = 6,in_channels = 64, **kwargs):
#         super(RFRModule, self).__init__(**kwargs)
#         self.freeze_enc_bn = False
#         self.layer_size = layer_size
#         self.encoder_blocks = {}
#         self.decoder_blocks = {}
#         self.in_channels = in_channels
        
#         for _ in range(5):
#             out_channels = self.in_channels*2**(_+1)
#             if(out_channels>512):#最大支持512channels
#                 out_channels = 512
#             block = models.Sequential([layers.Conv2D(out_channels,kernel_size = 3,strides = 2,padding="same",use_bias = False)
#                     ,layers.BatchNormalization(),layers.ReLU()])
#             name = "enc_{:d}".format(_+1)
#             self.encoder_blocks[name] = block
        
#         # 运行在 16 * 16 * 512
# #         for _ in range(3,6):
# #             block  = models.Sequential([layers.Conv2D(out_channels,kernel_size = 3,strides = 1,padding="same",dilation_rate=2,use_bias = False)
# #                     ,layers.BatchNormalization(),layers.ReLU()])
# #             name = "low_enc_{:d}".format(_+1)
# #             self.encoder_blocks[name] = block
# #         for _ in range(5,3,-1):
# #             block = models.Sequential([layers.Conv2D(out_channels,kernel_size = 3,strides = 1,padding="same",dilation_rate=2,use_bias = False)
# #                     ,layers.BatchNormalization(),layers.LeakyReLU(0.2)])
# #             name = "low_dec_{:d}".format(_)
# #             self.decoder_blocks[name] = block
#         # 运行在 4 * 4 * 512
#         for _ in range(5,8):
#             block  = models.Sequential([layers.Conv2D(out_channels,kernel_size = 3,strides = 1,padding="same",dilation_rate=1,use_bias = False)
#                     ,layers.BatchNormalization(),layers.ReLU()])
#             name = "high_enc_{:d}".format(_+1)
#             self.encoder_blocks[name] = block
#         for _ in range(7,5,-1):
#             block = models.Sequential([layers.Conv2D(out_channels,kernel_size = 3,strides = 1,padding="same",dilation_rate=1,use_bias = False)
#                     ,layers.BatchNormalization(),layers.LeakyReLU(0.2)])
#             name = "high_dec_{:d}".format(_)
#             self.decoder_blocks[name] = block
        
#         #知识一致注意力模块 TODO
#         #self.att = AttentionModule(512)
        
#         # 解码器过程
        
#         #高层解码器
#         self.decoder_blocks["dec_5"] = models.Sequential([
#             layers.Conv2DTranspose(8*self.in_channels,kernel_size = 4,strides = 2,padding="same",use_bias = False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ])
#         self.decoder_blocks["dec_4"] = models.Sequential([
#             layers.Conv2DTranspose(8*self.in_channels,kernel_size = 4,strides = 2,padding="same",use_bias = False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ])
#         #共用解码器
#         # 1024 -> 512
#         self.decoder_blocks["dec_3"] = models.Sequential([
#             layers.Conv2DTranspose(8*self.in_channels,kernel_size = 4,strides = 2,padding="same",use_bias = False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ])
#         # 768 -> 256
#         self.decoder_blocks["dec_2"] = models.Sequential([
#             layers.Conv2DTranspose(4*self.in_channels,kernel_size = 4,strides = 2,padding="same",use_bias = False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ])
#         # 384 ->64
#         self.decoder_blocks["dec_1"] = models.Sequential([
#             layers.Conv2DTranspose(self.in_channels,kernel_size = 4,strides = 2,padding="same",use_bias = False),
#             layers.BatchNormalization(),
#             layers.LeakyReLU(0.2)
#         ])
            
#     def build(self, input_shape):
#         super(RFRModule, self).build(input_shape)  

#     def call(self, input,mask,flag = True):
#         # flag 代表是否进行 更深层还是浅层
#         h_dict = {}
#         h_dict["h_0"] = input
#         h_key_prev = "h_0"
        
#         #下采样到 16 * 16
#         if(flag):
            
#             h1 = self.encoder_blocks["enc_1"](input)  
#             h2 = self.encoder_blocks["enc_2"](h1)
#             h3 = self.encoder_blocks["enc_3"](h2) #16 * 16 * 512
            
#             h4 = self.encoder_blocks["low_enc_4"](h3)
#             h5 = self.encoder_blocks["low_enc_5"](h4)
#             h6 = self.encoder_blocks["low_enc_6"](h5)
            
#             u_h5 = self.decoder_blocks["low_dec_5"](tf.concat([h6,h5],axis = -1))

#             u_h4 = self.decoder_blocks["low_dec_4"](tf.concat([u_h5,h4],axis = -1))
            
#             u_h3 = self.decoder_blocks["dec_3"](tf.concat([u_h4,h3],axis = -1))# 32 * 32 * 256
#             u_h2 = self.decoder_blocks["dec_2"](tf.concat([u_h3,h2],axis = -1))
#             u_h1 = self.decoder_blocks["dec_1"](tf.concat([u_h2,h1],axis = -1))
            
#             return u_h1
        
#         #如果是深层 下采样到4*4
#         h1 = self.encoder_blocks["enc_1"](input)  
#         h2 = self.encoder_blocks["enc_2"](h1)
#         h3 = self.encoder_blocks["enc_3"](h2) #16 * 16 * 512
#         h4 = self.encoder_blocks["enc_4"](h3)
#         h5 = self.encoder_blocks["enc_5"](h4) #4 * 4 * 512
        
#         h6 = self.encoder_blocks["high_enc_6"](h5)
#         h7 = self.encoder_blocks["high_enc_7"](h6)
#         h8 = self.encoder_blocks["high_enc_8"](h7)
        
#         u_h7 = self.decoder_blocks["high_dec_7"](tf.concat([h8,h7],axis = -1))
#         u_h6 = self.decoder_blocks["high_dec_6"](tf.concat([u_h7,h6],axis = -1))
#         u_h5 = self.decoder_blocks["dec_5"](tf.concat([u_h6,h5],axis = -1))
#         u_h4 = self.decoder_blocks["dec_4"](tf.concat([u_h5,h4],axis = -1))

#         u_h3 = self.decoder_blocks["dec_3"](tf.concat([u_h4,h3],axis = -1))# 32 * 32 * 256
#         u_h2 = self.decoder_blocks["dec_2"](tf.concat([u_h3,h2],axis = -1))
#         u_h1 = self.decoder_blocks["dec_1"](tf.concat([u_h2,h1],axis = -1))
        
#         return u_h1
    
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

# 最新 SK-Net

class PartialConv(layers.Layer):
    def __init__(self, kernel=3,dilation_rate=1 ,strides=2,in_channels = 64,out_channels = 64,activation="relu",flag = True,mul = True,training = True,**kwargs):
        super(PartialConv,self).__init__(**kwargs) 
        self.slide_window = kernel**2
        self.kernel = kernel
        self.strides = strides
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dilation_rate = dilation_rate
        self.dense2 = tf.keras.layers.Conv2D(self.out_channels,kernel_size=self.kernel,dilation_rate = self.dilation_rate,strides=self.strides, padding="same", use_bias=False, trainable=True)#
        self.flag = flag
        self.mul = mul
        #self.bias = self.add_weight(shape=(self.channels,),initializer=tf.constant_initializer(0.0),trainable=True)
        if(self.flag):
            self.leaky_relu = tf.nn.leaky_relu
            self.bn = layers.BatchNormalization()
        self.weights_updater =  tf.ones((self.kernel,self.kernel,self.in_channels,self.out_channels))
        self.training = training
    def call(self, input,mask):
        
        update_mask = tf.nn.conv2d(mask,self.weights_updater,strides = self.strides,padding="SAME")
        mask_ratio = (self.slide_window*self.in_channels) / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask
        
        output = self.dense2(input)#-self.bias
        if(self.mul):
            output = output*mask_ratio
            # output = self.relu(output*mask_ratio+self.bias)
            output = output*update_mask
        if(self.flag):
            return self.leaky_relu(self.bn(output,training=self.training)), update_mask
        return output , update_mask

class AtnConv(layers.Layer):
    def __init__(self,input_channels = 256, output_channels = 256, groups = 4, ksize = 3, stride = 1, rate = 1,softmax_scale = 10,fuse = True, rates = [1,2,4,8]):
        super(AtnConv,self).__init__()
        
        self.kernel = ksize
        self.strides = stride
        self.rate = rate
        self.softmax_scale = softmax_scale
        self.groups = groups
        self.fuse = fuse
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = tf.ones(1)
        #if(self.fuse):
        #    self.group_blocks = []
        #    for i in range(groups):
        #        self.group_blocks.append(
        #            models.Sequential([layers.Conv2D(output_channels//groups,kernel_size=3,dilation_rate=rates[i],padding="same"),layers.ReLU()])
        #        )
    #x1 lower-level  x2: high-level
    def call(self,x1,x2,mask):
        
        x1s = x1.shape
        x2s = x2.shape
        bz,height,width,channels = x1s
        kernel = 2*self.rate
        raw_w = tf.image.extract_patches(x1, [1,self.kernel,self.kernel,1], [1,self.rate*self.strides,self.rate*self.strides,1], [1,1,1,1], padding='SAME')
        raw_w = tf.reshape(raw_w, [x1s[0], -1, self.kernel, self.kernel, x1s[-1]]) 
        raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
        raw_w_groups = tf.split(raw_w,x1s[0],axis = 0)
        
        f_groups  =tf.split(x2,x2s[0],axis= 0)
       
        w = tf.image.extract_patches(x2, [1,self.kernel,self.kernel,1], [1,self.strides,self.strides,1], [1,1,1,1], padding='SAME')
        w = tf.reshape(w, [x2s[0], -1, self.kernel, self.kernel, x2s[-1]]) 
        w = tf.transpose(w, [0, 2, 3, 4, 1])
        w_groups = tf.split(w,x2s[0],axis= 0)

        ms = mask.shape
        if(mask is not None):
            mask = tf.image.resize(mask,x2s[1:3],"bilinear")
        else:
            mask = tf.zeros((x2s[0],x2s[1],x2s[2],x2s[3]))
        m = tf.image.extract_patches(mask, [1,self.kernel,self.kernel,1], [1,self.strides,self.strides,1], [1,1,1,1], padding='SAME')
        m = tf.reshape(m, [ms[0], -1, self.kernel, self.kernel, ms[-1]]) 
        m = tf.transpose(m, [0, 2, 3, 4, 1])# b k k c hw
        m = tf.cast(tf.equal(tf.reduce_mean(m, axis=[1,2,3], keepdims=True), 1.), tf.float32)
        mm = tf.squeeze(m,axis = 1) #b 1 1 hw
        mm_groups = tf.split(mm,ms[0],axis= 0)
        
        y = []
        att_score = []
        scale = self.softmax_scale
        for i in range(bz):
            
            xi,wi,raw_wi,mi = x2[i:i+1],w[i],raw_w[i],mm[i]#k k c hw
            escape_NaN = 1e-4
            wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), escape_NaN)#k k c hw
            
            yi = tf.nn.conv2d(xi, wi_normed, strides=1, padding="SAME")#1 h w  hw
            
            yi = tf.reshape(yi,[1,x2s[1],x2s[2],x2s[1]//self.strides*x2s[2]//self.strides])
            
            yi = tf.nn.avg_pool2d(yi,3,1,padding="SAME")*9
            
            attention_scores = tf.nn.softmax(yi*scale,axis = -1)

            if(self.att_scores_prev is not None):
                attention_scores = (self.att_scores_prev[i:i+1]*self.masks_prev[i:i+1]+attention_scores * (tf.abs(self.ratio)+1e-7))/(self.masks_prev[i:i+1]+(tf.abs(self.ratio)+1e-7))
                #pass
            att_score.append(attention_scores)
            yi = tf.nn.conv2d_transpose(attention_scores,raw_wi,tf.concat([[1], x1s[1:]], axis=0),strides=[1,self.rate,self.rate,1],padding="SAME")#/4.
            y.append(yi)
        
        y = tf.concat(y,axis = 0)
        self.att_scores_prev = tf.reshape(tf.concat(att_score,axis =0),[bz,height,width,height*width])   
        self.masks_prev = tf.reshape(mm,[bz,1,1,height * width])
        #if(self.fuse):
        #    tmp = []
        #    for i in range(self.groups):
        #        tmp.append(self.group_blocks[i](y))
        #    y = tf.concat(tmp,axis= -1)
        
        return y
# 集成 sk-net
class AttentionModule(layers.Layer):
    
    def __init__(self,inchannel = 256,patch_size_list = [1,2,4,8],**kwargs):
        super(AttentionModule, self).__init__(**kwargs)
        self.att = AtnConv(256,256)
        self.num_of_modules = len(patch_size_list)
        self.combiner = layers.Conv2D(inchannel,kernel_size = 1,padding="same")
        self.blocks = []
        for rate in patch_size_list:
            self.blocks.append(layers.Conv2D(inchannel,kernel_size = 3,activation = "relu",dilation_rate = rate,padding="same"))
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        self.mlp1 = layers.Dense(inchannel//8,activation = "relu")
        self.mlp2 = layers.Dense(inchannel*len(patch_size_list))
        
        #self.dropout1 = layers.Dropout(0.5)
        #self.dropout2 = layers.Dropout(0.5)
    def build(self, input_shape):
        super(AttentionModule, self).build(input_shape)  
        
    def call(self,x1,x2,mask):
        bz,h,w,c = x1.shape
        outputs = self.att(x1,x2, mask)
        #return outputs
        patch_list = []
        for block in self.blocks:
            patch_list.append(block(outputs)[tf.newaxis,...])
        patch_block = tf.concat(patch_list,axis = 0)
        block_sum = tf.reduce_sum(patch_block,axis = 0)
        
        avg_feat = self.avg_pool(block_sum)
        max_feat = self.max_pool(block_sum)
        avg_feat = self.mlp1(avg_feat)
        #avg_feat = self.dropout1(avg_feat)
        avg_feat = self.mlp2(avg_feat)
        
        max_feat = self.mlp1(max_feat)
        #max_feat = self.dropout2(max_feat)
        max_feat = self.mlp2(max_feat)
        
        feat = avg_feat + max_feat
        
        feat = tf.reshape(feat,[bz,1,1,c,self.num_of_modules])
        feat = tf.transpose(feat,[4,0,1,2,3])
        feat = tf.nn.softmax(feat,axis = 0)
        outputs = tf.reduce_sum(patch_block*feat,axis = 0)
        
        outputs = tf.concat([outputs,x2],axis = -1)
        outputs = self.combiner(outputs)
        
        return outputs

#是不是太深了，弄浅点
class PFRNet(models.Model):
    def __init__(self):
        super(PFRNet, self).__init__()
        
        self.Pconv01 = PartialConv(in_channels = 3,out_channels = 64,kernel = 7,strides = 2,flag = False)
        self.Pconv02 = PartialConv(in_channels = 64,out_channels = 64,kernel = 7,strides = 1)
        
        self.Pconv11 = PartialConv(in_channels = 64,out_channels = 128,kernel = 7,strides = 2)
        self.Pconv12 = PartialConv(in_channels = 128,out_channels = 256,kernel = 5,strides = 2,flag = False)
        
        self.u_Pconv11 = PartialConv(in_channels = 384,out_channels = 128,kernel = 3,strides = 1)
        self.u_Pconv10 = PartialConv(in_channels = 192,out_channels = 64,kernel = 3,strides = 1)
        
        self.up= layers.UpSampling2D(size = (2,2))
        
        self.Pconv23 = PartialConv(in_channels = 256,out_channels = 512,kernel = 3,strides = 2,mul = False)
        self.Pconv24 = PartialConv(in_channels = 512,out_channels = 512,kernel = 3,strides = 2,mul = False)
        
        self.u_Pconv22 = PartialConv(in_channels = 768,out_channels = 256,kernel = 3,strides = 1,mul = False)
        self.u_Pconv23 = PartialConv(in_channels = 1024,out_channels = 512,kernel = 3,strides = 1,mul = False)
        
        #三层定义
        self.Pconv35 = PartialConv(in_channels = 512,out_channels = 512,kernel = 3,strides = 2,mul = False)
        self.Pconv36 = PartialConv(in_channels = 512,out_channels = 512,kernel = 3,strides = 2,mul = False)

        self.u_Pconv34 = PartialConv(in_channels = 1024,out_channels = 512,kernel = 3,strides = 1,mul = False)
        self.u_Pconv35= PartialConv(in_channels = 1024,out_channels = 512,kernel = 3,strides = 1,mul = False)

        self.atn = AttentionModule()
        
        self.conv = layers.Conv2D(64,kernel_size = 1,strides = 1,padding="same",use_bias = False)
        self.conv2 = layers.Conv2D(64,kernel_size = 1,activation="sigmoid",strides = 1,padding="same",use_bias = False)
        
        #尾部定义
        self.Tconv = layers.Conv2DTranspose(64,kernel_size = 4,strides = 2,padding="same",use_bias = False)
        self.bn3 = layers.BatchNormalization()
        self.tail1 = PartialConv(in_channels = 67,out_channels = 32,kernel = 3,strides = 1,flag = False)
        self.tail2 = Bottleneck(output_dim = 32,strides = 1)
        self.out = layers.Conv2D(3,kernel_size = 3,activation="sigmoid",strides = 1,padding="same")

    def call(self,input,mask):
        
        #逐步渐进补全策略 先通过一个低层循环 在经过高层循环处理
        
        self.atn.att.att_scores_prev = None
        self.atn.att.masks_prev = None
        #前面的一次下采样
        x, m = self.Pconv01(input,mask)     # 128 * 128 * 64
        x = tf.nn.relu(x)
        x , m = self.Pconv02(x, m) # 128 * 128 * 64
        
        #一阶段
        x1_0,m1_0 = x,m
        x1_1, m1_1 = self.Pconv11(x1_0,m1_0)     # 64 * 64 * 128
        x1_1 = tf.nn.relu(x1_1)
        x1_2, m1_2 = self.Pconv12(x1_1,m1_1)     # 32 * 32 *256
        x1_2 = tf.nn.relu(x1_2)
        
        u_x1_1,u_m1_1 = self.u_Pconv11(tf.concat([self.up(x1_2),x1_1],axis= -1),tf.concat([self.up(m1_2),m1_1],axis = -1)) # in 64*64*(256+128) out 64*64*128
        u_x1_0,u_m1_0 = self.u_Pconv10(tf.concat([self.up(u_x1_1),x1_0],axis= -1),tf.concat([self.up(u_m1_1),m1_0],axis = -1)) # in 128*128*(128+64) out 128*128*64
        
        #二层
        x2_0,m2_0 = u_x1_0*u_m1_0,u_m1_0
        #x2_0,m2_0 = u_x1_0,self.up(self.up(m1_2))[...,:64]
        #x2_0 = (x1_0*m1_0 +x2_0*m2_0)/(m1_0+m2_0+10e-7)
        
        x2_1, m2_1 = self.Pconv11(x2_0,m2_0)     # 64 * 64 * 128
        x2_1 = tf.nn.relu(x2_1)
        x2_2, m2_2 = self.Pconv12(x2_1,m2_1)     # 32 * 32 *256
        x2_2 = tf.nn.relu(x2_2)
        #两种方案 是通过x1_2重构x2_2还是怎么样处理
        x2_3 = self.atn(x1_2,x2_2,m2_2)#*(1 - m1_2) + x1_2 * m1_2
        
        x2_3, m2_3 = self.Pconv23(x2_3,m2_2)     # 16 * 16 * 512
        x2_3 = tf.nn.relu(x2_3)
        x2_4, m2_4 = self.Pconv24(x2_3,m2_3)     # 8 * 8 * 512
        x2_4 = tf.nn.relu(x2_4)
        
        u_x2_3,u_m2_3 = self.u_Pconv23(tf.concat([self.up(x2_4),x2_3],axis= -1),tf.concat([self.up(m2_4),m2_3],axis = -1)) # in 16*16*(512 + 512) out 16 * 16 * 512
        u_x2_2,u_m2_2 = self.u_Pconv22(tf.concat([self.up(u_x2_3),x2_2],axis= -1),tf.concat([self.up(u_m2_3),m2_2],axis = -1)) # in 32 * 32 * (512+256) out 32 * 32 *256
        u_x2_1,u_m2_1 = self.u_Pconv11(tf.concat([self.up(u_x2_2),x2_1],axis= -1),tf.concat([self.up(m2_2),m2_1],axis = -1)) # in 64*64*(256+128) out 64*64*128
        u_x2_0,u_m2_0 = self.u_Pconv10(tf.concat([self.up(u_x2_1),x2_0],axis= -1),tf.concat([self.up(u_m2_1),m2_0],axis = -1)) # in 128*128*(128+ 64) out 128*128 * 64
        
        #三层
        x3_0,m3_0 = u_x2_0*u_m2_0,u_m2_0
   
        #x3_0 = (x2_0*m2_0 +x3_0*m3_0)/(m2_0+m3_0+10e-7)
        
        x3_1, m3_1 = self.Pconv11(x3_0,m3_0)     # 64 * 64 * 128
        x3_1 = tf.nn.relu(x3_1)
        x3_2, m3_2 = self.Pconv12(x3_1,m3_1)     # 32 * 32 *256
        x3_2 = tf.nn.relu(x3_2)
        x3_3 = self.atn(x2_2,x3_2,m3_2)#*(1-u_m2_2) + u_x2_2 * u_m2_2
        
        x3_3, m3_3 = self.Pconv23(x3_3,m3_2)     # 16 * 16 * 512
        x3_3 = tf.nn.relu(x3_3)
        x3_4, m3_4 = self.Pconv24(x3_3,m3_3)     # 8 * 8 * 512
        x3_4 = tf.nn.relu(x3_4)
        x3_5, m3_5 = self.Pconv35(x3_4,m3_4)     # 4 * 4 * 512
        x3_5 = tf.nn.relu(x3_5)
        x3_6, m3_6 = self.Pconv36(x3_5,m3_5)     # 2 * 2 * 512
        x3_6 = tf.nn.relu(x3_6)
        
        u_x3_5,u_m3_5 = self.u_Pconv35(tf.concat([self.up(x3_6),x3_5],axis= -1),tf.concat([self.up(m3_6),m3_5],axis = -1)) # in 4*4*(512 + 512) out 8 * 8 * 512
        u_x3_4,u_m3_4 = self.u_Pconv34(tf.concat([self.up(u_x3_5),x3_4],axis= -1),tf.concat([self.up(u_m3_5),m3_4],axis = -1)) # in 8*8*(512 + 512) out 8 * 8 * 512
        u_x3_3,u_m3_3 = self.u_Pconv23(tf.concat([self.up(u_x3_4),x3_3],axis= -1),tf.concat([self.up(u_m3_4),m3_3],axis = -1)) # in 16*16*(512 + 512) out 16 * 16 * 512
        u_x3_2,u_m3_2 = self.u_Pconv22(tf.concat([self.up(u_x3_3),x3_2],axis= -1),tf.concat([self.up(u_m3_3),m3_2],axis = -1)) # in 32 * 32 * (512+256) out 31 * 32 *256
        u_x3_1,u_m3_1 = self.u_Pconv11(tf.concat([self.up(u_x3_2),x3_1],axis= -1),tf.concat([self.up(m3_2),m3_1],axis = -1)) # in 64*64*(256+128) out 64*64*128
        u_x3_0,u_m3_0 = self.u_Pconv10(tf.concat([self.up(u_x3_1),x3_0],axis= -1),tf.concat([self.up(u_m3_1),m3_0],axis = -1)) # in 128*128*(128+ 64) out 128*128 * 64
        
        x3,m3 = u_x3_0*u_m3_0,u_m3_0
        
        #x3 = (x2_0 * m2_0 +x3_0 * m3_0 +x3 * m3)/(m2_0 + m3_0 + m3 + 1e-8)
        
        c_x = tf.concat([x2_0,x3_0,x3],axis = -1)
        c_m = tf.concat([m2_0,m3_0,m3],axis = -1)
        c_x = self.conv(c_x)
        c_m = self.conv2(tf.concat([c_x,c_m],axis = -1))
        
        c_x = c_x * c_m
        x3,m3 = c_x,c_m
        
        x4 = self.Tconv(x3)
        x4 = tf.nn.leaky_relu(self.bn3(x4))
        m4 = tf.image.resize(m3,(m3.shape[1]*2,m3.shape[2]*2),"bilinear")
        x5 = tf.concat([input,x4],axis = -1) #这里是c
        m5 = tf.concat([mask,m4],axis = -1)
        
        x5,_ = self.tail1(x5,m5)
        x5 = tf.nn.leaky_relu(x5)
        x6 = self.tail2(x5)
        x6 = tf.concat([x5,x6],axis = -1)
        output = self.out(x6)
        
        return output
inputs = layers.Input(batch_shape = (6,256,256,3))
masks  = layers.Input(batch_shape = (6,256,256,3))
outputs = PFRNet()(inputs,masks)

generator  = models.Model(inputs = [inputs,masks],outputs = outputs)
generator.summary()
generator.load_weights("/home/oyx/jjj/inpainting/deepfashion/RE_deepfashion.h5")

# class PatialConv(tf.keras.layers.Layer):
#     def __init__(self, kernel=3,dilation_rate=1 ,stride=2,channels = 32,activation="relu",name="PatialConv",**kwargs):
#         super(PatialConv,self).__init__(name=name,**kwargs)     # Python 2 下使用 super(MyModel, self).__init__()
#         # 此处添加初始化代码（包含 call 方法中会用到的层），例如
#         # layer1 = tf.keras.layers.BuiltInLayer(...)
#         # layer2 = MyCustomLayer(...)
#         self.slide_window = kernel**2
#         # self.mask = mask #[1,h,w,1]
#         self.kernel = kernel
#         self.stride = stride
#         self.channels = channels
#         self.dilation_rate = dilation_rate
#         self.dense1 = tf.keras.layers.Conv2D(filters=self.channels,kernel_size=self.kernel,dilation_rate = self.dilation_rate, kernel_initializer=tf.keras.initializers.Ones(),strides=self.stride, padding="same", use_bias=False, trainable=False)
#         self.dense2 = tf.keras.layers.Conv2D(self.channels,kernel_size=self.kernel,dilation_rate = self.dilation_rate,kernel_initializer="he_normal",strides=self.stride, padding="same", use_bias=False, trainable=True)#
#         self.bias = self.add_weight(shape=(self.channels,),initializer=tf.constant_initializer(0.0),trainable=True)
#         self.relu = tf.keras.activations.relu
#     def call(self, input,mask):
        
#         update_mask = self.dense1(mask)
#         mask_ratio = (self.slide_window*input.shape[-1]) / (update_mask + 1e-8)
#         update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
#         mask_ratio = mask_ratio * update_mask

#         output = self.dense2(input)#-self.bias
#         output = output*mask_ratio
#         # output = self.relu(output*mask_ratio+self.bias)
        
#         return output*update_mask , update_mask
#     def get_config(self):
#        config = {"kernel":self.kernel,"stride":self.stride,"channels":self.channels,"dilation_rate":self.dilation_rate}
#        base_config = super(PatialConv, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))
# class Partial_UNet():
#     def __init__(self):
#         print ('build Partial_UNet ...')

#     def get_crop_shape(self, target, refer):
#         # width, the 3rd dimension
#         cw = (target.get_shape()[2] - refer.get_shape()[2])
#         # print(target.shape,refer.shape)
# #         print(target.get_shape(), refer.get_shape())
#         assert (cw >= 0)
#         if cw % 2 != 0:
#             cw1, cw2 = int(cw/2), int(cw/2) + 1
#         else:
#             cw1, cw2 = int(cw/2), int(cw/2)
#         # height, the 2nd dimension
#         ch = (target.get_shape()[1] - refer.get_shape()[1])
#         assert (ch >= 0)
#         if ch % 2 != 0:
#             ch1, ch2 = int(ch/2), int(ch/2) + 1
#         else:
#             ch1, ch2 = int(ch/2), int(ch/2)

#         return (ch1, ch2), (cw1, cw2)

#     def create_model(self,train_bn=True):

#         concat_axis = 3
#         inputs = layers.Input(batch_shape = (6,256,256,3),name = "img_input")
#         mask = layers.Input(batch_shape=(6,256,256,3),name="mask_input")
        
#         def encoder_layer(img_in, mask_in, filters, kernel=3,strides= 1,rate=1,name="PatialConv", bn=True):
#             conv, mask = PatialConv(kernel=kernel,dilation_rate=rate,stride=strides,channels = filters,name=name,trainable = True)(img_in, mask_in)
#             #conv, mask = PConv2D(filters, kernel_size, strides=2, padding='same')([img_in, mask_in])
#             if bn:
#                 conv = layers.BatchNormalization(name='EncBN'+str(encoder_layer.counter))(conv, training=train_bn)
            
#             conv = layers.ReLU()(conv)

#             encoder_layer.counter += 1
#             return conv, mask
#         encoder_layer.counter = 0

#         # DECODER
#         def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel=3,strides= 1,rate=1,name="PatialConv", bn=True):
#             up_img = layers.UpSampling2D(size=(2,2))(img_in)
#             up_mask = layers.UpSampling2D(size=(2,2))(mask_in)
#             concat_img = layers.Concatenate(axis=3)([e_conv,up_img])
#             concat_mask = layers.Concatenate(axis=3)([e_mask,up_mask])
#             conv, mask = PatialConv(kernel=kernel,dilation_rate=rate,stride=strides,channels = filters,name=name,trainable=True)(concat_img, concat_mask)
#             if bn:
#                 conv = layers.BatchNormalization()(conv)
#             conv = layers.LeakyReLU(alpha=0.2)(conv)
#             return conv, mask
        
#         conv1,mask1 = encoder_layer(inputs,mask,filters=64,kernel=5,strides=2,rate=1,name="PatialConv1",bn=False)
      
#         conv2,mask2 = encoder_layer(conv1,mask1,filters=128,kernel=5,strides=2,rate=1,name="PatialConv2")

#         conv3,mask3 = encoder_layer(conv2,mask2,filters=256,kernel=3,strides=2,rate=1,name="PatialConv3")
        
#         conv4,mask4 = encoder_layer(conv3,mask3,filters=512,kernel=3,strides=2,rate=1,name="PatialConv4")

#         conv5,mask5 = encoder_layer(conv4,mask4,filters=512,kernel=3,strides=2,rate=1,name="PatialConv5")
        
#         conv6,mask6 = encoder_layer(conv5,mask5,filters=512,kernel=3,strides=2,rate=1,name="PatialConv6")
        
#         conv7,mask7 = encoder_layer(conv6,mask6,filters=512,kernel=3,strides=2,rate=1,name="PatialConv7")       
        
#         conv11,mask11 = decoder_layer(conv7,mask7,conv6,mask6,filters=512,kernel=3,strides=1,rate=1,name="PatialConv11")
       
#         conv12,mask12 = decoder_layer(conv11,mask11,conv5,mask5,filters=512,kernel=3,strides=1,rate=1,name="PatialConv12")
        
#         conv13,mask13 = decoder_layer(conv12,mask12,conv4,mask4,filters=512,kernel=3,strides=1,rate=1,name="PatialConv13")
         
#         conv14,mask14 = decoder_layer(conv13,mask13,conv3,mask3,filters=256,kernel=3,strides=1,rate=1,name="PatialConv14")
        
#         conv15,mask15 = decoder_layer(conv14,mask14,conv2,mask2,filters=128,kernel=3,strides=1,rate=1,name="PatialConv15")
        
#         conv16,mask16 = decoder_layer(conv15,mask15,conv1,mask1,filters=64,kernel=3,strides=1,rate=1,name="PatialConv16")
        
#         conv17,mask17 = decoder_layer(conv16,mask16,inputs,mask,filters=3,kernel=3,strides=1,rate=1,name="PatialConv17",bn=False)
        
#         outputs = layers.Conv2D(3, (1, 1),activation="sigmoid",name='img_outputs')(conv17)

#         model = models.Model(inputs=[inputs,mask], outputs=outputs)
#         return model
# generator = Partial_UNet().create_model()
# generator.trainable = True
# generator.summary()

def getDiscriminator(image_shape,num_class):
   inputs = layers.Input(shape=image_shape)

   conv1 = layers.Conv2D(64, kernel_size=5, input_shape=image_shape,activation="elu")(inputs)
   conv1 = layers.Conv2D(64, kernel_size=5,strides=2,activation="elu")(conv1)
   # pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)
  
   conv2 = layers.Conv2D(128, kernel_size=3,activation="elu")(conv1)
   conv2 = layers.Conv2D(128, kernel_size=3,strides=2,activation="elu")(conv2)
   # pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
   # 7, 7, 64 -> 4, 4, 128
   conv3 = layers.Conv2D(256, kernel_size=3,activation="elu")(conv2)
   conv3 = layers.Conv2D(256, kernel_size=3,strides=2,activation="elu")(conv3)
   # pool3 = layers.MaxPooling2D(pool_size=(2,2))(conv3)


   conv4 = layers.Conv2D(512, kernel_size=3,activation="elu")(conv3)
   conv4 = layers.Conv2D(512, kernel_size=3,activation="elu")(conv4)
   # pool4 = layers.MaxPooling2D(pool_size=(2,2))(conv4)

  #  cont = layers.concatenate([global_average,global_max],axis=-1)

#    flatten1 = layers.Flatten()(conv3)
#    flatten = layers.concatenate([flatten,flatten1],axis = -1)
   # 全连接
   outputs = layers.Dense(1, activation='sigmoid')(conv4)

   return models.Model(inputs=inputs, outputs = outputs,name="discriminator")

# generator.load_weights("/home/oyx/jjj/inpainting/deepfashion/RE_deepfashion.h5")

#添加样式

# 内容层将提取出我们的 feature maps （特征图）
# content_layers = ['conv10'] 

# # 我们感兴趣的风格层
# style_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

# 内容层将提取出我们的 feature maps （特征图）
# content_layers = ['block5_conv2'] 

# 我们感兴趣的风格层
# style_layers = ['block1_conv1',
                # 'block2_conv1',
                # 'block3_conv1', 
                # 'block4_conv1', 
                # 'block5_conv1']
style_layers = ['block1_pool','block2_pool','block3_pool']
content_layers = []
# num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var
#def total_variation_loss(image):
#  x_deltas, y_deltas = high_pass_x_y(image)
#  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)
def total_variation_loss(image,mask_list):

  kernel = tf.ones((3,3,mask_list.shape[3],mask_list.shape[3]))
  dilated_mask = tf.nn.conv2d(1-mask_list,kernel,strides=[1,1,1,1],padding="SAME")
  dilated_ratio = 9.*3/(dilated_mask+10e-6)
  dilated_mask = tf.cast(tf.greater(dilated_mask,0),"float32")
  image = dilated_mask * image
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

def border_loss(image,mask_list):
  x_var = (image[:,:,1:,:] - image[:,:,:-1,:])*(mask_list[:,:,1:,:] - mask_list[:,:,:-1,:])
  y_var = (image[:,1:,:,:] - image[:,:-1,:,:])*(mask_list[:,1:,:,:] - mask_list[:,:-1,:,:])
  return tf.reduce_mean(tf.abs(x_var)) + tf.reduce_mean(tf.abs(y_var))
    
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG 
  # vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  # vgg = tf.keras.models.load_model("/content/drive/My Drive/data/my_segmentation_model.h5")
  vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2]*input_shape[3], tf.float32)
  return result/(num_locations)
# def gram_matrix(x):
#     """Gram matrix used in the style losses."""
# #     assert K.ndim(x) == 4, 'Input tensor should be 4D (B, H, W, C).'
# #     assert K.image_data_format() == 'channels_last', "Use channels-last format."

#     # Permute channels and get resulting shape
#     x = tf.transpose(x, (0, 3, 1, 2))
    
#     shape = x.shape
#     B, C, H, W = shape[0], shape[1], shape[2], shape[3]
    
#     # Reshape x and do batch dot product
#     features = tf.reshape(x, K.stack([B, C, H*W]))
    
#     gram = K.batch_dot(features, features, axes=2)

#     # Normalize with channels, height and width
#     gram /= K.cast(C * H * W, x.dtype) 
    
#     return gram

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False
    # self.vgg.summary()
    self.mean = [0.485, 0.456, 0.406]
    self.std = [0.229, 0.224, 0.225]
  def call(self, inputs):
    #inputs = tf.image.resize(inputs, (224, 224))
    outputs = self.vgg(inputs)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],outputs[self.num_style_layers:])
    #style层的原始输出
    perceptual_dict = {style_name:value 
                    for style_name, value 
                    in zip(self.style_layers, style_outputs)}
    style_dict = {style_name:gram_matrix(value)
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'perceptual':perceptual_dict, 'style':style_dict}


# content_targets = extractor(object_image)['content']
# image = np.random.normal(0,1,(400,600,3))
# image = tf.Variable(style_image[tf.newaxis,...], dtype=tf.float32) 
# image = tf.Variable(np.concatenate((a,b,c),axis=2)[tf.newaxis,...], dtype=tf.float32)
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=10e-8, clip_value_max=1.0-10e-8)


# style_weight=1e-2
# content_weight=1e4

# content_image 其实就是图片显示区域，style_image是局部样式，必须保持content_image的强一致
def style_content_loss(outputs,style_targets):
    # print("样式")
    style_outputs = outputs['style']
    style_targets = style_targets['style']

    # content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean(tf.abs(style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    return style_loss 
def l1_loss(y_pred,y_true,mask_list):
  # print("l1")
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true = tf.cast(y_true,dtype=tf.float32)
  return 1.*tf.reduce_mean(tf.abs(y_pred - y_true))+5.*tf.reduce_mean(tf.abs(y_pred - y_true)*(1-mask_list))

def cal_perceptual(outputs,style_targets):
  # print("样")
  style_outputs = outputs['perceptual']
  style_targets = style_targets['perceptual']
  
  result = tf.add_n([tf.reduce_mean(tf.abs(style_outputs[name]-style_targets[name])) for name in style_outputs.keys()])
  return result
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#这里暂时只计算生成器的对抗
def cal_adv(real_image_list,fake_image_list):
  # print("对抗")
  real_loss = cross_entropy(tf.ones_like(real_image_list), clip_0_1(real_image_list))
  fake_loss = cross_entropy(tf.zeros_like(fake_image_list), clip_0_1(fake_image_list))
  total_loss = real_loss + fake_loss
  return total_loss
def cal_gen(fake_image_list):
  return cross_entropy(tf.ones_like(fake_image_list), clip_0_1(fake_image_list))

def resize_image(batch_image):#
  B,H,W,C = batch_image.shape
  insize = 224
  
  batch_image = (batch_image+1)*127.5 - [123.68, 116.779,103.939]
  limx = H - in_size
  limy = W - in_size
  xs = np.random.randint(0,limx,B)
  ys = np.random.randint(0,limy,B)
  
  return np.array(image)

extractor = StyleContentModel(style_layers, content_layers)
#generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005,rho=0.9, epsilon=1e-06)#,decay = 0.02
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5,beta_2=0.999)
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5,beta_2=0.999)
@tf.function()
def train_step(image_list,mask_image_list,mask_list): #image为原图 0 - 1.0
  with tf.GradientTape() as tape ,tf.GradientTape() as disc_tape:
    B,H,W,C = image_list.shape
    
    image_list = tf.cast(image_list,dtype=tf.float32)
    mask_image_list = tf.cast(mask_image_list,dtype=tf.float32)
    mask_list = tf.cast(mask_list,dtype=tf.float32)
    gen_image_list = generator([mask_image_list,mask_list])
    
    #dis_image_list = discriminator(gen_image_list)
    #real_image_list = discriminator(image_list)

    gen_image_list = tf.cast(gen_image_list,dtype=tf.float32)

    
    #需要转换成224**224
    style_targets = extractor(image_list) #这里的输出其实将content层与style层都获取到了，但是只用得到style
    outputs = extractor(gen_image_list) #通过生成图像导出特征层 与 原始图像特征层对比
    
    # real 非空洞 fake 空洞
    comp = image_list*mask_list + gen_image_list*(1-mask_list)
    comp_outputs = extractor(comp)

    style_loss = 120.*(style_content_loss(outputs,style_targets) + style_content_loss(comp_outputs,style_targets))
    perceptual_loss = 0.05*(cal_perceptual(outputs,style_targets) + cal_perceptual(comp_outputs,style_targets))
    #adver_loss = 0.2*0.3*cal_gen(dis_image_list)
    
    L1_loss = l1_loss(gen_image_list,image_list,mask_list) 
    tvl_loss = 0.01*total_variation_loss(comp,mask_list)#gen_image_list
    
    loss = L1_loss + tvl_loss +style_loss+  perceptual_loss #+0.1* border_loss(gen_image_list,mask_list)#+  adver_loss # 
    #判别器
    #disc_loss = cal_adv(real_image_list,dis_image_list) *0.5 *0.5
  grads = tape.gradient(loss, generator.trainable_variables)
  # grad_list = []
  # for grad,variable in zip(grads,generator.trainable_variables):
  #   grad_list.append((grad,variable))
  #判别器
#   gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  #if(loss<1.5):
  generator_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
  
#   discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  return loss,style_loss,L1_loss,tvl_loss,perceptual_loss
#保存读取mask
def saveMask(mask,name,prefix="/content/drive/My Drive/data/inpainting/mask/"):
  mask = mask[:,:,np.newaxis]
  mask = np.concatenate([mask,mask,mask],axis=-1)*255
  image = Image.fromarray(np.uint8(mask))
  image.save(prefix+name+".jpg")
def readMask(name,prefix ="/content/drive/My Drive/data/inpainting/mask/"):
  image = Image.open(prefix+name+".jpg")
  image = np.array(image)/255.
  # print(image.shape)
  return image[:,:,0]
def getMaskListPaths(name):
  path_root = pathlib.Path(name)
  mask_paths = list(path_root.glob("*.png"))
  return np.array(mask_paths)
def getMaskList(mask_paths,image_size = (256,256)):
    mask_list = []
    for path in mask_paths:
        path = str(path)
        # 添加图片
        image = np.array(Image.open(path))
        h, w = image.shape

        s_h = np.random.randint(0, h - image_size[0])
        s_w = np.random.randint(0, w - image_size[1])

        image = image[s_h:s_h + image_size[0], s_w:s_w + image_size[1]][...,np.newaxis]
        mask_list.append(1. - image / 255.)
    return np.array(mask_list)
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
def train(mask_num = 1,batch_size = 6,epochs = 100,show_num = 5):
  global all_image_paths
  # mask_num = 1 #mask数量
  # batch_size = 6 #批量大小
  # epochs = 100 #执行次数
  # discriminator_num = 1 #每次判别器执行次数
  train_length = len(all_image_paths)
  loss_list = []
  #多个掩码
  # mask_list = getHoles((256,256),mask_num)
  # mask_list = getMaskList(10,"/usr/jjj/mask/")
  
#   hole_list = getHoles((256,256),holesNum)[...,np.newaxis]
  mask_paths = getMaskListPaths("/home/oyx/jjj/inpainting/mask/testing_mask_dataset/")#[8000:]
  holesNum = len(mask_paths)
  
  holeidx = [_ for _ in range(holesNum)]
  for epoch in range(epochs):
    np.random.shuffle(all_image_paths)
    np.random.shuffle(mask_paths)
    np.random.shuffle(holeidx)
    for load_train in range(int(np.ceil(len(all_image_paths)/(20*batch_size)))): #最后一个迭代用来测试 train_length//(10*batch_size) -1
      #初始化数据
      X_train = getDataset(all_image_paths[load_train*(20*batch_size):(load_train+1)*(20*batch_size)])/255.
    
      X_labels = X_train
      
      # maskconv = maskConv(mask[np.newaxis,:,:,np.newaxis])[0]
      # maskconv_list = np.array([maskconv for n in range(batch_size)])
      
      #对所有数据进行一轮循环学习
      id_list = [j for j in range(X_train.shape[0])]
      np.random.shuffle(id_list)
      for t in range(math.ceil(X_train.shape[0]/batch_size)):
        idx = id_list[t*batch_size:(t+1)*batch_size]
        X_ = X_train[idx]
#         mask_list = hole_list[np.random.randint(holesNum)][np.newaxis,...]
        mask_list = getMaskList(mask_paths[np.random.choice(holeidx,1)] )
        #mask_list = getHoles((256,256),1)[...,np.newaxis]
        #mask_list = np.concatenate([mask_list for _ in range(3)],axis = -1)
        mask_list = np.concatenate([mask_list for _ in range(len(X_))],axis = 0)
#         print(X_.shape,mask_list.shape)
        X_ =X_* mask_list
        loss,style_loss,L1_loss,tvl_loss,perceptual_loss = train_step(X_labels[idx],X_,mask_list) #训练生成器
      loss_list.append([loss.numpy(),style_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy()])  
      #test_images = getDataset(np.random.choice(test_image_paths,batch_size))/255. ," 测试 ",l1_loss(test_images,generator([test_images*mask_list,mask_list]),mask_list).numpy()
      print("训练第 ",load_train+1," 轮 损失",loss.numpy(),style_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy())
      if((load_train+1)%1 ==0):
        generator.save_weights("/home/oyx/jjj/inpainting/deepfashion/RE_deepfashion.h5")

  return loss_list


if(__name__=="__main__"):
  #all_image_paths = getAllImagePath("/root/sfwy/inpainting/CeleAHQ/img_align_celeba/")#/root/sfwy/inpainting/Paris_StreetView_Dataset/paris_train_original/
  all_image_paths = getAllImagePath("/home/oyx/jjj/inpainting/deepfashion/deepfashion_child/")
#   np.random.shuffle(all_image_paths) /root/sfwy/inpainting/test_256/
  print("训练数据： ",len(all_image_paths))
  #test_image_paths = all_image_paths[180000:181000]
  all_image_paths = all_image_paths[:]#14898 19998
  loss_list = train(mask_num = 1,batch_size = 6,epochs = 200,show_num = 5)
  
