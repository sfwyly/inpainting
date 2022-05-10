#encoding:utf-8
#PYTHONIOENCODING='utf-8'  编码问题命令行执行添加前缀

#部分卷积实现


# 局部样式+局部对抗损失
import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers 
import tensorflow.keras.backend as K
import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import time
# import functools
#tf.compat.v1.disable_eager_execution()
# def getHoles(x, num = 1):
#   direction = [[0,1],[0,-1],[1,0],[-1,0]]
#   height,width = x[0] , x[1]
#   #初始化所有数据
#   result = []
#   all = height * width
#   for _ in range(num):
#     t = 0 #孔洞数
#     mask = np.ones((height,width),dtype=np.float32)
#     x = np.random.randint(height/4,height/4*3)
#     y = np.random.randint(width/4,width/4*3)
#     for __ in range(all):
#       forward = np.random.randint(4)
      
#       while(x + direction[forward][0]<height/8 or x + direction[forward][0]>height/8*7 or y + direction[forward][1]<width/8 or y + direction[forward][1]>width/8*7):
#         forward = np.random.randint(4)
        
#       x = x + direction[forward][0]
#       y = y + direction[forward][1]

#       if(x < 0):
#         x = 0
#       elif(x > height-1):
#         x = height-1
      
#       if(y < 0):
#         y = 0
#       elif(y > width-1):
#         y = width -1
#       if(mask[x][y]==1):
#         t = t+1
#         mask[x][y] = 0.
      
#     result.append((mask,t*1.0/all))
#   return np.array(result)
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
    image = image.resize((256,256*218//178),Image.BILINEAR)
    image = np.array(image)
    train.append(image[24:280,...])
  return np.array(train)

class ContextualAttention(layers.Layer):
  def __init__(self,name="SpatialAttention",**kwargs):
        super(ContextualAttention,self).__init__(name,**kwargs)
  def call(self,input,mask):
    return self.contextual_attention(input,mask)
  def contextual_attention(self,x,mask,kernel=3,stride=1,rate=2,fuse = True):
      x_s = x.get_shape().as_list()
      softmax_scale = 10.
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

      mm = tf.cast(tf.equal(tf.reduce_mean(mask_patches[0], axis=[0,1,2], keepdims=True), 0.), tf.float32)

      k = 3#fuse_k
      fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
      y = []
      attention_matrix = []
      for xi, wi,raw_wi, m in zip(tf.split(f, x_s[0], axis=0),tf.split(x_patches, x_s[0], axis=0),tf.split(raw_x_patches, x_s[0], axis=0),tf.split(mask_patches, x_s[0], axis=0)):
        mm = tf.cast(tf.equal(tf.reduce_mean(m[0], axis=[0,1,2], keepdims=True), 1.), tf.float32)
        wi = wi[0]
        xi = tf.cast(xi,tf.float32)
        wi = tf.cast(wi,tf.float32)
        
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")#1 h/rate w/rate (h/rate)*(w/rate)
        attention_matrix.append(yi)
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
        yi = tf.nn.softmax(yi*softmax_scale, 3)
        yi *=  mm  # mask
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], x_s[1:]], axis=0), strides=[1,rate,rate,1]) / rate**2
        y.append(yi)
      
      y = tf.concat(y, axis=0)
      attention_matrix = tf.concat(attention_matrix,axis = 0)
      
      return y,attention_matrix

def getNetwork():
    
    def Conv(cnum = 32,ksize = 3,stride = 1, rate = 1,name = "conv",padding="same",activation=tf.nn.elu,training = True):
        
        x = layers.Conv2D(cnum,ksize,stride,dilation_rate=  rate,padding = padding,activation=activation,name = name)
        
        return x
    #阶段2
    cnum = 32
    inputs = layers.Input(batch_shape = (6,256,256,3))
    mask = layers.Input(batch_shape = (6,256,256,3))
    
    x  =  Conv(cnum,5,1,name = "conv1")(inputs)
    x = Conv(2*cnum,3,2,name = "conv2_downsample")(x)
    x = Conv(2*cnum,3,1,name = "conv3")(x)
    x = Conv(4*cnum,3,2,name = "conv4_downsample")(x)
    x = Conv(4*cnum,3,1,name = "conv5")(x)
    x = Conv(4*cnum,3,1,name = "conv6")(x)
    
    x = Conv(4*cnum,3,rate = 2,name = "conv7_atrous")(x)
    x = Conv(4*cnum,3,rate = 4,name = "conv8_atrous")(x)
    x = Conv(4*cnum,3,rate = 8,name = "conv9_atrous")(x)
    x = Conv(4*cnum,3,rate = 16,name = "conv10_atrous")(x)
    x = Conv(4*cnum,3,1,name = "conv11")(x)
    x = Conv(4*cnum,3,1,name = "conv12")(x)
    
    x = layers.UpSampling2D(size=(2,2),name = "up13")(x)
    x = Conv(2*cnum,3,1,name = "conv13_upsample")(x)
    x = Conv(2*cnum,3,1,name = "conv14")(x)
    
    x = layers.UpSampling2D(size=(2,2),name = "up15")(x)
    x = Conv(cnum,3,1,name = "conv15_upsample")(x)
    x = Conv(cnum//2,3,1,name = "conv16")(x)
    
    x = Conv(3,3,1,activation="sigmoid",name="conv17")(x)

    #x = tf.clip_by_value(x, -1., 1.)
    x_stage1 = x
    
    x = x*(1-mask)+inputs*mask
    xnow = x
    #阶段2
    x  =  Conv(cnum,5,1,name = "xconv1")(xnow)
    x = Conv(cnum,3,2,name = "xconv2_downsample")(x)
    x = Conv(2*cnum,3,1,name = "xconv3")(x)
    x = Conv(2*cnum,3,2,name = "xconv4_downsample")(x)
    x = Conv(4*cnum,3,1,name = "xconv5")(x)
    x = Conv(4*cnum,3,1,name = "xconv6")(x)
    mask_s = tf.image.resize(mask,(x.shape)[1:3],"bilinear")
    x = Conv(4*cnum,3,rate = 2,name = "xconv7_atrous")(x)
    x = Conv(4*cnum,3,rate = 4,name = "xconv8_atrous")(x)
    x = Conv(4*cnum,3,rate = 8,name = "xconv9_atrous")(x)
    x = Conv(4*cnum,3,rate = 16,name = "xconv10_atrous")(x)
    
    x_hallu = x
    
    #attention branch
    x = Conv(cnum,5,1,name = "pmconv1")(xnow)
    x = Conv(cnum,3,2,name = "pmconv2_downsample")(x)
    x = Conv(2*cnum,3,1,name="pmconv3")(x)
    x = Conv(4*cnum,3,2,name="pmconv4_downsample")(x)
    x = Conv(4*cnum,3,1,name="pmconv5")(x)
    x = Conv(4*cnum,3,1,name="pmconv6",activation = tf.nn.relu)(x)
    #注意力
    x,attention_matrix = ContextualAttention()(x,mask_s)
    x = Conv(4*cnum,3,1,name = "pmconv9")(x)
    x = Conv(4*cnum,3,1,name = "pmconv10")(x)
    pm = x
    x = tf.concat([x_hallu,pm],axis = -1)
    
    x = Conv(4*cnum,3,1,name = "allconv11")(x)
    x = Conv(4*cnum,3,1,name = "allconv12")(x)
    x = layers.UpSampling2D(size=(2,2),name = "allup13")(x)
    x = Conv(2*cnum,3,1,name = "allconv13_upsample")(x)
    x = Conv(2*cnum,3,1,name = "allconv14")(x)
    x = layers.UpSampling2D(size=(2,2),name = "allup15")(x)
    x = Conv(cnum,3,1,name = "allconv15_upsample")(x)
    x = Conv(cnum//2,3,1,name = "allconv16")(x)
    x = Conv(3,3,1,activation="sigmoid",name="allconv17")(x)
    
    x_stage2 = x
    #x_stage2 = tf.clip_by_value(x, -1., 1.)
    
    model = models.Model(inputs= [inputs,mask],outputs = [x_stage2,attention_matrix])
    return model
generator = getNetwork()

# 分类器基于SeResneXt

class BAP(layers.Layer):

    def __init__(self, pool="GAP"):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if (pool == "GAP"):
            self.pool = None
        else:
            self.pool = layers.GlobalMaxPool2D()
        self.EPSILON = 1e-12

    def call(self, features, attentions):

        B, H, W, C = features.shape
        _, AH, AW, M = attentions.shape

        # match size
        if (AH != H or AW != W):
            attentions = tf.image.resize(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C )
        if (self.pool is None):
            feature_matrix = tf.linalg.einsum('ijkm,ijkn->imn', attentions, features) / (H * W)
            feature_matrix = tf.reshape(feature_matrix, (B, -1))
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[..., i:i + 1])
                feature_matrix.append(AiF)
            feature_matrix = tf.concat(feature_matrix, axis=-1)

        # sign-sqrt
        feature_matrix = tf.sign(feature_matrix) * tf.sqrt(tf.abs(feature_matrix) + self.EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = tf.nn.l2_normalize(feature_matrix, axis=-1)

        return feature_matrix

class SEResNeXt(object):

    def __init__(self, size=256, num_classes=256, depth=32, reduction_ratio=4, num_split=8, num_block=3):
        self.depth = depth  # number of channels
        self.ratio = reduction_ratio  # ratio of channel reduction in SE module
        self.num_split = num_split  # number of splitting trees for ResNeXt (so called cardinality)
        self.num_block = num_block  # number of residual blocks
        self.channel_axis = 3
        self.M = 32  # attentions
        self.model = self.build_model(layers.Input(batch_shape=(6, size, size, 3)), num_classes)

    def conv_bn(self, x, filters, kernel_size, stride, padding='same', rate=1):
        x = layers.Conv2D(filters=filters, kernel_size=[kernel_size, kernel_size],
                          strides=[stride, stride], dilation_rate=rate, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        return x

    def activation(self, x, func='relu'):
        return layers.Activation(func)(x)

    def channel_zeropad(self, x):
        shape = list(x.shape)
        y = tf.zeros_like(x)

        if self.channel_axis == 3:
            y = y[:, :, :, :shape[self.channel_axis] // 2]
        else:
            y = y[:, :shape[self.channel_axis] // 2, :, :]

        return tf.concat([y, x, y], self.channel_axis)

    def channel_zeropad_output(self, input_shape):
        shape = list(input_shape)
        shape[self.channel_axis] *= 2

        return tuple(shape)

    def initial_layer(self, inputs):

        x = self.conv_bn(inputs, self.depth, 3, 1)
        x = self.activation(x)

        return x

    def transform_layer(self, x, stride, rate):
        x = self.conv_bn(x, self.depth, 1, 1)
        x = self.activation(x)

        x = self.conv_bn(x, self.depth, 3, stride, rate=(rate if stride == 1 else 1))
        x = self.activation(x)

        return x

    def split_layer(self, x, stride):

        splitted_branches = list()
        for i in range(self.num_split):
            branch = self.transform_layer(x, stride, rate=i + 1)
            splitted_branches.append(branch)
        return tf.concat(splitted_branches, axis=self.channel_axis)

    def squeeze_excitation_layer(self, x, out_dim):

        squeeze = layers.GlobalAveragePooling2D()(x)

        excitation = layers.Dense(units=out_dim // self.ratio)(squeeze)
        excitation = self.activation(excitation)
        excitation = layers.Dense(units=out_dim)(excitation)
        excitation = self.activation(excitation, 'sigmoid')
        excitation = layers.Reshape((1, 1, out_dim))(excitation)

        scale = tf.multiply(x, excitation)

        return scale

    def residual_layer(self, x, out_dim):

        for i in range(self.num_block):
            input_dim = int(np.shape(x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
            else:
                flag = False
                stride = 1

            subway_x = self.split_layer(x, stride)
            subway_x = self.conv_bn(subway_x, out_dim, 1, 1)
            subway_x = self.squeeze_excitation_layer(subway_x, out_dim)

            if flag:
                pad_x = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
                pad_x = layers.Lambda(self.channel_zeropad, output_shape=self.channel_zeropad_output)(pad_x)
            else:
                pad_x = x

            x = self.activation(layers.add([pad_x, subway_x]))

        return x

    def build_model(self, inputs, num_classes):

        x = self.initial_layer(inputs)

        x = self.residual_layer(x, out_dim=64)
        x = self.residual_layer(x, out_dim=128)
        feature_maps = self.residual_layer(x, out_dim=256)  # 32 32 256

        attentions = self.conv_bn(feature_maps, self.M, 1, 1)

        # x = layers.GlobalAveragePooling2D()(x)
        # x = layers.Dense(units=num_classes)(x)

        self.bap = BAP()

        feature_matrix = self.bap(feature_maps, attentions)
        outputs = layers.Dense(num_classes, use_bias=False)(feature_matrix * 100)

        return models.Model(inputs, [outputs,feature_matrix])
# 分类器 基于SE_Net 实现细分类
senet = SEResNeXt()
classifier = senet.model

if(os.path.exists("/root/sfwy/inpainting/scf.h5")):
    generator.load_weights("/root/sfwy/inpainting/scf.h5")
if(os.path.exists("/root/sfwy/inpainting/scf_classifier.h5")):
    classifier.load_weights("/root/sfwy/inpainting/scf_classifier.h5")
#添加样式

# 内容层将提取出我们的 feature maps （特征图）
# content_layers = ['conv10'] 

# # 我们感兴趣的风格层
# style_layers = ["conv1", "conv2", "conv3", "conv4", "conv5"]

# 内容层将提取出我们的 feature maps （特征图）
# content_layers = ['block5_conv2'] 

# 我们感兴趣的风格层
#style_layers = ['block1_conv1','block2_conv1','block3_conv1', 'block4_conv1', 'block5_conv1']
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
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits= False)
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
# generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001,rho=0.9, epsilon=1e-06)#,decay = 0.02
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5,beta_2=0.999)
classifier_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5,beta_2=0.999)

h,w = 32,32
cate = 256
feature_center = tf.zeros((cate,h,w,h*w),dtype = tf.float32)
categorate_center = tf.zeros((cate,256*32),dtype = tf.float32)
beta = 5e-2

#从硬盘加载
#center  ={"feature_center":feature_center}

def cal_feature_center(feature_center,attention_matrix,categories):
    categories = tf.cast(tf.argmax(categories,axis = -1),tf.int32)
    #print(feature_center.shape)
    feature_center_batch = tf.nn.l2_normalize(tf.gather(feature_center,categories), axis=-1)
    #print(feature_center_batch.shape)
    #attention_matrix = tf.nn.l2_normalize(attention_matrix,axis=-1)
    #print(attention_matrix.shape)
    result = beta * (attention_matrix - feature_center_batch)
    r = []
    for j in range(feature_center.shape[0]):
        x = feature_center[j]
        t= 0
        for i in categories:
            if(i==j):
                x +=result[t] 
            t +=1
        r.append(x)#[tf.newaxis,...])
    return tf.stack(r,axis = 0 ),feature_center_batch

cate_loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
def selfSuperviseLoss(categroies,true_categroies):

    true_categroies = tf.argmax(true_categroies,axis = -1)
    # one - hot
    true_categroies = tf.keras.utils.to_categorical(true_categroies,cate)

    loss = cate_loss(true_categroies,categroies)

    return loss

#@tf.function()
def train_step(image_list,mask_image_list,mask_list): #image为原图 0 - 1.0
  global feature_center,beta,categorate_center
  with tf.GradientTape() as tape ,tf.GradientTape() as disc_tape:
    B,H,W,C = image_list.shape
    
    image_list = tf.cast(image_list,dtype=tf.float32)
    mask_image_list = tf.cast(mask_image_list,dtype=tf.float32)
    mask_list = tf.cast(mask_list,dtype=tf.float32)

    gen_image_list,attention_matrix = generator([mask_image_list,mask_list])
    #分离 无监督分类器 与 生成器 分类器生成有效的分类
    categories,_ = classifier(gen_image_list)
    true_categories,feature_matrix = classifier(image_list)

    gen_image_list = tf.cast(gen_image_list, dtype=tf.float32)
    attention_matrix = tf.cast(attention_matrix, tf.float32)
    feature_center, feature_center_batch = cal_feature_center(feature_center, attention_matrix, true_categories)
    categorate_center, categorate_center_batch = cal_feature_center(categorate_center, feature_matrix, true_categories)


    #需要转换成224**224
    style_targets = extractor(image_list) #这里的输出其实将content层与style层都获取到了，但是只用得到style
    outputs = extractor(gen_image_list) #通过生成图像导出特征层 与 原始图像特征层对比
    
    # real 非空洞 fake 空洞
    comp = image_list*mask_list + gen_image_list*(1-mask_list)
    comp_outputs = extractor(comp)

    style_loss = 120.*(style_content_loss(outputs,style_targets) + style_content_loss(comp_outputs,style_targets))
    perceptual_loss = 0.05*(cal_perceptual(outputs,style_targets) + cal_perceptual(comp_outputs,style_targets))
    center_loss = 100.*tf.reduce_mean(tf.square(attention_matrix - feature_center_batch))
    L1_loss = l1_loss(gen_image_list,image_list,mask_list)
    tvl_loss = 0.1*total_variation_loss(comp,mask_list)#gen_image_list

    loss = L1_loss + tvl_loss + style_loss + perceptual_loss + center_loss

    # 非完整图像类别需要与原始图像类别保持一致 自监督 分类器
    categorate_center_loss = 100*tf.reduce_mean(tf.square(feature_matrix - categorate_center_batch))
    # 将二者分类趋于一致
    sim_cate_loss = selfSuperviseLoss(categories, true_categories)
    classifier_loss = categorate_center_loss + sim_cate_loss

  grads = tape.gradient(loss, generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
  disc_grads = disc_tape.gradient(classifier_loss, classifier.trainable_variables)
  classifier_optimizer.apply_gradients(zip(disc_grads, classifier.trainable_variables))

  return loss,style_loss,L1_loss,tvl_loss,perceptual_loss,center_loss,categorate_center_loss,classifier_loss

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
    image  =np.array(image)/255.
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
  mask_paths = getMaskListPaths("/root/sfwy/inpainting/mask/testing_mask_dataset/")[8000:]
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
        mask_list  =np.concatenate([mask_list for _ in range(len(X_))],axis = 0)
        X_ =X_* mask_list
        loss,style_loss,L1_loss,tvl_loss,perceptual_loss,center_loss,categorate_center_loss,classifier_loss = train_step(X_labels[idx],X_,mask_list) #训练生成器
      print("训练第 ",load_train+1," 轮 损失",loss.numpy(),style_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy(),center_loss.numpy(),categorate_center_loss.numpy(),classifier_loss.numpy())
      if((load_train+1)%50 ==0):
        generator.save_weights("/root/sfwy/inpainting/scf.h5")
        classifier.save_weights("/root/sfwy/inpainting/scf_classifier.h5")
  return loss_list

if(__name__=="__main__"):
  all_image_paths = getAllImagePath("/root/sfwy/inpainting/CeleAHQ/img_align_celeba/")
  #all_image_paths = getAllImagePath("/root/sfwy/inpainting/place2/data_256/z/zen_garden/")
#   np.random.shuffle(all_image_paths)
  print("训练数据： ",len(all_image_paths))
  test_image_paths = all_image_paths[20000:21000]
  all_image_paths = all_image_paths[:20000-2]
  loss_list = train(mask_num = 1,batch_size = 6,epochs = 2000,show_num = 5)

