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
  result = list(path_root.glob("*"))
  
  return result

def getDataset(all_image_paths):

  train = []
  # labels = []
  for path in all_image_paths:
    path = str(path)
    #添加图片
    image = Image.open(path)
    image = image.resize((256,256),Image.BILINEAR)#*218//178
    image = np.array(image)
    if(len(image.shape)!=3):
        continue
    train.append(image)#[24:280,...])
  return np.array(train)

# def getDataset(all_image_paths):

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


"""
    复现 Recurrent Feature Reasoning 希望能找到一些灵感
    移植到残差特征循环推理机制

"""

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow as tf
import numpy as np

# Bottle块
class Bottleneck(layers.Layer):

    def __init__(self, output_dim,strides = 1, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.conv1 = layers.Conv2D(output_dim//4,kernel_size = 1,padding = "same",use_bias =False) 
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_dim//4,kernel_size = 3,strides = strides,padding = "same",use_bias = False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(output_dim,kernel_size = 1,padding = "same",use_bias = False)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def build(self, input_shape):
        super(Bottleneck, self).build(input_shape)  

    def call(self, x):
      
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out =  out + residual
        out = self.relu(out)
        
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# 知识一致性注意力
class KnowledgeConsistentAttention(layers.Layer):

    def __init__(self, patch_size = 3, propagate_size = 3,stride=1, output_dim = 256, **kwargs):
        self.output_dim = output_dim
        super(KnowledgeConsistentAttention, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.prop_kernels = None
        self.att_scores_prev = None
        self.masks_prev = None
        self.ratio = tf.ones(1)
        
    def build(self, input_shape):
        super(KnowledgeConsistentAttention, self).build(input_shape)

    def call(self, foreground , masks):
        
        bz , w , h ,nc = foreground.shape
        # masks shape == foreground.shape
        if(masks.shape[1]!=h):
            masks = tf.image.resize(masks,(h,w),"bilinear")
        
        background = foreground[:]
        
        conv_kernels_all = tf.transpose(background,[0,3,1,2])
        conv_kernels_all = tf.reshape(conv_kernels_all,[bz,nc,h*w,1,1])#他这直接单个特征通道直接进行相似度计算，感觉误差会很大啊
        conv_kernels_all = tf.transpose(conv_kernels_all,[0,3,4,1,2]) # b k k c hw  
        
        output_tensor = []
        att_score = []
        for i in range(bz):
            feature_map = foreground[i:i+1]
            conv_kernels = conv_kernels_all[i] + 0.0000001 # k k c hw
            norm_factor = tf.reduce_sum(conv_kernels**2,[0,1,2],keepdims=True)**0.5
            conv_kernels = conv_kernels / norm_factor
            conv_result = tf.nn.conv2d(feature_map,conv_kernels,strides = [1,1,1,1],padding = "SAME") # 1 h w hw
            if(self.propagate_size !=1):
                if(self.prop_kernels is None):
                    self.prop_kernels = tf.ones([conv_result.shape[-1],1,self.propagate_size,self.propagate_size])
                    self.prop_kernels.requires_grad = False
                
                conv_result = tf.nn.avg_pool2d(conv_result,3,1,padding="SAME")*9
            attention_scores = tf.nn.softmax(conv_result,axis = -1)
            
            if(self.att_scores_prev is not None):
                attention_scores = (self.att_scores_prev[i:i+1]*self.masks_prev[i:i+1]+attention_scores * (tf.abs(self.ratio)+1e-7))/(self.masks_prev[i:i+1]+(tf.abs(self.ratio)+1e-7))

            att_score.append(attention_scores)
            feature_map = tf.nn.conv2d_transpose(attention_scores, conv_kernels,output_shape = [1,w,h,nc],strides = 1,padding="SAME")
            final_output = feature_map
            output_tensor.append(final_output)
        self.att_scores_prev = tf.reshape(tf.concat(att_score,axis =0),[bz,h,w,h*w])
        self.masks_prev = tf.reshape(masks,[bz,h,w,1])
        return tf.concat(output_tensor,axis = 0)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
class AttentionModule(layers.Layer):
    
    def __init__(self,inchannel,patch_size_list = [1],propagate_size_list = [3],stride_list = [1],**kwargs):
        super(AttentionModule, self).__init__(**kwargs)
        self.att = KnowledgeConsistentAttention(patch_size_list[0],propagate_size_list[0],stride_list[0])
        self.num_of_modules = len(patch_size_list)
        self.combiner = layers.Conv2D(inchannel,kernel_size = 1,padding="same")
    def build(self, input_shape):
        super(AttentionModule, self).build(input_shape)  

    def call(self, foreground, mask):
        outputs = self.att(foreground, mask)
        outputs = tf.concat([outputs,foreground],axis = -1)
        outputs = self.combiner(outputs)
        return outputs        

class RFRModule(layers.Layer):

    def __init__(self, layer_size = 6,in_channel = 64, **kwargs):
        super(RFRModule, self).__init__(**kwargs)
        self.freeze_enc_bn = False
        self.layer_size = layer_size
        self.encoder_blocks = {}
        self.in_channel = in_channel
        for _ in range(3):
            out_channel = in_channel * 2
            block = models.Sequential([layers.Conv2D(out_channel,kernel_size = 3,strides = 2,padding="same",use_bias = False)
                    ,layers.BatchNormalization(),layers.ReLU()])
            name = "enc_{:d}".format(_+1)
            self.encoder_blocks[name] = block
            in_channel = out_channel
        
        for _ in range(3,6):
            block  = models.Sequential([layers.Conv2D(out_channel,kernel_size = 3,strides = 1,padding="same",dilation_rate=2,use_bias = False)
                    ,layers.BatchNormalization(),layers.ReLU()])
            name = "enc_{:d}".format(_+1)
            self.encoder_blocks[name] = block
        #知识一致注意力模块 TODO
        self.att = AttentionModule(512)
        
        self.decoder_blocks = {}
        for _ in range(5,3,-1):
            block = models.Sequential([layers.Conv2D(in_channel,kernel_size = 3,strides = 1,padding="same",dilation_rate=2,use_bias = False)
                    ,layers.BatchNormalization(),layers.LeakyReLU(0.2)])
            name = "dec_{:d}".format(_)
            self.decoder_blocks[name] = block
            
        # 1024 -> 512
        self.decoder_blocks["dec_3"] = models.Sequential([
            layers.Conv2DTranspose(8*self.in_channel,kernel_size = 4,strides = 2,padding="same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        # 768 -> 256
        self.decoder_blocks["dec_2"] = models.Sequential([
            layers.Conv2DTranspose(4*self.in_channel,kernel_size = 4,strides = 2,padding="same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        # 384 ->64
        self.decoder_blocks["dec_1"] = models.Sequential([
            layers.Conv2DTranspose(self.in_channel,kernel_size = 4,strides = 2,padding="same",use_bias = False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])
        
        
    def build(self, input_shape):
        super(RFRModule, self).build(input_shape)  

    def call(self, input,mask):
        
        h_dict = {}
        h_dict["h_0"] = input
        
        h_key_prev = "h_0"
        for i in range(1,self.layer_size + 1):
            l_key = "enc_{:d}".format(i)
            h_key = "h_{:d}".format(i)
            h_dict[h_key] = self.encoder_blocks[l_key](h_dict[h_key_prev])
            h_key_prev = h_key
            
        h = h_dict[h_key]
        
        for i in range(self.layer_size - 1,0,-1):
            enc_h_key = "h_{:d}".format(i)
            dec_l_key = "dec_{:d}".format(i)
            h = tf.concat([h,h_dict[enc_h_key]],axis = -1)
            h = self.decoder_blocks[dec_l_key](h)
            if(i==3):
                h = self.att(h,mask)#h 32*32 mask 128*128
        return h

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#部分卷积

class PartialConv(layers.Layer):
    def __init__(self,kernel_size = 3,strides = 1,dilation_rate = 1,in_channels = 256,padding="same",out_channels=256,use_bias = True,return_mask = True,multi_channel = False,**kwargs):
        super(PartialConv, self).__init__(**kwargs)
        
        self.multi_channel = multi_channel
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_mask = return_mask
        self.conv = layers.Conv2D(self.out_channels,kernel_size = self.kernel_size,strides = self.strides,dilation_rate = self.dilation_rate,padding="same")
        if(self.multi_channel):
            self.weight_maskUpdater = tf.ones((self.kernel_size,self.kernel_size,self.in_channels,self.out_channels))
        else:
            self.weight_maskUpdater = tf.ones((self.kernel_size,self.kernel_size,1,1))
        
        self.slide_winsize = self.weight_maskUpdater.shape[1]*self.weight_maskUpdater.shape[2]*self.weight_maskUpdater.shape[0]
        
        self.last_size = (None,None)
        self.update_mask = None
        self.mask_ratio = None
        
    def build(self, input_shape):
        super(PartialConv, self).build(input_shape) 
    def call(self,input,mask = None):
        
        if(mask is None):
            #没有mask就创建一个全为1的mask，这样和普通卷积没有什么区别
            if(self.multi_channel):
                mask = tf.ones((input.shape[0],input.shape[1],input.shape[2],input.shape[3]))
            else:
                mask = tf.ones((1,input.shape[1],input.shape[2],1))
                
        
        self.update_mask = tf.nn.conv2d(mask,self.weight_maskUpdater,strides = self.strides,padding="SAME")
        self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
        self.update_mask = tf.clip_by_value(self.update_mask,0,1)
        self.mask_ratio = self.mask_ratio * self.update_mask
        # mask and input are similar data type
        
        raw_output = self.conv(input)
        if(self.use_bias):
            bias_view = self.add_weight(name='kernel', 
                                      shape=(self.out_channels,),
                                      initializer='uniform',
                                      trainable=True)
            output = (raw_output - bias_view)*self.mask_ratio + bias_view
            output = output*self.update_mask
        else:
            output = raw_output*self.mask_ratio
            
        if(self.return_mask):
            return output,self.update_mask
        else:
            return output

#构造RFR

class RFRNet(models.Model):
    def __init__(self):
        super(RFRNet, self).__init__()
        
        self.Pconv1 = PartialConv(in_channels = 3,out_channels = 64,kernel_size = 7,strides = 2,padding="same",multi_channel = True,use_bias = False)
        self.bn1 = layers.BatchNormalization()
        self.Pconv2 = PartialConv(in_channels = 64,out_channels = 64,kernel_size = 7,strides = 1,padding="same",multi_channel = True,use_bias = False)
        self.bn20 = layers.BatchNormalization()
        self.Pconv21 = PartialConv(in_channels = 64,out_channels = 64,kernel_size = 7,strides = 1,padding="same",multi_channel = True,use_bias = False)
        self.Pconv22 = PartialConv(in_channels = 64,out_channels = 64,kernel_size = 7,strides = 1,padding="same",multi_channel = True,use_bias = False)
        self.bn2 = layers.BatchNormalization()
        self.RFRModule = RFRModule(trainable = True)
        self.Tconv = layers.Conv2DTranspose(64,kernel_size = 4,strides = 2,padding="same",use_bias = False)
        self.bn3 = layers.BatchNormalization()
        self.tail1 = PartialConv(in_channels = 67,out_channels = 32,kernel_size = 3,strides = 1,padding="same",multi_channel = True,use_bias = False)
        self.tail2 = Bottleneck(output_dim = 32,strides = 1)
        self.out = layers.Conv2D(3,kernel_size = 3,activation="sigmoid",strides = 1,padding="same",use_bias = True)
    
    def call(self,input,mask):

        #一次下采样后进行循环推理
        x1 , m1 = self.Pconv1(input,mask)
        x1 = tf.nn.relu(self.bn1(x1))
        x1 , m1 = self.Pconv2(x1, m1)
        x1 = tf.nn.relu(self.bn20(x1))
        x2 = x1
        x2, m2 = x1, m1
        n, h, w, c = x2.shape
        feature_group = []
        mask_group = []
        
        self.RFRModule.att.att.att_scores_prev = None
        self.RFRModule.att.att.masks_prev = None
        
        #循环推理
        for i in range(6):
            x2, m2 = self.Pconv21(x2, m2)
            x2, m2 = self.Pconv22(x2, m2)
            x2 = tf.nn.leaky_relu(self.bn2(x2))
            x2 = self.RFRModule(x2,m2[...,0:1])#这里的x2 m2都是128*128大小 在RFR模块里dec_3执行后h是32*32大小 需要对mask进行一个resize
            
            x2 = x2 * m2
            feature_group.append(x2[...,tf.newaxis])
            mask_group.append(m2[...,tf.newaxis])
            
        x3 = tf.concat(feature_group,axis = -1)
        m3 = tf.concat(mask_group , axis = -1)
        
        amp_vec = tf.reduce_mean(m3,axis = -1)
        x3 =  tf.reduce_mean(x3 * m3 , axis= -1)/(amp_vec + 1e-7)
        x3 = tf.reshape(x3,[n,h,w,c])
        m3 = m3[...,-1]
        
        x4 = self.Tconv(x3)
        x4 = tf.nn.leaky_relu(self.bn3(x4))
        m4 = tf.image.resize(m3,(m3.shape[1]*2,m3.shape[2]*2),"bilinear")
        #m4 = layers.UpSampling2D(size = (2,2))(m3)
        x5 = tf.concat([input,x4],axis = -1) #这里是c
        m5 = tf.concat([mask,m4],axis = -1)
        
        x5,_ = self.tail1(x5,m5)
        x5 = tf.nn.leaky_relu(x5)
        x6 = self.tail2(x5)
        x6 = tf.concat([x5,x6],axis = -1)
        output = self.out(x6)
        
        return output
inputs = layers.Input(batch_shape = (6,256,256,3))
masks = layers.Input(batch_shape = (6,256,256,3))

outputs = RFRNet()(inputs,masks)
generator = models.Model(inputs = [inputs,masks],outputs = outputs)
generator.summary()

#generator.load_weights("/home/oyx/jjj/inpainting/deepfashion/deepfashion/RFR_deepfashion.h5")

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
# generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001,rho=0.9, epsilon=1e-06)#,decay = 0.02
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5,beta_2=0.999)#
#discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001,beta_1=0.5,beta_2=0.999)
@tf.function()
def train_step(image_list,mask_image_list,mask_list): #image为原图 0 - 1.0
  with tf.GradientTape() as tape:
    B,H,W,C = image_list.shape
    image_list = tf.cast(image_list,dtype=tf.float32)
    mask_image_list = tf.cast(mask_image_list,dtype=tf.float32)
    mask_list = tf.cast(mask_list,dtype=tf.float32)
    gen_image_list = generator([mask_image_list,mask_list])
    
    #dis_image_list = discriminator(gen_image_list)
    #real_image_list = discriminator(image_list)
    
#     mean = [0.485,0.456,0.406]
#     std = [0.229,0.224,0.225]
#     image_list = (image_list-mean)/std
#     gen_image_list = (gen_image_list - mean)/std
#     print(tf.reduce_mean(image_list),tf.reduce_mean(gen_image_list))
    gen_image_list = tf.cast(gen_image_list,dtype=tf.float32)

    
#     in_size = 224
#     new_image_list = image(image_list+1)*127.5 - [123.68, 116.779,103.939]
#     new_gen_image_list = (gen_image_list+1)*127.5 - [123.68, 116.779,103.939]
    
#     xs = np.random.randint(0,33,B)
#     ys= np.random.randint(0,33,B)
    
#     lis = [image_list[i,x:x+in_size,y:y+in_size,:][tf.newaxis,...] for i,(x,y) in enumerate(zip(xs,ys))]
#     new_image_list = tf.concat(lis,axis=0)
    
#     lis = [gen_image_list[i,x:x+in_size,y:y+in_size,:][tf.newaxis,...] for i,(x,y) in enumerate(zip(xs,ys))]
#     new_gen_image_list = tf.concat(lis,axis=0)
    
# #     print("形状 ",mask_list.shape)
#     lis = [mask_list[:,x:x+in_size,y:y+in_size,:] for i,(x,y) in enumerate(zip(xs,ys))]
#     new_mask_list = tf.concat(lis,axis=0)
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
    
    loss = L1_loss + tvl_loss +style_loss+  perceptual_loss #+  adver_loss # 
    #判别器
    #disc_loss = cal_adv(real_image_list,dis_image_list) *0.5 *0.5
  grads = tape.gradient(loss, generator.trainable_variables)
  # grad_list = []
  # for grad,variable in zip(grads,generator.trainable_variables):
  #   grad_list.append((grad,variable))
  #判别器
#   gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  if(loss):
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
        mask_list  =np.concatenate([mask_list for _ in range(len(X_))],axis = 0)
#         print(X_.shape,mask_list.shape)
        X_ =X_* mask_list #+ (1-mask_list)
        loss,style_loss,L1_loss,tvl_loss,perceptual_loss = train_step(X_labels[idx],X_,mask_list) #训练生成器
      #loss_list.append([loss.numpy(),style_loss.numpy(),adver_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy()])  
      #test_images = getDataset(np.random.choice(test_image_paths,batch_size))/255. ," 测试 ",l1_loss(test_images,generator([test_images*mask_list,mask_list]),mask_list).numpy()
      print("训练第 ",load_train+1," 轮 损失",loss.numpy(),style_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy())
      if((load_train+1)%1 ==0):
        generator.save_weights("/home/oyx/jjj/inpainting/deepfashion/RFR_deepfashion.h5")#generator_partial_attention_CeleA
        #generator.save_weights("/root/sfwy/inpainting/generator_partial_attention_Place2.h5")
  return loss_list


if(__name__=="__main__"):
  all_image_paths = getAllImagePath("/home/oyx/jjj/inpainting/deepfashion/deepfashion_child")#/root/sfwy/inpainting/Paris_StreetView_Dataset/paris_train_original/ /root/sfwy/inpainting/place2/data_256/z/zen_garden/
  #all_image_paths = getAllImagePath("/root/sfwy/inpainting/test_256/")
#   np.random.shuffle(all_image_paths)
  print("训练数据： ",len(all_image_paths))
  #test_image_paths = all_image_paths[180000:181000]
  all_image_paths = all_image_paths[:]#19998
  loss_list = train(mask_num = 1,batch_size = 6,epochs = 200,show_num = 5)
  
