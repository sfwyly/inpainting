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

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
# import functools

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
def getAllImagePath(name):
  path_root = pathlib.Path(name)
  return list(path_root.glob("*.jpg"))

def getDataset(all_image_paths):

  train = []
  # labels = []
  for path in all_image_paths:
    path = str(path)
    #添加图片
    image = Image.open(path)
    
    image = image.resize((256,256),Image.BILINEAR)
    train.append(np.asarray(image))
    # #添加掩码
    # path_length = len(path)
    # image_number = path[path_length-8:path_length-4]
    # data, flag = readpgm("/content/drive/My Drive/data/sky/groundtruth/"+str(image_number)+"_gt.pgm")
    # if(flag):
    #   data = np.reshape(data[0],data[1])
    # labels.append(dealData(data))
  return np.array(train)

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
    maxout = self.MLP(1./(self.max_pool(input)+10e-6))
    return self.sigmoid(avgout+maxout)
  def get_config(self):
    config = {"channels":self.channels}
    base_config = super(ChannelAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self,shape=(32,32,128), kernel_size=7,name="SpatialAttention",**kwargs):
        super(SpatialAttention, self).__init__(name=name,**kwargs)
        # assert kernel_size in (3,7), "kernel size must be 3 or 7"
        # padding = 3 if kernel_size == 7 else 1
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1,kernel_size,padding="same",use_bias=False)
        self.sigmoid = tf.keras.activations.sigmoid
        self.bias = self.add_weight(shape=[shape[0],shape[1],shape[2]],initializer=tf.random_uniform_initializer(),trainable = True)
    def build(self, input_shape):
        # self.kernel = self.add_weight("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        pass
    def call(self, input,mask):#batch 32 32 128
        i_s = input.get_shape().as_list()

        avgout = tf.reduce_mean(input, axis=-1, keepdims=True)
        maxout= tf.reduce_max(input, axis=-1, keepdims=True)
        
        #mask : 1 h w 1
        x = layers.concatenate([avgout, maxout], axis=-1)
        y = self.contextualAttention(x,mask)#batch 32 32 2
        inpainting = tf.concat([y[...,0][...,tf.newaxis] for _ in range(i_s[3])],axis=-1)+self.bias
        # inpainting = self.contextualAttention(input,mask)
        x = self.conv(x)
        return self.sigmoid(x)*(input + inpainting*(1-mask))
    def get_config(self):
        config = {"kernel_size":self.kernel_size}
        base_config = super(SpatialAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def contextualAttention(self,x,mask,kernel=3):
      x_s = x.get_shape().as_list()
      x_patches = tf.image.extract_patches(x, [1,kernel,kernel,1], [1,1,1,1], [1,1,1,1], padding='SAME')
      x_patches = tf.reshape(x_patches, [x_s[0], -1, kernel, kernel, x_s[3]]) 
     
      x_patches = tf.transpose(x_patches, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
      mask_patches = tf.image.extract_patches(mask, [1,kernel,kernel,1], [1,1,1,1], [1,1,1,1], padding='SAME')
      mask_patches = tf.reshape(mask_patches, [1, -1, kernel, kernel, 1])
      mask_patches = tf.transpose(mask_patches, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
      mm = tf.cast(tf.equal(tf.reduce_mean(mask_patches[0], axis=[0,1,2], keepdims=True), 1.), tf.float32)

      y = []
      for xi, wi in zip(tf.split(x, x_s[0], axis=0),tf.split(x_patches, x_s[0], axis=0)):
        wi = wi[0]
        xi = tf.cast(xi,tf.float32)
        wi = tf.cast(wi,tf.float32)
        
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")
        yi = tf.reshape(yi, [1, x_s[1], x_s[2], x_s[1]*x_s[2]])

        yi *=  mm  # mask
        yi = tf.nn.softmax(yi, 3)
        yi *=  mm  # mask

        yi = tf.nn.conv2d_transpose(yi, wi, tf.concat([[1], x_s[1:]], axis=0), strides=[1,1,1,1]) / 4.
        y.append(yi)
      
      y = tf.concat([y[_] for _ in range(len(y))],axis=0)
      return y

class PatialConv(tf.keras.layers.Layer):
    def __init__(self, kernel=3,dilation_rate=1 ,stride=2,channels = 32,name="PatialConv",**kwargs):
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
        self.dense1 = tf.keras.layers.Conv2D(filters=1,kernel_size=self.kernel,dilation_rate = self.dilation_rate, kernel_initializer=tf.keras.initializers.Ones(),strides=self.stride, padding="same", use_bias=False, trainable=False)
        self.dense2 = tf.keras.layers.Conv2D(self.channels,kernel_size=self.kernel,dilation_rate = self.dilation_rate,strides=self.stride, padding="same", use_bias=False, trainable=True)
        self.bias = self.add_weight(shape=(self.channels,),initializer=tf.constant_initializer(0.0),trainable=True)
    def call(self, input,mask):

        update_mask = self.dense1(mask)
        mask_ratio = self.slide_window / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        output = self.dense2(input)
        output = output*mask_ratio
        
        return (output+self.bias)*update_mask , update_mask
    def get_config(self):
       config = {"kernel":self.kernel,"stride":self.stride,"channels":self.channels,"dilation_rate":self.dilation_rate}
       base_config = super(PatialConv, self).get_config()
       return dict(list(base_config.items()) + list(config.items()))

_custom_objects = {
    "ChannelAttention" :  ChannelAttention,
   "SpatialAttention" : SpatialAttention,
   "PatialConv" :PatialConv
}

#改 基于部分卷积的channel-wise注意力

class Partial_UNet():
    def __init__(self):
        print ('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2])
        # print(target.shape,refer.shape)
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

    def create_model(self, img_shape, num_class):

        concat_axis = 3
        inputs = layers.Input(batch_shape = (6,256,256,3),name = "img_input")
        mask = layers.Input(shape=(256,256,1),name="mask_input")

        # conv1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same',name="conv1_1",kernel_initializer="random_normal")(inputs)
        conv1,new_mask = PatialConv(kernel=7, stride=1,channels = 64,name="PatialConv1",trainable = True)(inputs,mask)
        conv1_2,new_mask = PatialConv(kernel=7, stride=2,channels = 64,name="PatialConv1_2",trainable = True)(conv1,new_mask)
        # conv1 = layers.Conv2D(64, (7, 7), activation='relu', padding='same',name="conv1_11",kernel_initializer="random_normal")(conv1)
        # conv1_2 = layers.Conv2D(64, (7, 7), activation='relu',strides=2, padding='same', name='conv1_2' , kernel_initializer="random_normal")(conv1)

        conv2,new_mask = PatialConv(kernel=5, stride=1,channels = 128,name="PatialConv2",trainable = True)(conv1_2,new_mask)
        conv2_2,new_mask = PatialConv(kernel=5, stride=2,channels = 128,name="PatialConv2_2",trainable = True)(conv2,new_mask)
        # conv2 = layers.Conv2D(128, (5, 5), activation='relu', padding='same',name="conv2_1",kernel_initializer="random_normal")(conv1_2)
        # conv2_2 = layers.Conv2D(128, (5, 5), activation='relu',strides=2, padding='same',name='conv2_2',kernel_initializer="random_normal")(conv2)
        

        conv3,new_mask = PatialConv(kernel=3, stride=1,channels = 128,name="PatialConv3",trainable = True)(conv2_2,new_mask)
        conv3_2,new_mask = PatialConv(kernel=3, stride=2,channels = 128,name="PatialConv3_2",trainable = True)(conv3,new_mask)
        # conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',name="conv3_1",kernel_initializer="random_normal")(conv2_2)
        # conv3_2 = layers.Conv2D(128, (3, 3), activation='relu',strides=2, padding='same',name='conv3_2',kernel_initializer="random_normal")(conv3)

        #spacial attention
        conv3_2 = SpatialAttention(kernel_size=3,name="SpatialAttention",trainable=True)(conv3_2,new_mask)

        conv4,new_mask = PatialConv(kernel=3, stride=1,channels = 256,name="PatialConv4",trainable = True)(conv3_2,new_mask)
        conv4_2,new_mask = PatialConv(kernel=3, stride=2,channels = 256,name="PatialConv4_2",trainable = True)(conv4,new_mask)
        # conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',name="conv4_1",kernel_initializer="random_normal")(conv3_2)
        # conv4_2 = layers.Conv2D(256, (3, 3), activation='relu',strides=2, padding='same',name='conv4_2',kernel_initializer="random_normal")(conv4)

        # this reweight by mask conv  16*16*256 * 16*16*1
        # weight_inputs = layers.Input(shape = (16,16,1),name="weight_input")
        # conv4_2 = layers.Multiply()([conv4_2,weight_inputs])

        #全局映射层
        # pool4 = pool4 *input_weight
        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',name="conv5_1")(conv4_2)
        global_conv = layers.Conv2D(512, (1, 1), activation='relu', padding='same',name="global_conv1")(conv5)
        global_conv = layers.Conv2D(512, (3, 3),dilation_rate=(2,2), activation='relu', padding='same',name="global_conv2")(global_conv)
        global_conv = layers.Conv2D(512, (3, 3),dilation_rate=(2,2), activation='relu', padding='same',name="global_conv3")(global_conv)
        global_conv = layers.Conv2D(512, (3, 3),dilation_rate=(2,2), activation='relu', padding='same',name="global_conv4")(global_conv)
        global_conv = layers.Conv2D(512, (3, 3),dilation_rate=(2,2), activation='relu', padding='same',name="global_conv5")(global_conv)
        global_conv = layers.Conv2D(512, (1, 1), activation='relu', padding='same',name="global_conv6")(global_conv)
        conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',name='conv5_2')(conv5)
       

        up_conv5 = layers.UpSampling2D(size=(2, 2),name="up1_1")(conv5)
        up_conv5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',name="up1_2")(up_conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = layers.Cropping2D(cropping=(ch,cw))(conv4)
        up6 = layers.concatenate([up_conv5, crop_conv4],name='cont1', axis=concat_axis)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',name="conv6_1")(up6)
        conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',name="conv6_2")(conv6)

        up_conv6 = layers.UpSampling2D(size=(2, 2),name="up2_1")(conv6)
        up_conv6 = layers.Conv2D(128,(3, 3), activation='relu', padding='same',name="up2_2")(up_conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = layers.Cropping2D(cropping=(ch,cw))(conv3)
        up7 = layers.concatenate([up_conv6, crop_conv3],name='cont2', axis=concat_axis) 
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',name="conv7_1")(up7)
        conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',name="conv7_2")(conv7)

        #通道注意力
        attention7 = ChannelAttention(channels=128,ratio=16,name="ChannelAttention",trainable=True)(conv7)
        conv7 = tf.transpose(tf.transpose(conv7,(1,2,0,3))*attention7,(2,0,1,3))

        up_conv7 = layers.UpSampling2D(size=(2, 2),name="up3_1")(conv7)
        up_conv7 = layers.Conv2D(128,(3, 3), activation='relu', padding='same',name="up3_2")(up_conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = layers.Cropping2D(cropping=(ch,cw))(conv2)
        up8 = layers.concatenate([up_conv7, crop_conv2],name='cont3', axis=concat_axis)
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',name="conv8_1")(up8)
        conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',name="conv8_2")(conv8)

        up_conv8 = layers.UpSampling2D(size=(2, 2),name="up4_1")(conv8)
        up_conv8 = layers.Conv2D(64,(3, 3), activation='relu', padding='same',name="up4_2")(up_conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = layers.Cropping2D(cropping=(ch,cw))(conv1)
        up9 = layers.concatenate([up_conv8, crop_conv1],name='cont4', axis=concat_axis)
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',name="conv9_1")(up9)
        conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',name="conv9_2")(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)
        conv9 = layers.ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        # conv9 = layers.Conv2D(3, (3, 3),activation='relu', padding='same',name='conv10')(conv9)
        conv10 = layers.Conv2D(3, (1, 1),activation="sigmoid",name='conv10_1')(conv9)

        model = models.Model(inputs=[inputs,mask], outputs=conv10)#[conv1,conv2,conv3,conv4,conv5,up6,up7,up8,up9,

        return model
generator = Partial_UNet().create_model((256,256,3),3)
# generator.summary()
# generator.compile(optimizer=tf.keras.optimizers.RMSprop(0.003), loss='mean_absolute_error',metrics=['accuracy'])

def getDiscriminator(image_shape,num_class):
   inputs = layers.Input(shape=image_shape)

   conv1 = layers.Conv2D(32, kernel_size=3, input_shape=image_shape,activation="elu", padding="same")(inputs)
   conv1 = layers.Conv2D(32, kernel_size=3,activation="elu", padding="same")(conv1)
   pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)
  
   conv2 = layers.Conv2D(64, kernel_size=3,activation="elu", padding="same")(pool1)
   conv2 = layers.Conv2D(64, kernel_size=3,activation="elu", padding="same")(conv2)
   pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
   # 7, 7, 64 -> 4, 4, 128
   conv3 = layers.Conv2D(128, kernel_size=3,activation="elu", padding="same")(pool2)
   conv3 = layers.Conv2D(128, kernel_size=3,activation="elu", padding="same")(conv3)
   pool3 = layers.MaxPooling2D(pool_size=(2,2))(conv3)

   conv4 = layers.Conv2D(128, kernel_size=3,activation="elu", padding="same")(pool3)
   conv4 = layers.Conv2D(128, kernel_size=3,activation="elu", padding="same")(conv4)
   pool4 = layers.MaxPooling2D(pool_size=(2,2))(conv4)

   conv5 = layers.Conv2D(256, kernel_size=4,activation="elu", padding="same")(pool4)
   conv5 = layers.Conv2D(256, kernel_size=4,activation="elu", padding="same")(conv5)
   pool5 = layers.MaxPooling2D(pool_size=(2,2))(conv5)

   conv6 = layers.Conv2D(256, kernel_size=4,activation="elu")(pool5)
   conv6 = layers.Conv2D(256, kernel_size=4,activation="elu")(conv6)

  #  global_average = layers.GlobalAveragePooling2D()(conv5)
  #  global_max = layers.GlobalMaxPooling2D()(conv5)

  #  cont = layers.concatenate([global_average,global_max],axis=-1)

   flatten = layers.Flatten()(conv6)
   # 全连接
   outputs = layers.Dense(1, activation='sigmoid')(flatten)

   return models.Model(inputs=inputs, outputs = outputs,name="discriminator")

discriminator = getDiscriminator((256,256,3),1)
# discriminator.summary()#binary_crossentropy
# # discriminator.compile(loss='binary_crossentropy',
# #             optimizer=tf.keras.optimizers.RMSprop(0.0003),
# #             metrics=['accuracy'])


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
def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

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
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False
    # self.vgg.summary()

  def call(self, inputs):
    "Expects float input in [0,1]"
    # inputs = inputs*255.0
    # preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(inputs)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])
    #style层的原始输出
    perceptual_dict = {style_name:value/(1.*value.shape[1]*value.shape[2]*value.shape[3]) 
                    for style_name, value 
                    in zip(self.style_layers, style_outputs)}
    ##求gram矩阵
    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'perceptual':perceptual_dict, 'style':style_dict}


# content_targets = extractor(object_image)['content']
# image = np.random.normal(0,1,(400,600,3))
# image = tf.Variable(style_image[tf.newaxis,...], dtype=tf.float32) 
# image = tf.Variable(np.concatenate((a,b,c),axis=2)[tf.newaxis,...], dtype=tf.float32)
def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.0)


# style_weight=1e-2
# content_weight=1e4

# content_image 其实就是图片显示区域，style_image是局部样式，必须保持content_image的强一致
def style_content_loss(outputs,style_targets):
    # print("样式")
    style_outputs = outputs['style']
    style_targets = style_targets['style']

    # content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    # style_loss *= style_weight / num_style_layers

    # content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
    #                          for name in content_outputs.keys()])
    # content_loss *= content_weight / num_content_layers
    # loss = style_loss + content_loss
    return style_loss 
def l1_loss(y_pred,y_true):
  # print("l1")
  y_pred = tf.cast(y_pred,dtype=tf.float32)
  y_true = tf.cast(y_true,dtype=tf.float32)
  # TODO 未来可以看看是不是可以看 求sum
  # return tf.reduce_mean(tf.reduce_sum(tf.abs(y_pred-y_true),(1,2,3)))
  new_mask = np.array([mask,mask,mask])
  new_mask = np.transpose(new_mask,(1,2,0))
  return tf.reduce_mean(10.*(tf.reduce_sum(tf.abs(y_pred-y_true)*(1-new_mask),(1,2,3)))+1.*(tf.reduce_sum(tf.abs(y_pred-y_true)*new_mask,(1,2,3))))

def cal_perceptual(outputs,style_targets):
  # print("样")
  style_outputs = outputs['perceptual']
  style_targets = style_targets['perceptual']
  # loss = 0.
  # for name in style_outputs.keys():
  #   print(style_outputs[name].shape)
  #   print(style_targets[name].shape)
  #   batch, channel ,channel = style_outputs[name].shape
  #   loss = loss + tf.reduce_mean((tf.abs(style_outputs[name]-style_targets[name])))*1.0/(channel*channel)
  result = tf.add_n([tf.reduce_mean(tf.reduce_sum(tf.math.abs(style_outputs[name]-style_targets[name]),(1,2,3))) for name in style_outputs.keys()])
  return result
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
#这里暂时只计算生成器的对抗
def cal_adv(real_image_list,fake_image_list):
  # print("对抗")
  real_loss = cross_entropy(tf.ones_like(real_image_list), real_image_list)
  fake_loss = cross_entropy(tf.zeros_like(fake_image_list), fake_image_list)
  total_loss = real_loss + fake_loss
  return total_loss
def cal_gen(fake_image_list):
  return cross_entropy(tf.ones_like(fake_image_list), fake_image_list)

def resize_image(image):
  image = Image.fromarray(image)
  image = image.resize((224,224),Image.BILINEAR)
  return np.array(image)

extractor = StyleContentModel(style_layers, content_layers)
generator_optimizer = tf.optimizers.Adam(learning_rate=0.0001)
discriminator_optimizer = tf.optimizers.Adam(learning_rate=0.0001,beta_1=0.5)
@tf.function()
def train_step(image_list,image_list_vgg,mask_image_list,discriminator_num): #image为原图 0 - 1.0
  with tf.GradientTape() as tape ,tf.GradientTape() as disc_tape:
    #需要转换成224**224
    style_targets = extractor(image_list_vgg) #这里的输出其实将content层与style层都获取到了，但是只用得到style
    gen_image_list = generator([mask_image_list,mask[np.newaxis,:,:,np.newaxis]])

    dis_image_list = discriminator(gen_image_list)
    real_image_list = discriminator(image_list)

    # gen_image_list,dis_image_list = combined([mask_image_list,maskconv_list])
    #TODO 需要先转换成224**224
    gen_vgg_list = tf.image.resize(gen_image_list, (224, 224), method="bilinear")
    # print(gen_vgg_list.shape)
    outputs = extractor(gen_vgg_list) #通过生成图像导出特征层 与 原始图像特征层对比
    style_loss = 12.*style_content_loss(outputs,style_targets)
    adver_loss = 0.3*cal_gen(dis_image_list)
    L1_loss = 1.*l1_loss(gen_image_list,image_list)
    tvl_loss = 0.01*total_variation_loss(gen_image_list)
    perceptual_loss = 0.7*cal_perceptual(outputs,style_targets)
    
    loss = style_loss+ adver_loss+ L1_loss + tvl_loss +  perceptual_loss

    #判别器
    disc_loss = cal_adv(real_image_list,dis_image_list)
  grads = tape.gradient(loss, generator.trainable_variables)
  # grad_list = []
  # for grad,variable in zip(grads,generator.trainable_variables):
  #   grad_list.append((grad,variable))
  generator_optimizer.apply_gradients(zip(grads,generator.trainable_variables))
  #判别器
  gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  return loss,style_loss,adver_loss,L1_loss,tvl_loss,perceptual_loss,disc_loss
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
def getMaskList(files_num,prefix):
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

def train(mask_num = 1,batch_size = 6,epochs = 100,discriminator_num = 1,show_num = 5):
  global mask
  # mask_num = 1 #mask数量
  # batch_size = 6 #批量大小
  # epochs = 100 #执行次数
  # discriminator_num = 1 #每次判别器执行次数
  train_length = len(all_image_paths)
  loss_list = []
  #多个掩码
  # mask_list = getHoles((256,256),mask_num)
  mask_list = getMaskList(10,"/usr/jjj/mask/")
  for load_train in range(len(all_image_paths)//120): #最后一个迭代用来测试 train_length//(10*batch_size) -1
    #初始化数据
    X_train = getDataset(all_image_paths[load_train*(20*batch_size):(load_train+1)*(20*batch_size)])
    
    X_labels = X_train/255.

    X_labels_vgg = np.array(list(map(resize_image,X_train)))/255.
    
    for mask_data in mask_list:
      #菜鸡 看到没 这里的mask变了需要更新新的训练集
      mask = mask_data
      # maskconv = maskConv(mask[np.newaxis,:,:,np.newaxis])[0]
      # maskconv_list = np.array([maskconv for n in range(batch_size)])
      X_ = np.array(list(map(multimask,X_train)))/255.
      for i in range(epochs):
        #对所有数据进行一轮循环学习
        id_list = [j for j in range(X_train.shape[0])]
        np.random.shuffle(id_list)
        for t in range(math.ceil(X_train.shape[0]/batch_size)):
          idx = id_list[t*batch_size:(t+1)*batch_size]
          # idx = np.random.randint(0, X_train.shape[0], batch_size)
          loss,style_loss,adver_loss,L1_loss,tvl_loss,perceptual_loss,disc_loss = train_step(X_labels[idx],X_labels_vgg[idx],X_[idx],discriminator_num) #训练生成器
        loss_list.append([loss.numpy(),style_loss.numpy(),adver_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy(),disc_loss.numpy()])  
        if((i+1)%show_num==0):
          print("训练第 ",i+1," 轮 损失",loss.numpy(),style_loss.numpy(),adver_loss.numpy(),L1_loss.numpy(),tvl_loss.numpy(),perceptual_loss.numpy(),disc_loss.numpy())
          generator.save("/usr/jjj/inpainting/generator_partial_attention.h5")
          discriminator.save("/usr/jjj/inpainting/discriminator_partial_attention.h5")
  return loss_list


if(__name__=="__main__"):
  all_image_paths = getAllImagePath("/usr/jjj/imgs/")
  loss_list = train(mask_num = 1,batch_size = 6,epochs = 20,discriminator_num = 1,show_num = 5)
  
