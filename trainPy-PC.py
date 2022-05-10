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
        mask_ratio = (self.slide_window*input.shape[-1]) / (update_mask + 1e-8)
        update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
        mask_ratio = mask_ratio * update_mask

        output = self.dense2(input)#-self.bias
        output = output*mask_ratio
        # output = self.relu(output*mask_ratio+self.bias)
        
        return output*update_mask , update_mask
    def get_config(self):
       config = {"kernel":self.kernel,"stride":self.stride,"channels":self.channels,"dilation_rate":self.dilation_rate}
       base_config = super(PatialConv, self).get_config()
       return dict(list(base_config.items()) + list(config.items()))



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
            if bn:
                conv = layers.BatchNormalization()(conv)
            conv = layers.LeakyReLU(alpha=0.2)(conv)
            return conv, mask
        
        conv1,mask1 = encoder_layer(inputs,mask,filters=64,kernel=5,strides=2,rate=1,name="PatialConv1",bn=False)
      
        conv2,mask2 = encoder_layer(conv1,mask1,filters=128,kernel=5,strides=2,rate=1,name="PatialConv2")

        conv3,mask3 = encoder_layer(conv2,mask2,filters=256,kernel=3,strides=2,rate=1,name="PatialConv3")
        
        conv4,mask4 = encoder_layer(conv3,mask3,filters=512,kernel=3,strides=2,rate=1,name="PatialConv4")

        conv5,mask5 = encoder_layer(conv4,mask4,filters=512,kernel=3,strides=2,rate=1,name="PatialConv5")
        
        conv6,mask6 = encoder_layer(conv5,mask5,filters=512,kernel=3,strides=2,rate=1,name="PatialConv6")
        
        conv7,mask7 = encoder_layer(conv6,mask6,filters=512,kernel=3,strides=2,rate=1,name="PatialConv7")       
        
        conv11,mask11 = decoder_layer(conv7,mask7,conv6,mask6,filters=512,kernel=3,strides=1,rate=1,name="PatialConv11")
       
        conv12,mask12 = decoder_layer(conv11,mask11,conv5,mask5,filters=512,kernel=3,strides=1,rate=1,name="PatialConv12")
        
        conv13,mask13 = decoder_layer(conv12,mask12,conv4,mask4,filters=512,kernel=3,strides=1,rate=1,name="PatialConv13")
         
        conv14,mask14 = decoder_layer(conv13,mask13,conv3,mask3,filters=256,kernel=3,strides=1,rate=1,name="PatialConv14")
        
        conv15,mask15 = decoder_layer(conv14,mask14,conv2,mask2,filters=128,kernel=3,strides=1,rate=1,name="PatialConv15")
        
        conv16,mask16 = decoder_layer(conv15,mask15,conv1,mask1,filters=64,kernel=3,strides=1,rate=1,name="PatialConv16")
        
        conv17,mask17 = decoder_layer(conv16,mask16,inputs,mask,filters=3,kernel=3,strides=1,rate=1,name="PatialConv17",bn=False)
        
        outputs = layers.Conv2D(3, (1, 1),activation="sigmoid",name='img_outputs')(conv17)

        model = models.Model(inputs=[inputs,mask], outputs=outputs)
        return model
generator = Partial_UNet().create_model()
# generator.trainable = True
generator.summary()


#generator.load_weights("/home/oyx/jjj/inpainting/deepfashion/PC_deepfashion.h5")

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
        generator.save_weights("/home/oyx/jjj/inpainting/deepfashion/PC_deepfashion.h5")

  return loss_list


if(__name__=="__main__"):
  #all_image_paths = getAllImagePath("/root/sfwy/inpainting/CeleAHQ/img_align_celeba/")#/root/sfwy/inpainting/Paris_StreetView_Dataset/paris_train_original/
  all_image_paths = getAllImagePath("/home/oyx/jjj/inpainting/deepfashion/deepfashion_child/")
#   np.random.shuffle(all_image_paths) /root/sfwy/inpainting/test_256/
  print("训练数据： ",len(all_image_paths))
  #test_image_paths = all_image_paths[180000:181000]
  all_image_paths = all_image_paths[:]#14898 19998
  loss_list = train(mask_num = 1,batch_size = 6,epochs = 200,show_num = 5)
  
