
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np

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

class PartialConv(layers.Layer):
    def __init__(self, kernel=3,dilation_rate=1 ,strides=2,in_channels = 64,out_channels = 64,activation="relu",flag = True,mul = True,**kwargs):
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
        if(self.flag):
            self.leaky_relu = tf.nn.leaky_relu
            self.bn = layers.BatchNormalization()
        self.weights_updater =  tf.ones((self.kernel,self.kernel,self.in_channels,self.out_channels))
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
            return self.leaky_relu(self.bn(output)), update_mask
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
    def call(self,x1,x2,mask):
        
        x1s = x1.shape
        x2s = x2.shape
        bz,height,width,channels = x1s
        kernel = 2*self.rate
        raw_w = tf.image.extract_patches(x1, [1,self.kernel,self.kernel,1], [1,self.rate*self.strides,self.rate*self.strides,1], [1,1,1,1], padding='SAME')
        raw_w = tf.reshape(raw_w, [x1s[0], -1, self.kernel, self.kernel, x1s[-1]]) 
        raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1]) 
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
        m = tf.transpose(m, [0, 2, 3, 4, 1])
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
        
        return y
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
        avg_feat = self.mlp2(avg_feat)
        
        max_feat = self.mlp1(max_feat)
        max_feat = self.mlp2(max_feat)
        
        feat = avg_feat + max_feat
        
        feat = tf.reshape(feat,[bz,1,1,c,self.num_of_modules])
        feat = tf.transpose(feat,[4,0,1,2,3])
        feat = tf.nn.softmax(feat,axis = 0)
        outputs = tf.reduce_sum(patch_block*feat,axis = 0)
        
        outputs = tf.concat([outputs,x2],axis = -1)
        outputs = self.combiner(outputs)
        
        return outputs

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
        
        self.Pconv35 = PartialConv(in_channels = 512,out_channels = 512,kernel = 3,strides = 2,mul = False)
        self.Pconv36 = PartialConv(in_channels = 512,out_channels = 512,kernel = 3,strides = 2,mul = False)

        self.u_Pconv34 = PartialConv(in_channels = 1024,out_channels = 512,kernel = 3,strides = 1,mul = False)
        self.u_Pconv35= PartialConv(in_channels = 1024,out_channels = 512,kernel = 3,strides = 1,mul = False)

        self.atn = AttentionModule()
        
        self.conv = layers.Conv2D(64,kernel_size = 1,strides = 1,padding="same",use_bias = False)
        self.conv2 = layers.Conv2D(64,kernel_size = 1,activation="sigmoid",strides = 1,padding="same",use_bias = False)
        
        self.Tconv = layers.Conv2DTranspose(64,kernel_size = 4,strides = 2,padding="same",use_bias = False)
        self.bn3 = layers.BatchNormalization()
        self.tail1 = PartialConv(in_channels = 67,out_channels = 32,kernel = 3,strides = 1,flag = False)
        self.tail2 = Bottleneck(output_dim = 32,strides = 1)
        self.out = layers.Conv2D(3,kernel_size = 3,activation="sigmoid",strides = 1,padding="same")

    def call(self,input,mask):
        
        self.atn.att.att_scores_prev = None
        self.atn.att.masks_prev = None
        x, m = self.Pconv01(input,mask)
        x = tf.nn.relu(x)
        x , m = self.Pconv02(x, m) 
        
        x1_0,m1_0 = x,m
        x1_1, m1_1 = self.Pconv11(x1_0,m1_0)
        x1_1 = tf.nn.relu(x1_1)
        x1_2, m1_2 = self.Pconv12(x1_1,m1_1)
        x1_2 = tf.nn.relu(x1_2)
        
        u_x1_1,u_m1_1 = self.u_Pconv11(tf.concat([self.up(x1_2),x1_1],axis= -1),tf.concat([self.up(m1_2),m1_1],axis = -1))
        u_x1_0,u_m1_0 = self.u_Pconv10(tf.concat([self.up(u_x1_1),x1_0],axis= -1),tf.concat([self.up(u_m1_1),m1_0],axis = -1))
        
        x2_0,m2_0 = u_x1_0*u_m1_0,u_m1_0
        
        x2_1, m2_1 = self.Pconv11(x2_0,m2_0)
        x2_1 = tf.nn.relu(x2_1)
        x2_2, m2_2 = self.Pconv12(x2_1,m2_1)
        x2_2 = tf.nn.relu(x2_2)
        
        x2_3 = self.atn(x1_2,x2_2,m2_2)
        
        x2_3, m2_3 = self.Pconv23(x2_3,m2_2)
        x2_3 = tf.nn.relu(x2_3)
        x2_4, m2_4 = self.Pconv24(x2_3,m2_3)
        x2_4 = tf.nn.relu(x2_4)
        
        u_x2_3,u_m2_3 = self.u_Pconv23(tf.concat([self.up(x2_4),x2_3],axis= -1),tf.concat([self.up(m2_4),m2_3],axis = -1))
        u_x2_2,u_m2_2 = self.u_Pconv22(tf.concat([self.up(u_x2_3),x2_2],axis= -1),tf.concat([self.up(u_m2_3),m2_2],axis = -1))
        u_x2_1,u_m2_1 = self.u_Pconv11(tf.concat([self.up(u_x2_2),x2_1],axis= -1),tf.concat([self.up(m2_2),m2_1],axis = -1))
        u_x2_0,u_m2_0 = self.u_Pconv10(tf.concat([self.up(u_x2_1),x2_0],axis= -1),tf.concat([self.up(u_m2_1),m2_0],axis = -1)) 
        
        x3_0,m3_0 = u_x2_0*u_m2_0,u_m2_0
        
        x3_1, m3_1 = self.Pconv11(x3_0,m3_0)  
        x3_1 = tf.nn.relu(x3_1)
        x3_2, m3_2 = self.Pconv12(x3_1,m3_1)
        x3_2 = tf.nn.relu(x3_2)
        x3_3 = self.atn(x2_2,x3_2,m3_2)
        
        x3_3, m3_3 = self.Pconv23(x3_3,m3_2)
        x3_3 = tf.nn.relu(x3_3)
        x3_4, m3_4 = self.Pconv24(x3_3,m3_3)
        x3_4 = tf.nn.relu(x3_4)
        x3_5, m3_5 = self.Pconv35(x3_4,m3_4) 
        x3_5 = tf.nn.relu(x3_5)
        x3_6, m3_6 = self.Pconv36(x3_5,m3_5)
        x3_6 = tf.nn.relu(x3_6)
        
        u_x3_5,u_m3_5 = self.u_Pconv35(tf.concat([self.up(x3_6),x3_5],axis= -1),tf.concat([self.up(m3_6),m3_5],axis = -1)) 
        u_x3_4,u_m3_4 = self.u_Pconv34(tf.concat([self.up(u_x3_5),x3_4],axis= -1),tf.concat([self.up(u_m3_5),m3_4],axis = -1)) 
        u_x3_3,u_m3_3 = self.u_Pconv23(tf.concat([self.up(u_x3_4),x3_3],axis= -1),tf.concat([self.up(u_m3_4),m3_3],axis = -1)) 
        u_x3_2,u_m3_2 = self.u_Pconv22(tf.concat([self.up(u_x3_3),x3_2],axis= -1),tf.concat([self.up(u_m3_3),m3_2],axis = -1))
        u_x3_1,u_m3_1 = self.u_Pconv11(tf.concat([self.up(u_x3_2),x3_1],axis= -1),tf.concat([self.up(m3_2),m3_1],axis = -1))
        u_x3_0,u_m3_0 = self.u_Pconv10(tf.concat([self.up(u_x3_1),x3_0],axis= -1),tf.concat([self.up(u_m3_1),m3_0],axis = -1))
        
        x3,m3 = u_x3_0,u_m3_0
        
        c_x = tf.concat([x2_0,x3_0,x3],axis = -1)
        c_m = tf.concat([m2_0,m3_0,m3],axis = -1)
        c_x = self.conv(c_x)
        c_m = self.conv2(tf.concat([c_x,c_m],axis = -1))
        
        c_x = c_x * c_m
        x3,m3 = c_x,c_m
        
        x4 = self.Tconv(x3)
        x4 = tf.nn.leaky_relu(self.bn3(x4))
        m4 = tf.image.resize(m3,(m3.shape[1]*2,m3.shape[2]*2),"bilinear")
        x5 = tf.concat([input,x4],axis = -1)
        m5 = tf.concat([mask,m4],axis = -1)
        
        x5,_ = self.tail1(x5,m5)
        x5 = tf.nn.leaky_relu(x5)
        x6 = self.tail2(x5)
        x6 = tf.concat([x5,x6],axis = -1)
        output = self.out(x6)

        return output,m1_0,m2_0,m3_0,m3
inputs = layers.Input(batch_shape = (6,256,256,3))
masks  = layers.Input(batch_shape = (6,256,256,3))
outputs = PFRNet()(inputs,masks)
generator  = models.Model(inputs = [inputs,masks],outputs = outputs)
