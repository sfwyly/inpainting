#encoding:utf-8

"""
   @Author : sfwy
   @Date : 2020/9/9
   @Descriptiom ： 提取元素对 ， front -> back

    mask 生成
    
   1. mask rcnn 分割 对前背景数据进行分割
   2. sca unet 生成 对分割后的局部前景数据生成背景局部数据

"""
from PIL import Image
import pathlib
import numpy as np
import os

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg



# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import fashion


prefix_root = "F:/deepfashion/"

weights_path = "E:/cnn/pretrain/mask_rcnn_fashion_0019.h5" 


ROOT_DIR = os.path.abspath("/")
sys.path.append(ROOT_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = fashion.BalloonConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    # Give the configuration a recognizable name
    # Give the configuration a recognizable name
    NAME = "fashion"
    BATCH_SIZE = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + bag +top +boots

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 125

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    TRAIN_ROIS_PER_IMAGE = 200

config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
model.load_weights(weights_path, by_name=True)

def getMask(img_name):
  image = mpimg.imread(str(img_name))
  results = model.detect([image], verbose=1)
  r = results[0]

  class_ids = r["class_ids"]
  #分类1 为上衣
  if(1 not in class_ids):
    return np.array([])
  # 检索上衣位置
  pos = class_ids.tolist().index(1)
  mask = r["masks"][:,:,pos] #这里是True 与 False
  mask = np.array(mask,dtype=np.int32) #转为0 1
  return mask 

def getImage(image_root):

	#获取所有文件路径
	images_path = np.array(list(image_root.glob("*.jpg")))

	result = []
	#提取所有匹配的文件对路径
	if(len(images_path)<=0):
		return []
	for image_path in images_path:
		#图片
		if(str(image_path).find("1_front")>=0 and str(image_path).find("1_front_IUV")<0):

			#截取编号
			back_path = str(image_path)[:str(image_path).find("1_front")]+"3_back.jpg"
			if(os.path.exists(back_path)):
				result.append((str(image_path),back_path))
			pass
	return result

#女衣服种类
women_catagory = ["Blouses_Shirts","Cardigans","Dresses","Graphic_Tees","Jackets_Coats","Rompers_Jumpsuits","Sweaters","Sweatshirts_Hoodies","Tees_Tanks"]

men_catagory = ["Jackets_Vests","Shirts_Polos","Suiting","Sweaters","Sweatshirts_Hoodies","Tees_Tanks"]

def getFrontToBack(prefix = "F:/deepfashion/",flag = "WOMEN"):
  result = []#遍历每个图像的根
  if(flag =="WOMEN"):
    path_root = pathlib.Path(prefix+"/In-shop Clothes Retrieval Benchmark/Anno/densepose/img/WOMEN/")
    for catagory in women_catagory:
      images_root = np.array(list(path_root.glob(catagory+"/*")))
      for image_root in images_root:
        result.extend(getImage(image_root))
  else:
    path_root = pathlib.Path(prefix+"/In-shop Clothes Retrieval Benchmark/Anno/densepose/img/MEN/")
    for catagory in men_catagory:
      images_root = np.array(list(path_root.glob(catagory+"/*")))
      for image_root in images_root:
        result.extend(getImage(image_root))
  return result

def saveMask(mask,path,flag = "front"):
  mask = mask[:,:,np.newaxis]
  mask = np.concatenate([mask,mask,mask],axis=-1)*255
  image = Image.fromarray(np.uint8(mask))
  path =prefix_root+"/"+flag+"/"+ path[path.find("In-shop"):]
  route = path[:path.rfind("\\")]
  if(not os.path.exists(route)):
    os.makedirs(route)
  image.save(path)

# step1 :保存mask
#输入图像对 （front - back）
# 读取时刚好进行分割局部  图片全为  256*256
def proceedSegmentation(all_image_paths):

  train = []
  #遍历前背景路径进行读取
  for front_path,back_path in all_image_paths:
    front_path,bach_path  = str(front_path),str(back_path)
   
    # front = np.array(Image.open(front_path))
    # back = np.array(Image.open(back_path)) #直接是256*256

    front_mask , back_mask = getMask(front_path),getMask(back_path)
    #这里mask可能为None
    if(front_mask.size==0 or back_mask.size == 0):
      continue

    #保存mask
    saveMask(front_mask,front_path,flag = "front")
    saveMask(back_mask,back_path,flag = "back")

proceedSegmentation(getFrontToBack(prefix_root,flag="MEN"))
