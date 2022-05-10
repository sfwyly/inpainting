
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from skimage.measure import compare_ssim as ssim


def getPSNR(image , true_image):
  height,width,_ = image.shape
  channel_mse = np.sum((image-true_image)**2,axis = (0,1))/(height*width)
  mse = np.mean(channel_mse)
  Max = 1.0 #最大图像
  PSNR = 10.*np.log10(Max**2/mse) #峰值信噪比

  return PSNR
import cv2

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

			startX = np.random.randint(imageHeight)
			startY = np.random.randint(imageWidth)

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

from PIL import Image
import pathlib
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
    mask_list.append((1-np.array(image))[...,np.newaxis])
  return np.array(mask_list)

mask_paths = getMaskListPaths("F:/mask/testing_mask_dataset/")
mask_list = getMaskList(mask_paths[:10])
print(mask_list.shape)

plt.imshow(mask_list[0][...,0])
plt.show()

a = mask_list[0][...,0]
print(a)

# def multimask(image):
#   # x,y,_ = image.shape
#   return np.transpose(np.transpose(image,(2,0,1)) *mask  ,(1,2,0))

# def saveMask(mask,name,prefix="/content/drive/My Drive/data/inpainting/mask/"):
#   mask = mask[:,:,np.newaxis]
#   mask = np.concatenate([mask,mask,mask],axis=-1)*255
#   image = Image.fromarray(np.uint8(mask))
#   image.save(prefix+name+".jpg")
# def readMask(name,prefix ="/content/drive/My Drive/data/inpainting/mask/"):
#   image = Image.open(prefix+name+".jpg")
#   image = np.array(image)/255
#   # print(image.shape)
#   return image[:,:,0]
# def getMaskList(prefix,files_num):
#   #处理 几位数 添0
#   file_list = [str(_) for _ in range(1,files_num+1)]
#   mask_list = []
#   for _ in file_list:
#     mask = readMask(_,prefix)

#     mask = np.array(mask,np.float32)
#     for i in range(256):
#       for j in range(256):
#         if(mask[i,j]<0.5):
#           mask[i,j] = 0
#         else:
#           mask[i,j] =1
#     mask_list.append(mask)
    
#   return np.array(mask_list)


# maskList = getMaskList("E:/cnn/mask/",10)
# mask = maskList[0]

# img = Image.open("E:/cnn/test/000631_3.jpg")
# img = np.array(img.resize((256,256), Image.ANTIALIAS))/255.

    
# ref = Image.open("E:/cnn/PM/000631_3_0.jpg")
# ref = np.array(ref.resize((256,256), Image.ANTIALIAS))/255.


# print(np.mean(np.abs(img-ref)))
# print(getPSNR(img,ref))
# print(ssim(img,ref,multichannel=True,data_range=1.0))
