import numpy as np
import os
from PIL import Image
import pickle
import sklearn.preprocessing as preprocessing
from numpy import newaxis




pwd=os.getcwd()

# bvlc_reference_caffenet

data_total=[]

IMG_SIZE=50

for directory in os.listdir('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP'):
	res_images=[]
	for filename in os.listdir('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/'+directory):
		print ('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/'+directory+'/'+filename)
		pic=Image.open('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/'+directory+'/'+filename)
		pic=pic.resize( (IMG_SIZE,IMG_SIZE) , Image.ANTIALIAS )
		pic_data=np.asarray(pic.convert('L'),dtype=float)
		res_np=pic_data.flatten()
		res=preprocessing.scale(res_np)
		res_images.append(res)
	with open('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/'+directory+'/'+'total_data_raw.pickle', 'wb') as handle:
		pickle.dump(res_images,handle)


