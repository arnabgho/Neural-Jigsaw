import numpy as np
import os
from PIL import Image
import pickle
import caffe
import sklearn.preprocessing as preprocessing
from numpy import newaxis


caffe_root='/data/gpuuser/CAFFE/caffe/'


pwd=os.getcwd()

# bvlc_alexnet
net=caffe.Net(caffe_root+'models/bvlc_alexnet/train_val.prototxt',caffe_root+'/models/bvlc_alexnet/bvlc_alexnet.caffemodel',caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
data_total=[]

IMG_SIZE=256

for directory in os.listdir('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP'):
	res_images=[]
	for filename in os.listdir('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/'+directory):
		pic=Image.open('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/'+directory+'/'+filename)
		pic=pic.resize( (IMG_SIZE,IMG_SIZE) , Image.ANTIALIAS )
		pic_data=np.array(pic)
		pic_data = pic_data[:,:,::-1]
		pic_data = pic_data.transpose((2,0,1))
		pic_data=pic_data[newaxis,:,:,:]
		net.blobs['data'].data[...] = pic_data
		out=net.forward()
		res_list=net.blobs['fc7'].data[0]
		res_np=np.array(res_list)
		res=preprocessing.scale(res_np)
		res_images.append(res)
	data_total.append(res_images)

with open('total_data.pickle', 'wb') as handle:
	pickle.dump(data_total,handle)


