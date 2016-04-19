import pickle
import os
import numpy as np

from random import randint

pwd=os.getcwd()
from itertools import permutations, islice
def nth(iterable, n, default=None):
	return next(islice(iterable, n, None), default)
	



def datagen():
	while (1):
		for directory in os.listdir('/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP'):
			with open( "/data/gpuuser/IMAGENET/ILSVRC2015/Data/DET/MLP/data_MLP/"+directory+'/total_data_raw.pickle'  ) as f:
				x=pickle.load(f)
				rand=randint(0,23)
				perm=nth(permutations(range(4),4), rand)	
				y_val=np.zeros(24,)
				y_val[rand]=1
				x_val_stack=np.zeros((4,2500))
				i=0
				for p in perm:
					x_val_stack[i]=x[p]
					i+=1
				x_val=x_val_stack.flatten()
				x_val=x_val.reshape((1,10000))
				y_val=y_val.reshape((1,24))
				yield x_val,y_val


