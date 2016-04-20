import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
np.random.seed(1234)
from datagenerator import *

input_sz=2500
output_sz=24
dropout_ratio=0.5
hidden_neurons=2000
model=Sequential()
model.add(LSTM(
		input_dim=input_sz,
		output_dim=hidden_neurons,
		return_sequences=False)
	)
model.add(Dropout(dropout_ratio))
model.add(Dense(24))
model.add( Activation('softmax') )

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

model.fit_generator(datagen(), samples_per_epoch = 128, nb_epoch = 20000, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=8)
