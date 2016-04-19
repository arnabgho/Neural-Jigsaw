from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD

from datagenerator import *

model=Sequential()

model.add( Dense( 5000,input_dim=10000,init='uniform' )  )
model.add( Activation ('relu')  )
model.add( Dropout(0.25) )

model.add( Dense( 10000,init='uniform'  )  )
model.add( Activation('relu') )
model.add( Dropout(0.25) )

model.add( Dense( 5000,init='uniform'  )  )
model.add( Activation('relu') )
model.add( Dropout(0.25) )


model.add( Dense( 1000,init='uniform'  )  )
model.add( Activation('relu') )
model.add( Dropout(0.25) )

model.add( Dense( 500,init='uniform'  )  )
model.add( Activation('relu') )
model.add( Dropout(0.25) )

model.add( Dense( 100,init='uniform'  )  )
model.add( Activation('relu') )
model.add( Dropout(0.25) )



model.add( Dense( 24,init='uniform'  )  )
model.add( Activation( 'softmax' ) )

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy' , optimizer=sgd )


#nb_epoch=30

#for e in range(nb_epoch):
#	print(" epoch %d " %e )
#	for X_train,Y_train in datagen():
#		model.fit(X_train,Y_train,batch_size=32,nb_epoch=1)


model.fit_generator(datagen(), samples_per_epoch = 100, nb_epoch = 20000, verbose=2, show_accuracy=True, callbacks=[], validation_data=None, class_weight=None, nb_worker=8)
