from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from keras.optimizers import SGD
import os

MODEL_H5_NAME = "model.h5"
IMG_SIZE = [160,160,1]
IMG_SUBDIVIDE = 80


def doesModelExist():
	return os.path.isfile(MODEL_H5_NAME)

def createModel(loadFromDisk):

	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[2])))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	
	model.add(Conv2D(256, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(2048))
	model.add(Activation('relu'))
	model.add(Dense(IMG_SUBDIVIDE+IMG_SUBDIVIDE))
	model.add(Activation('sigmoid'))
	
	model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

	print(model.summary())
	
	if loadFromDisk and os.path.isfile(MODEL_H5_NAME):
		model.load_weights(MODEL_H5_NAME)
	
	return model