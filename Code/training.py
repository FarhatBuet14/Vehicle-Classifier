# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:07:19 2018

@author: Suhail
"""

##################### Library Imports ################################

import numpy as np
import pandas as pd
import matplotlib as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


#################### Environment & Variables ############################

from keras import backend as K
K.set_image_data_format('channels_last')

seed = 7
np.random.seed(seed)
epochs = 65
batch_size = 32
verbose = 1

fldr = './Own_Dataset/Train/'

weightFile = './Model/best.hdf5'
datafile = './Data/datafile.npz'

####################### Loading the Data ################################

dataset = np.load(datafile)

X = dataset['X']
y = dataset['y']


random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=random_seed)


X_train = X_train/255
X_test = X_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_train.shape[1]
imageSize = X_train.shape[1]


random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)


####################### Defining the model ##############################

model = Sequential()
model.add(Convolution2D(32, 3, 3 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(Convolution2D(32, 3, 3 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Convolution2D(64, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(Convolution2D(32, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Convolution2D(128, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(Convolution2D(128, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(128, activation= 'relu' ))
model.add(Dense(num_classes, activation= 'softmax' ))

#################### Summary of the Model ###############################

model.summary()

#################### Compiling the Model ################################
optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss= 'categorical_crossentropy' , 
              optimizer= optimizer , metrics=[ 'accuracy' ])

#################### Defining the Checkpoints ###########################

l_r = ReduceLROnPlateau(monitor='val_acc', factor=0.5, 
                                  patience=3, verbose=1, 
                                  min_lr=0.000001)

wigth  = ModelCheckpoint(weightFile, monitor = 'val_categorical_accuracy' )
callbacks = [wigth, l_r]



datagen = ImageDataGenerator(featurewise_center=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, y_train, batch_size = 32),
                    validation_data = datagen.flow(X_val, y_val, batch_size = 32),
                    steps_per_epoch = len(X_train) / 32, 
                    validation_steps = len(X_val) / 16, epochs = epochs, 
                    callbacks = callbacks, verbose = verbose)

model.save('Saved_Model.h5')



########################  Prediction  ###################

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)


labels =  ['Car','Bicycle','Bus']
cm = confusion_matrix(y_test, y_pred)
print(cm)

pd.DataFrame(cm).to_csv("CM.csv")



























