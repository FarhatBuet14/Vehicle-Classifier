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

datafile = './Data/datafile.npz'

weightFile = './WeightFile/best.hdf5'
####################### Loading the Data ################################


dataset = np.load(datafile)

X_train = dataset['X_train']
y_train = dataset['y_train']
X_test = dataset['X_test']
y_test = dataset['y_test']
X_val = dataset['X_val']
y_val = dataset['y_val']


num_classes = y_train.shape[1]
imageSize = X_train.shape[1]

epochs = 100
batch_size = 32
verbose = 1

####################### Defining the model ##############################

model = Sequential()
model.add(Convolution2D(32, 3, 3 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(Convolution2D(32, 3, 3 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(64, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(Convolution2D(32, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(128, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(Convolution2D(128, 2, 2 , 
                        input_shape=(imageSize,imageSize,3),activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))


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





























