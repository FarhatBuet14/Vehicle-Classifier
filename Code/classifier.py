####################### Library Imports ################################

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D,MaxPooling2D

################# Environment and Variables #############################

from keras import backend as K
K.set_image_data_format('channels_last')

seed = 7
np.random.seed(seed)
num_classes = 3
imageSize = 128

weightFile = './Model/best.hdf5'
imagePath = 'download 2.jpg'

###################### Taking image input ###############################

img = cv2.imread(imagePath)
imgRes = cv2.resize(img,(imageSize,imageSize))

X_temp = []
X_temp.append(imgRes)
X = np.asarray(X_temp)
X = X/255

################## Defining & Loading Model #############################

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


model.load_weights(weightFile)

################### Predicting the Class ###############################

y = model.predict_classes(X)
classno = np.ndarray.tolist(y)


dict = {0: 'Car', 1: 'Bicycle', 2: 'Bus'}
objectClass = dict[classno[0]]
print(objectClass)

################### Previewing Prediction ##############################

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, objectClass,(50,50), font, 2, (200,255,0), 5, cv2.LINE_AA)
cv2.imshow('Prediction',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




