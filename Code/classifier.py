import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout, BatchNormalization
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

################# Environment and Variables #############################

from keras import backend as K
K.set_image_data_format('channels_last')

seed = 7
np.random.seed(seed)
num_classes = 3
imageSize = 128

weightFile = './WeightFile/best.hdf5'
imagePath = 'download 2.jpg'
datafile = './Data/datafile.npz'


####################### Loading the Data ################################

dataset = np.load(datafile)

X_test = dataset['X_test']
y_test = dataset['y_test']

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


################### Predicting the Class ###############################

y_pred = model.predict_classes(X_test)

y_test = np.argmax(y_test, axis=1)


labels =  ['Car','Bicycle','Bus']
cm = confusion_matrix(y_test, y_pred)
print(cm)


pd.DataFrame(cm).to_csv("CM.csv")


error_image = X_test[(y_test != y_pred)]
error_prediction = y_test[(y_test != y_pred)]
        

###########################  Showing the error images  ############################


def show_error_images(error_image, error_prediction, img_per_fig = 40, num_rows = 4):
    
    dict = {0: 'Car', 1: 'Bicycle', 2: 'Bus'}
    index = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for fig_count in range(0, int(len(error_image)/img_per_fig)):
        
        img = np.zeros((error_image.shape[1], 
                           error_image.shape[2] * (int(img_per_fig/num_rows)), 
                           error_image.shape[3]))
        
        for row in range(0, num_rows):
            
            row_img = np.zeros((error_image.shape[1], 
                           error_image.shape[2], error_image.shape[3]))
            
            for col in range(0, int(img_per_fig/num_rows)):
                
                cv2.putText(error_image[index, :, :, :], 
                        dict[error_prediction[index]], 
                        (50,115), font, 1, (200,255,0), 2, cv2.LINE_AA)
                     
                if(col == 0): 
                    row_img = error_image[index, :, :, :]
                
                else:
                    row_img = cv2.hconcat((row_img, 
                                           error_image[index, :, :, :]))
                
                index += 1
                
            img = cv2.vconcat((img, row_img))
            
    
        
        cv2.imshow("Errors - Figure No. " + str(fig_count), img)
    
        
        while True:    
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break

img_per_fig = 40
num_rows = 4
show_error_images(error_image, error_prediction, img_per_fig, num_rows)












