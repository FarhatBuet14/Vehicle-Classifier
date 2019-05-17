import cv2
import os
import numpy as np
from keras.utils import to_categorical

######################### Parameters #######################################
from keras import backend as K
K.set_image_data_format('channels_last')

imageSize = 128

fldr = './Own_Dataset/Train/'

datafile = './Data/datafile.npz'


######################## Helper Functions #################################

# Function to shuffle two arrays in Unison

def shuffleData(X,y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X_shuffled = X[randomize]
    y_shuffled = y[randomize]
    return X_shuffled,y_shuffled

# Function to get all images from a directory

def getData(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    folders = os.listdir(folder)
    image_names = []
    for folderName in folders:
        if not folderName.startswith('.'):
            if folderName in ['Car']:
                label = 0
            elif folderName in ['Bicycle']:
                label = 1
            elif folderName in ['Bus']:
                label = 2       
            filenames = os.listdir(folder + folderName)
            for image_filename in filenames:
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    dim = (imageSize,imageSize)
                    img_file = cv2.resize(img_file,dim,cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return image_names,y

##################### Main Code ############################################

X, y = getData(fldr)
X, y = shuffleData(X,y)


random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=random_seed)


X_train = X_train/255
X_test = X_test/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


random_seed = 0
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)



################# Storing the Dataset to an npz file #######################

np.savez(datafile,
         X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test,
         X_val=X_val, y_val=y_val)












