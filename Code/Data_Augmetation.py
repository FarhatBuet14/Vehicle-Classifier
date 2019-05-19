# -*- coding: utf-8 -*-
"""
Created on Sun May 19 02:09:38 2019

@author: User
"""

##################### Library Imports ################################

import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot



#################### Environment & Variables ############################
from keras import backend as K
K.set_image_data_format('channels_last')

datafile = './Data/datafile.npz'

aug_data_dir = './Augmented_Data/channel_shift/'


batch_size = 16

####################### Loading the Data ################################


dataset = np.load(datafile)

X_train = dataset['X_train']
y_train = dataset['y_train']


####################### Data Augmentation ################################


from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(channel_shift_range=0.2)

datagen.fit(X_train)

# configure batch size and retrieve one batch of images
X_batch, y_batch = datagen.flow(X_train, y_train, batch_size = batch_size, 
                                save_to_dir = aug_data_dir, save_prefix = 'Aug')






























