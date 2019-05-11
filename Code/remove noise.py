# -*- coding: utf-8 -*-
"""
Created on Fri May 10 03:41:28 2019

@author: User
"""


import urllib.request
import cv2
import numpy as np
import os

foldered = './Car'





def remove_noise(folder):
    X = []
    filenames = os.listdir(folder)
    count = 0
    pic_num  = 1
    
    for image_filename in filenames:
        img_file = cv2.imread(folder + '/' + image_filename)
        
        if img_file is not None: 
            #print("yes1")
            img_arr = np.asarray(img_file)
            
            if(pic_num == 1):
                X.append(img_arr)
            else:
                for num in range(0, np.array(X).shape[0]):
                    if(np.array_equal(img_arr, X[num])):
                        #print("yes2")
                        if os.path.exists(folder + '/' + image_filename):
                            os.remove(folder + '/' + image_filename)
                            count += 1
                            print(image_filename + "   - Duplicate File Deleted")
                            diff = False
                            break
                        else:
                            print(image_filename + "   - The file does not exist")
                        
                    else:
                        diff = True
                
                if(diff):
                    X.append(img_arr)
    
            pic_num += 1
    
    return X, count



[X_bla, del_count] = remove_noise(foldered)




def remove_blanks(folder):
    count = 0
    filenames = os.listdir(folder)
    
    for image_filename in filenames:
        img_file = cv2.imread(folder + '/' + image_filename)
        
        if img_file is None: 
            os.remove(folder + '/' + image_filename)
            count += 1
            print(image_filename + "   - None File Deleted")
    
    return count


del_count = remove_blanks(foldered)














