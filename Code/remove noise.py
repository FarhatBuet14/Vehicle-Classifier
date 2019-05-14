import cv2
import numpy as np
import os
import pandas as pd 
import shutil



folder = './Bicycle'



######################################################################

def resize_images(folder, save_folder, img_size = 128):
    count = 0
    filenames = os.listdir(folder)
    
    for image_filename in filenames:
        img_file = cv2.imread(folder + '/' + image_filename)
        
        if img_file is not None: 
            img_file = cv2.resize(img_file, (img_size, img_size))
            count += 1
            cv2.imwrite(save_folder + image_filename, img_file)
    
    return count

save_folder = './Resized/Car/'
img_size = 256
count = resize_images(folder, save_folder, img_size)



######################################################################



def remove_noise(folder):
    X = []
    filenames = os.listdir(folder)
    count = 0
    pic_num  = 1
    
    for image_filename in filenames:
        img_file = cv2.imread(folder + '/' + image_filename)
        
        if img_file is not None: 
            #print(pic_num)
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



[X_bla, del_count] = remove_noise(folder)




######################################################################


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


del_count = remove_blanks(folder)




######################################################################

 
deleted_folder = "./Deleted"
csv_file_name = "Bicycle.csv"

def delete_the_rejected_images(csv_file_name, deleted_folder):
    
    file = pd.read_csv(csv_file_name) 
    file = np.array(file)
    image_count = 0
    for file_name in file[:, 0]:
        if(file[image_count, 1] == 2):
            shutil.move(folder + '/' + file_name, deleted_folder + '/' + 'deleted_' + file_name) 
            print(file_name + ' rejected, so removed - ' + str(file[image_count, 1]))
        
        image_count += 1

    return image_count

image_count = delete_the_rejected_images(csv_file_name, deleted_folder)


