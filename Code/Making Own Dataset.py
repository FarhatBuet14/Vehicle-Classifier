# -*- coding: utf-8 -*-
"""
Created on Wed May  8 21:15:47 2019

@author: User
"""

import urllib.request
import cv2
import numpy as np
import os




plane_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04552348'
rickshaw_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03599486'
car_links = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02958343'
train_links = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04468005'
bicycle_links = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02835271'
ship_links = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289'
bus_links = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02924116'
bicycle2_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03792782'
bicycle6_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04126066'
bicycle3_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03853924'
bicycle5_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04026813'
bicycle7_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04524716'
ship2_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04194289'
ship3_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03095699'
ship4_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02965300'
ship5_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03896103'
ship6_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04128837'
car2_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04037443'
car3_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02814533'
car4_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02930766'
car5_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03079136'
bus2_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04146614'











bus3_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04487081'

bus4_links = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03769881'

bus5_links = ''

bus6_links = ''





def store_raw_images(link):
    urls = urllib.request.urlopen(link).read().decode()
    pic_num = 1129
    
#    if not os.path.exists('Own_Dataset'):
#        os.makedirs('Own_Dataset')
        
    for i in urls.split('\n'):
        try:
            if(pic_num > 0):          
                print(i)
                urllib.request.urlretrieve(i, str(pic_num)+".jpg")
                img = cv2.imread(str(pic_num)+".jpg")
                
                if img is not None:
                    cv2.imwrite(str(pic_num)+".jpg",img)
                    pic_num += 1
            
        except Exception as e:
            print(str(e)) 


store_raw_images(bus2_links)
























