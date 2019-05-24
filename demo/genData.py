import cv2
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt 
import random

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

from time import time
t0 = time()

PROJECT_PATH = '/home/van/Desktop/Project2/'
IMG_SIZE = 28
N_CLASSES = 26


tf.reset_default_graph()

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) #1

network = conv_2d(network, 32, 3, activation='relu') #2
network = max_pool_2d(network, 2) #3

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 1024, activation='relu') #4
network = dropout(network, 0.8) #5

network = fully_connected(network, N_CLASSES, activation='softmax')#6
network = regression(network)

model = tflearn.DNN(network) #7



model.load(PROJECT_PATH + 'model/mymodel.tflearn')


import cv2
import numpy as np

IMG_SIZE = 28
N_CLASSES = 26
MIN_SIZE = 14
results= [chr(char) for char in range(65, 91)]
label = results
for i in label:
    imagePath = PROJECT_PATH + 'camera/' + i + '.jpg'
    image = cv2.imread(imagePath)

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_blur = cv2.GaussianBlur(im_gray, (5,5), 0)
    # im, thres = cv2.threshold(im_blur, 160, 255, cv2.THRESH_BINARY_INV)
    thre = cv2.adaptiveThreshold(im_blur,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV,blockSize=29,C=9)
    
    contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    
    dem = 1
    
    for j in contours:
        (x,y,w,h) = cv2.boundingRect(j)
        if (w < MIN_SIZE or h < MIN_SIZE):
            continue
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        roi = thre[y:y+h,x:x+w]
        
        roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        test_x = np.array([roi,])
        test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        # dự đoán với tập dữ liệu test
        test_logits = model.predict(test_x)
        # #lấy phần tử có giá trị lớn nhất 
        test_logits = np.argmax(test_logits, axis=-1)

        res = results[int(test_logits)]
        
        if (str(res) == i):
            file1 = PROJECT_PATH + 'image/'+ i + '/' + i +str(dem)+'.jpg'
            cv2.imwrite(file1, roi)
            dem += 1
