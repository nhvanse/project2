import cv2
import numpy as np
import random

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical


IMG_SIZE = 28
N_CLASSES = 26
PROJECT_PATH = '/home/van/Desktop/Project2/' 


tf.reset_default_graph()

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1]) 

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)

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

network = fully_connected(network, 1024, activation='relu') 
network = dropout(network, 0.8)

network = fully_connected(network, N_CLASSES, activation='softmax')
network = regression(network)

model = tflearn.DNN(network)

model.load(PROJECT_PATH + 'modelNew/NewModel.tflearn')


import cv2
image = cv2.imread(PROJECT_PATH + 'camera/A.jpg')
imGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imBlur = cv2.GaussianBlur(imGray, (5,5), 0)
thre = cv2.adaptiveThreshold(imBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 6)
cv2.imshow('thres', thre)
cv2.waitKey(3000)
cv2.destroyAllWindows()
contours, hierachy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

results= [chr(char) for char in range(65, 91)]

for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
        roi = thre[y:y+h,x:x+w]
        roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        test_x = np.array([roi,])
        test_x = np.reshape(test_x, (-1, IMG_SIZE, IMG_SIZE, 1))
        # dự đoán với tập dữ liệu test
        test_logits = model.predict(test_x)
        # #lấy phần tử có giá trị lớn nhất 
        test_logits = np.argmax(test_logits, axis=-1)
        
        res = results[int(test_logits)]
        cv2.putText(image, str(res), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255 , 255), 3)
cv2.imwrite('result.jpg', image)
