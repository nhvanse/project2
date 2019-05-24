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
# img = cv2.imread('image/a1.jpg', 0)
# print(img.shape)
# # crop = img[20:120,120:450,:]
# img = cv2.resize(img, (28,28))

# img = img.reshape((28*28))
# img = img / 255.0
# for i in range(len(img)):
#     if img[i] < 0.6:
#         img[i] = 0
#     else:
#         img[i] = 1
# img = img.reshape((28,28))
# cv2.imshow('image', img)

# cv2.waitKey(3000)
# cv2.destroyAllWindows()

# test_x = np.array([img,])







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


model.load('/home/van/Desktop/Project2/model/mymodel.tflearn')






import cv2
import numpy as np
image = cv2.imread("image/char.jpg")
cv2.imshow("image", image)
cv2.waitKey(2000)
cv2.destroyAllWindows()
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
im_blur = cv2.GaussianBlur(im_gray, (5,5), 0)
im, thre = cv2.threshold(im_blur, 90, 255, cv2.THRESH_BINARY_INV)
contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


rects = [cv2.boundingRect(cnt) for cnt in contours]

for i in contours:
    (x,y,w,h) = cv2.boundingRect(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    roi = thre[y:y+h,x:x+w]
    roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    """ interpolation=cv2.INTER_AREA"""
    roi = cv2.dilate(roi, (3, 3))

    # print(np.shape(roi))
    # cv2.imshow("image", roi)
    # cv2.waitKey(2000)
    # cv2.destroyAllWindows()

    test_x = np.array([roi,])
    test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # dự đoán với tập dữ liệu test
    test_logits = model.predict(test_x)
    # #lấy phần tử có giá trị lớn nhất 
    test_logits = np.argmax(test_logits, axis=-1)
    results= [chr(char) for char in range(65, 91)]
    res = results[int(test_logits)]
    cv2.putText(image, str(res), (x, y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv2.imshow("image",image)
cv2.imwrite("image_result.jpg",image)
cv2.waitKey()
cv2.destroyAllWindows()


# test_logits = model.predict(test_x)
# test_logits = np.argmax(test_logits, axis=-1)
# print(test_logits)