import numpy as np 
import csv
import matplotlib
import matplotlib.pyplot as plt 
import cv2
import random

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

from time import time
t0 = time()

BATCH_SIZE = 32
IMG_SIZE = 28
N_CLASSES = 26
LR = 0.001
N_EPOCHS = 50
import csv
import numpy as np 
from time import time
t0 = time()
with open('demo/hand_written.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    
    train_data = []
    train_label = []

    for letter in result:
        # x = np.array([int(j) for j in letter[1:]])
        x = np.array(letter[1:])
        x = x.reshape(28, 28)
        train_data.append(x)
        train_label.append(int(letter[0]))
print(time() - t0)
print(train_label[100])       
print(len(train_label))
print(np.shape(train_label))

#xáo trộn dữ liệu 
l = len(train_label)
shuffer_order = list(range(l))
random.shuffle(shuffer_order)

train_data = np.array(train_data)
train_label = np.array(train_label)

train_data = train_data[shuffer_order]
train_label = train_label[shuffer_order]

# chia tập dữ liệu ra training set, test set và validation set
train_x = train_data[:300000]
train_y = train_label[:300000]

val_x = train_data[300000:330000]  
val_y = train_label[300000:330000]

test_x = train_data[330000:]
test_y = train_label[330000:] 

# img = cv2.imread('image/char2.jpg', 0)
# print(img.shape)
# # crop = img[20:120,120:450,:]
# img = cv2.resize(img, (28,28))

# img = img.reshape((28*28))
# img = img / 255.0
# for i in range(len(img)):
#     if img[i] < 0.55:
#         img[i] = 0
#     else:
#         img[i] = 1
# img = img.reshape((28,28))
# cv2.imshow('image', img)

# # cv2.waitKey(1000)
# # img = test_x[0]
# # print(img.shape)
# # img = cv2.resize(img, (28,28))
# # cv2.imshow('image', img)
# cv2.waitKey(2000)
# cv2.destroyAllWindows();
# test_x = np.array([img,])

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

# print(np.shape(train_y))
# đưa dữ liệu về định dạng phù hợp
train_x = train_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
val_x = val_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

original_test_y = test_y # được sử dụng để test ở bước sau
# chuyển label về dạng onhot vector 
train_y = to_categorical(train_y, N_CLASSES)
val_y = to_categorical(val_y, N_CLASSES)
test_y = to_categorical(test_y, N_CLASSES)

# training
model.fit(train_x, train_y, n_epoch=N_EPOCHS, validation_set=(val_x, val_y), show_metric=True)
model.save('model/mymodel.tflearn')

# thử nghiệm mô hình 

model.load('model/mymodel.tflearn')
# dự đoán với tập dữ liệu test
test_logits = model.predict(test_x)
# #lấy phần tử có giá trị lớn nhất 
test_logits = np.argmax(test_logits, axis=-1)
print(test_logits)
print(np.sum(test_logits == original_test_y) / len(test_logits))
#result: 0.9964297306069458
print(time()-t0)
from time import time
import numpy as np
from matplotlib import pyplot as plt
import csv
import matplotlib
t = time()
with open('demo/hand_written.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []
    rows = np.array(rows)
    for row in result:
        np.append(rows, row)
# print(rows[100000])
print(time() -t)from time import time
import numpy as np
from matplotlib import pyplot as plt
import csv
import matplotlib
t = time()
with open('demo/hand_written.csv', 'r') as csv_file:
    result = csv.reader(csv_file)
    rows = []
    rows = np.array(rows)
    for row in result:
        np.append(rows, row)
# print(rows[100000])
print(time() -t)