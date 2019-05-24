import cv2
import numpy as np
import random

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers import regression
from tflearn.data_utils import to_categorical

from time import time

IMG_SIZE = 28
N_CLASSES = 26
PROJECT_PATH = '/home/van/Desktop/Project2/' 
TRAIN_SIZE = 7000
chars = [chr(i) for i in range(65, 91)]

def processData():
        import os
        import cv2
        import numpy as np
        import random

        data = []
        label = []

        global chars
        for i, char in enumerate(chars):
                dirPath = PROJECT_PATH + 'image/' +  char + '/'
                fileNames = os.listdir(dirPath)
                for imageFile in fileNames:
                        image = cv2.imread(dirPath + imageFile, 0)
                        data.append(image)
                        label.append(i)

        # xáo trộn dữu liệu
        l = len(label)  # l =7820
        sf = list(range(l))
        random.shuffle(sf)
        
        data = np.array(data)
        label = np.array(label)
        data = data[sf]
        label = label[sf]

        return data, label


data, label = processData()


train_x = data[:TRAIN_SIZE]
train_y = label[:TRAIN_SIZE]
test_x = data[TRAIN_SIZE:]
test_y = label[TRAIN_SIZE:]

# lưu lại tập test để test về sau
np.save(PROJECT_PATH + 'numpy/test_x', test_x)
np.save(PROJECT_PATH + 'numpy/test_y', test_y)

# chuyển dữ liệu train về dạng phù hợp
train_x = np.reshape(train_x, (-1, IMG_SIZE, IMG_SIZE, 1))
train_y = to_categorical(train_y, N_CLASSES)


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
network = dropout(network, 0.8) #5

network = fully_connected(network, N_CLASSES, activation='softmax')
network = regression(network)

model = tflearn.DNN(network) 

# train
t0 = time()
model.fit(train_x, train_y, n_epoch= 50, shuffle= True, show_metric=True, validation_set=0.1)
# model.save(PROJECT_PATH + 'model/model.tflearn')
print('Time: {}'.format(time() - t0))


# load model và tập test
model.load(PROJECT_PATH + 'model/model.tflearn')
test_x = np.load(PROJECT_PATH + 'numpy/test_x.npy')
test_y = np.load(PROJECT_PATH + 'numpy/test_y.npy')


test_x = np.reshape(test_x, (-1, IMG_SIZE, IMG_SIZE, 1))
origin_test_y = test_y  # để kiểm tra độ chính xác
test_y = to_categorical(test_y, N_CLASSES)

test_logist =model.predict(test_x)
pre_y = np.argmax(test_logist, axis=-1)
print('accuracy: %f'%(np.sum(origin_test_y==pre_y) / len(pre_y)))




def testInImage(imagePath):
        import cv2
        import numpy as np

        results= [chr(char) for char in range(65, 91)]

        image = cv2.imread(imagePath)

        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5,5), 0)
        # im, thres = cv2.threshold(im_blur, 160, 255, cv2.THRESH_BINARY_INV)
        thre = cv2.adaptiveThreshold(im_blur,maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV,blockSize=19,C=9)


        contours,hierachy = cv2.findContours(thre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        MIN_SIZE = 14
        for i in contours:
                (x,y,w,h) = cv2.boundingRect(i)
                if (h < MIN_SIZE or w < MIN_SIZE):
                        continue
                cv2.rectangle(image,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
                roi = thre[y:y+h,x:x+w]
                
                roi = np.pad(roi,(20,20),'constant',constant_values=(0,0))
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))

                test_x = np.array([roi,])
                test_x = test_x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                # dự đoán với tập dữ liệu test
                global model
                test_logits = model.predict(test_x)
                # #lấy phần tử có giá trị lớn nhất 
                test_logits = np.argmax(test_logits, axis=-1)
                res = results[int(test_logits)]
                cv2.putText(image, str(res), (x, y),cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 0), 3)
        cv2.namedWindow('im', cv2.WINDOW_NORMAL)
        cv2.imwrite(PROJECT_PATH + 'test_image/result.jpg', image)
        cv2.imshow('im', image)
        cv2.waitKey(20000)
        cv2.destroyAllWindows()

testInImage(PROJECT_PATH + 'test_image/test.jpg')

