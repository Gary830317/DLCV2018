# Import libraries
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras.datasets import fashion_mnist
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.misc import imresize,imread,imsave
from sklearn.cross_validation import train_test_split
import os
from keras.models import Model,load_model
import csv
import keras.backend as K
import argparse

parser = argparse.ArgumentParser(description="task1")
parser.add_argument("-tr","--train_folder",type = str, default = 'data/train/')
parser.add_argument("-te","--test_folder",type = str, default = 'data/test/')
parser.add_argument("-to","--save_path",type = str, default = './')
args = parser.parse_args()


# Hyper Parameters
train_folder = args.train_folder
test_folder = args.test_folder
save_path = args.save_path

# Load data
file_name = train_folder

# Load data
path = test_folder
#for i,file in enumerate(image_file):
    #img_name = np.array(glob(os.path.join(file,'*.png')))
imgs = np.array([imread(os.path.join(path,str(i)+'.png')) for i in range(10000)])

print(imgs.shape)

x_test = imgs.astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#x_test = np.repeat(x_test.astype('float32'), 3, 3)
num_classes = 10
model = Sequential()

model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
#model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="valid", activation='relu'))
#model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="valid", activation='relu'))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
#model.add(ReLU())
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
#model.add(ReLU())
#model2.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))
#model.load_weights('checkpoint/task_my_vgg_channel_3.h5')
model = load_model('checkpoint/task_my_vgg.h5')
y_predict = model.predict(x_test)


with open(os.path.join(save_path,'my_vgg.csv'), 'w', newline='') as csvfile:
    #writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer = csv.writer(csvfile)
    y_predict = np.argmax(y_predict,1)
    print(y_predict.shape)
    writer.writerow(['image_id','predicted_label',])
    for i in range(y_predict.shape[0]):
        #writer.writerow({'name_id': %d, 'predict_label': %d} %(im_id[i],y_predict[i]))
        writer.writerow([str(i), str(y_predict[i]),])
# You can check metrics name in a vector above.