# Import libraries
from keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, LeakyReLU, MaxPool2D
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
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
image_file = np.array(glob(os.path.join(file_name,'*')))
print(image_file.shape)
labels = []
#label_file = np.array(glob('../HW5_data/FullLengthVideos/labels/train/*.txt'))
for i,file in enumerate(image_file):
    img_name = np.array(glob(os.path.join(file,'*.png')))
    img = np.array([imread(name) for name in img_name])
    if i == 0:
        imgs = img
    else:
        imgs = np.append(imgs,img,0)
    label = np.zeros(img.shape[0])+i
    labels = np.append(labels,label)
print(imgs.shape,labels.shape)

x_train, x_valid, y_train, y_valid = train_test_split( imgs, labels, test_size=0.1, random_state=1 )
print(x_train.shape,x_valid.shape)
print(y_train.shape,y_valid.shape)
# Function load_minst is available in git.
#(x_semi_train, y_semi_train), (x_test, y_test) = fashion_mnist.load_data()
#x_semi_train = x_semi_train.astype('float32') / 255
#x_semi_train = x_semi_train.reshape(x_semi_train.shape[0], 28, 28, 1)
# Prepare datasets
# This step contains normalization and reshaping of input.
# For output, it is important to change number to one-hot vector. 
#idx = np.random.randint(low = 0,high = 60000,size = 2000)
x_train = x_train.astype('float32') / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#x_train = np.repeat(x_train.astype('float32'), 3, 3)
x_valid = x_valid.astype('float32') / 255
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
#x_valid = np.repeat(x_valid.astype('float32'), 3, 3)
#x_test = x_test.astype('float32') / 255
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#x_test = np.repeat(x_test.astype('float32'), 3, 3)

y_train = np_utils.to_categorical(y_train, 10)
y_valid = np_utils.to_categorical(y_valid, 10)
#y_test = np_utils.to_categorical(y_test, 10)
# Create model in Keras
num_classes = 10

model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=x_train.shape[1:], activation='relu'))
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
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model = load_model('task1.h5')
callbacks = [ ModelCheckpoint('checkpoint/task_my_vgg.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_valid, y_valid),callbacks=callbacks)
loss = history.history['loss']
val_loss = history.history['val_loss']
np.save('./task_my_vgg.npy',np.array([loss,val_loss]))

#score = model.evaluate(x_test, y_test, verbose=1)
#print(score)