
# coding: utf-8

# In[1]:


import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


np.random.seed(100)

shot = int(sys.argv[1]) #10 
train_base_dir = sys.argv[2] #'task2_dataset/base'
train_novel_dir = sys.argv[3] #'task2_dataset/novel'
output_dir = sys.argv[4] #'./'

use_trained_resnet = False

e = {1:5, 5:1, 10:1}
epoch = e[shot]

im_length, im_width = 224, 224


# In[3]:


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.0)


# In[4]:


label_dir = [file for file in os.listdir(train_base_dir) if file.startswith('class')]
label_dir.sort()
train_image = []
for d in label_dir:
    images_dir = [file for file in os.listdir(os.path.join(train_base_dir, d, 'train')) if file.endswith('.png')]
    images_dir.sort()
    for image in images_dir:
        train_image.append(cv2.resize((cv2.imread(os.path.join(train_base_dir,d,'train',image))/255).astype('float32'),(im_length, im_width),interpolation=cv2.INTER_CUBIC))
train_image = np.array(train_image)
    
train_label = []
for i in range(80):
    train_label.extend([i]*500) 
train_label = to_categorical(train_label, num_classes=80)

train_label_dic = np.array([int(label[-2:]) for label in label_dir])


# In[5]:


label_dir = [file for file in os.listdir(train_base_dir) if file.startswith('class')]
label_dir.sort()

valid_image = []
for d in label_dir:
    images_dir = [file for file in os.listdir(os.path.join(train_base_dir, d, 'test')) if file.endswith('.png')]
    images_dir.sort()
    for image in images_dir:
        valid_image.append(cv2.resize((cv2.imread(os.path.join(train_base_dir,d,'test',image))/255).astype('float32'),(im_length, im_width),interpolation=cv2.INTER_CUBIC))
valid_image = np.array(valid_image)
        
valid_label = []
for i in range(80):
    valid_label.extend([i]*100) 
valid_label = to_categorical(valid_label, num_classes=80)    
    
valid_label_dic = train_label_dic


# In[6]:


label_dir = [file for file in os.listdir(train_novel_dir) if file.startswith('class')]
label_dir.sort()

novel_image = []
for d in label_dir:
    images_dir = [file for file in os.listdir(os.path.join(train_novel_dir, d, 'train')) if file.endswith('.png')]
    images_dir.sort()
    for image in images_dir:
        novel_image.append(cv2.resize((cv2.imread(os.path.join(train_novel_dir,d,'train',image))/255).astype('float32'),(im_length, im_width),interpolation=cv2.INTER_CUBIC))
novel_image = np.array(novel_image)
        
novel_label = []
for i in range(20):
    novel_label.extend([i]*500) 
    
one_hot_novel_label = to_categorical(novel_label, num_classes=20) 
    
novel_label_dic = np.array([str(label[-2:]) for label in label_dir])


# In[7]:


select = np.random.choice(500, 20*shot)
select = select+np.repeat(np.array(range(0,20))*500,shot)

shots_novel_image = novel_image[select]
shots_novel_label = np.array(novel_label)[select]
shots_one_hot_novel_label = one_hot_novel_label[select]


# In[ ]:


Resnet = ResNet50(include_top=True, weights=None, classes=80)
#Resnet.summary()


# In[ ]:


if use_trained_resnet == True:
    Resnet.load_weights('data_augmentation_Resnet.h5')
else:
    print('Training Resnet...')

    opt = Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0)
    Resnet.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='Resnet.h5', verbose=1, save_best_only=True)
    Resnet_history = Resnet.fit_generator(datagen.flow(train_image, train_label, batch_size=16, subset='training'),
                                          steps_per_epoch=len(train_image)/4, epochs=100, shuffle=True, 
                                          validation_data = (valid_image, valid_label),
                                          callbacks=[EarlyStopping(patience = 5), checkpointer])


# In[9]:


layer_name = 'flatten_1' # modify the layer name every time you run the program

extractor = Model(inputs=Resnet.input, outputs=Resnet.get_layer(layer_name).output)


# In[10]:


inputs = Input(shape=(im_length, im_width, 3))
x = extractor(inputs)
out = Dense(20, activation='softmax')(x)
mlp2 = Model(inputs = inputs, outputs = out)

opt = Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0)
mlp2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

mlp2.fit_generator(datagen.flow(shots_novel_image, shots_one_hot_novel_label, batch_size=16),
                  steps_per_epoch=len(shots_novel_image), epochs=epoch, shuffle=True)


# In[11]:


mlp2.save_weights(os.path.join(output_dir, '{}_shot_Resnet.h5'.format(shot)))

