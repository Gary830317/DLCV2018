
# coding: utf-8

# In[1]:


import re
import glob
import scipy.misc
import glob
import numpy as np
import time
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.layers import * 
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


# In[2]:


start = time.time()


# In[33]:


def data_processing():
    
    tra_data_x = glob.glob('./data/train/*sat.jpg')
    tra_data_y = glob.glob('./data/train/*mask.png')
    
    tra_data_x.sort()
    tra_data_y.sort()

    n_img=len(tra_data_x)

    tra_x=[]
    for i in tra_data_x:
        tra_x.append(scipy.misc.imread(i))
    tra_x = np.array(tra_x)/255
    
    tra_y = np.empty((n_img, 512, 512, 7))
    ll=np.eye(7)
    
    for i, file in enumerate(tra_data_y):
        mask = scipy.misc.imread(file)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        tra_y[i, mask == 3] = ll[0]  # (Cyan: 011) Urban land 
        tra_y[i, mask == 6] = ll[1]  # (Yellow: 110) Agriculture land 
        tra_y[i, mask == 5] = ll[2]  # (Purple: 101) Rangeland 
        tra_y[i, mask == 2] = ll[3]  # (Green: 010) Forest land 
        tra_y[i, mask == 1] = ll[4]  # (Blue: 001) Water 
        tra_y[i, mask == 7] = ll[5]  # (White: 111) Barren land 
        tra_y[i, mask == 0] = ll[6]  # (Black: 000) Unknown 
        tra_y[i, mask == 4] = ll[6]  # (???: 100) Unknown 
    
    return(tra_x, tra_y)


# In[6]:


def FCN_Vgg16_32s(input_shape=None, classes=None):
    
    img_input = Input(shape=input_shape)

    x = BatchNormalization()(img_input)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(7, (1, 1), kernel_initializer='he_normal')(x)
    # Deconv Layer
    x = Conv2DTranspose(classes, kernel_size=(64, 64), strides=(32, 32), activation='linear',use_bias=False, padding='same')(x)
    # softmax
    x = Activation('softmax')(x)
    
    model = Model(img_input, x)
    
    model.summary()
    
    return model


# In[ ]:


def VGGSegnet(input_shape, classes):

    img_input = Input(shape=input_shape)

    x = BatchNormalization()(img_input)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)


    # Up Block 3
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 4
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Up Block 5
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(classes, (1, 1), activation='linear', padding='same')(x)
    

    x = (Activation('softmax'))(x)
    model = Model( img_input , x )
    
    model.summary()

    return model


# In[ ]:


def VGGUnet (input_shape=None, classes=None): 

    img_input = Input(shape=input_shape)

    inputs = BatchNormalization()(img_input)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4)
  

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(7, (1, 1), activation='relu', padding='same')(conv9)
    
    conv11 = Activation('softmax')(conv10)
    
    model = Model(img_input, conv11)

    model.summary()

    return model


# In[106]:


print('loading data...')


# In[34]:


tra_x, tra_y = data_processing()


# In[111]:


f1 = time.time()
print('data loading time: {:.2f} min'.format((f1-start)/60))


# In[112]:


print('training model...')


# In[29]:


#model=[FCN_Vgg16_32s, VGGSegnet, VGGUnet]

m='VGGUnet'

print('model: {}'.format(m))

if m == 'FCN_Vgg16_32s':
    model = FCN_Vgg16_32s(input_shape=(512, 512, 3), classes=7)
    model.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    checkpointer = ModelCheckpoint(filepath='./FCN_Vgg16_32s.h5', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit(tra_x, tra_y, epochs=20, batch_size=16, validation_split=0.1, verbose = 1, callbacks=[checkpointer])


elif m=='VGGSegnet':
    model = VGGSegnet(input_shape=(512, 512, 3), classes=7)
    model.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    checkpointer = ModelCheckpoint(filepath='./VGGSegnet.h5', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit(tra_x, tra_y, epochs=40, batch_size=8, validation_split=0.1, verbose = 1, callbacks=[checkpointer])


    
elif m=='VGGUnet':
    model = VGGUnet(input_shape=(512, 512, 3), classes=7)
    model.load_weights('./vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
    checkpointer = ModelCheckpoint(filepath='tmp/VGGUnet.h5', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    model.fit(tra_x, tra_y, epochs=40, batch_size=8, validation_split=0.1, verbose = 1, callbacks=[checkpointer])


# In[24]:


f2=time.time()
print('training time: {:.2f} min'.format((f2-f1)/60))


# In[25]:


print('total time: {:.2f} min'.format((f2-start)/60))

