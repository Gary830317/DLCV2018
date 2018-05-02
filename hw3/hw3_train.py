
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
from model import *

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

