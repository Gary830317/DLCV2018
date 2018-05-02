
# In[5]:


import re
import glob
import sys
import scipy.misc
import glob
import numpy as np
import time
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.layers import * 


# In[19]:


def to_image(predict, dir_, output_dir):
    label2rgb = np.array([[0.,255.,255.], [255.,255.,0.], [255.,0.,255.], [0.,255.,0.], [0.,0.,255.], [255.,255.,255.], [0.,0.,0.]])
    for idx, i in enumerate(predict):
        mask=label2rgb[np.argmax(i, axis=2)]
        k=os.path.split(test_data_x[idx])[1]
        scipy.misc.imsave(os.path.join(output_dir, '{}_mask.png'.format(k[:4])).replace('\r', ''), mask)

# In[3]:


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
    #x = UpSampling2D(size = (32,32))(x)
    x = Conv2DTranspose(classes, kernel_size=(64, 64), strides=(32, 32), activation='linear',use_bias=False, padding='same')(x)
    # softmax
    #x = Reshape((-1,classes))(x)
    x = Activation('softmax')(x)
    
    model = Model(img_input, x)
    
    model.summary()
    
    return model


# In[ ]:


def segnet(input_shape, classes):
    
    img_input = Input(shape=input_shape)
    
    x = BatchNormalization()(img_input)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Up Block 2
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
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
    
    # softmax

    x = Activation('softmax')(x)
    
    model = Model(img_input, x)

    model.summary()    
    return model

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
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

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

# In[ ]:


start=time.time()


# In[ ]:


#model=[FCN_Vgg16_32s, segnet]

if sys.argv[1] == 'hw3.sh':
    model = FCN_Vgg16_32s(input_shape=(512, 512, 3), classes=7)
    model.load_weights('FCN_Vgg16_32s.h5')   

elif sys.argv[1] == 'hw3_best.sh':
    model = VGGUnet(input_shape=(512, 512, 3), classes=7)
    model.load_weights('VggUnet.h5')



# In[16]:


test_dir=sys.argv[2]


test_data_x = [file for file in os.listdir(test_dir) if file.endswith('.jpg')]

test_x=[]
for i in test_data_x:
    test_x.append(scipy.misc.imread(os.path.join(test_dir,i)))
test_x = np.array(test_x)/255


# In[20]:


output_dir=sys.argv[3]

predict = model.predict(test_x, batch_size=2)
to_image(predict, test_data_x, output_dir)


# In[ ]:


print('finished in {:.2f} min'.format((time.time()-start)/60))

