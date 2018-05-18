
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras import metrics
from keras.losses import mean_squared_error
import scipy.misc
from matplotlib.pyplot import imread
import time
from keras.models import *
from keras.layers import * 
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.optimizers import Adam

from sklearn.manifold.t_sne import TSNE
import pickle


# In[2]:


# sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon


# In[3]:


def vae_loss(img_input, x_decoded_mean):
    img_input = K.flatten(img_input)
    x_decoded_mean = K.flatten((x_decoded_mean+1)/2)
    #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(img_input, x_decoded_mean)
    xent_loss = K.mean((img_input - x_decoded_mean)**2)
    lam = 5e-5
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss * lam)


# In[4]:


def xent_loss(img_input, x_decoded_mean):
    img_input = K.flatten(img_input)
    x_decoded_mean = K.flatten((x_decoded_mean+1)/2)
    #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(img_input, x_decoded_mean)
    xent_loss = K.mean((img_input - x_decoded_mean)**2)
    return K.mean(xent_loss)


# In[5]:


def kl_loss(img_input, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)


# In[7]:


train_dir='./hw4_data/train/'
train_data_x = [file for file in os.listdir(train_dir)]
train_data_x.sort()
train_x=[]
for i in train_data_x:
    train_x.append(imread(os.path.join(train_dir,i)))
#train_x = (np.array(train_x)-1.0)*2
train_x = np.array(train_x)

# In[13]:


(img_rows, img_cols) = (64, 64)
latent_dim = 1024
epochs=100
batch_size=64

# encoder architecture
img_input = Input(shape=(img_rows, img_cols, 3))
e = Conv2D(32, kernel_size=(3, 3), padding='same')(img_input)
e = BatchNormalization()(e)
e = Activation('relu')(e)
e = Conv2D(64, kernel_size=(3, 3), padding='same', strides=(2, 2))(e)
e = BatchNormalization()(e)
e = Activation('relu')(e)
e = Conv2D(128, kernel_size=(3, 3), padding='same', strides=(2, 2))(e)
e = BatchNormalization()(e)
e = Activation('relu')(e)
e = Conv2D(256, kernel_size=(3, 3), padding='same', strides=(2, 2))(e)
e = BatchNormalization()(e)
e = Activation('relu')(e)
e = Conv2D(512, kernel_size=(3, 3), padding='same', strides=(2, 2))(e)
e = BatchNormalization()(e)
e = Activation('relu')(e)
e = Conv2D(512, kernel_size=(3, 3), padding='same')(e)
e = BatchNormalization()(e)
e = Activation('relu')(e)
flat = Flatten()(e)

# mean and variance for latent variables
z_mean = Dense(latent_dim)(flat)
z_log_var = Dense(latent_dim)(flat)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder architecture
decoder_hidden = Dense(4*4*512, activation='relu')
decoder_reshape = Reshape((4,4,512))
decoder_deconv_1 = Conv2DTranspose(512, kernel_size=(3, 3), padding='same')
bn_1 = BatchNormalization()
A_1 = Activation('relu')
decoder_deconv_2 = Conv2DTranspose(512, kernel_size=(3, 3), padding='same', strides=(2, 2))
bn_2 = BatchNormalization()
A_2 = Activation('relu')
decoder_deconv_3 = Conv2DTranspose(256, kernel_size=(3, 3), padding='same')
bn_3 = BatchNormalization()
A_3 = Activation('relu')
decoder_deconv_4 = Conv2DTranspose(128, kernel_size=(3, 3), padding='same', strides=(2, 2))
bn_4 = BatchNormalization()
A_4 = Activation('relu')
decoder_deconv_5 = Conv2DTranspose(64, kernel_size=(3, 3), padding='same')
bn_5 = BatchNormalization()
A_5 = Activation('relu')
decoder_deconv_6 = Conv2DTranspose(32, kernel_size=(3, 3), padding='same', strides=(2, 2))
bn_6 = BatchNormalization()
A_6 = Activation('relu')
decoder_mean = Conv2DTranspose(3, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='tanh')

d = decoder_hidden(z)
d = decoder_reshape(d)
d = decoder_deconv_1(d)
d = bn_1(d)
d = A_1(d)
d = decoder_deconv_2(d)
d = bn_2(d)
d = A_2(d)
d = decoder_deconv_3(d)
d = bn_3(d)
d = A_3(d)
d = decoder_deconv_4(d)
d = bn_4(d)
d = A_4(d)
d = decoder_deconv_5(d)
d = bn_5(d)
d = A_5(d)
d = decoder_deconv_6(d)
d = bn_6(d)
d = A_6(d)
x_decoded_mean = decoder_mean(d)

# entire model
vae = Model(img_input, x_decoded_mean)
# Compile
vae.compile(optimizer=Adam(lr=1e-4, beta_1=0.5), metrics=[xent_loss, kl_loss], loss=vae_loss)
vae.summary()

# training
history = vae.fit(train_x, train_x, shuffle=True, epochs=epochs, batch_size=batch_size, validation_split=0.1, 
                  callbacks=[EarlyStopping(patience = 3)])

# encoder from learned model
encoder = Model(img_input, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(latent_dim,))
_d = decoder_hidden(decoder_input)
_d = decoder_reshape(_d)
_d = decoder_deconv_1(_d)
_d = bn_1(_d)
_d = A_1(_d)
_d = decoder_deconv_2(_d)
_d = bn_2(_d)
_d = A_2(_d)
_d = decoder_deconv_3(_d)
_d = bn_3(_d)
_d = A_3(_d)
_d = decoder_deconv_4(_d)
_d = bn_4(_d)
_d = A_4(_d)
_d = decoder_deconv_5(_d)
_d = bn_5(_d)
_d = A_5(_d)
_d = decoder_deconv_6(_d)
_d = bn_6(_d)
_d = A_6(_d)
_x_decoded_mean = decoder_mean(_d)

decoder = Model(decoder_input, _x_decoded_mean)



# In[ ]:


# save all 3 models for future use
vae.save_weights('./models/vae.h5')
encoder.save_weights('./models/encoder.h5')
decoder.save_weights('./models/generator.h5')

# save training history
fname = './models/history.p'
with open(fname, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

