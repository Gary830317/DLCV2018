
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras import backend as K
from keras import metrics
import scipy.misc
from matplotlib.pyplot import imread
import time
from keras.models import *
from keras.layers import * 
from keras.callbacks import EarlyStopping, TensorBoard, Callback
from keras.optimizers import Adam
import matplotlib.gridspec as gridspec
from sklearn.manifold.t_sne import TSNE
import pickle
from matplotlib.ticker import MaxNLocator
import csv
import sys


# In[2]:


def plot(imgs, row, col):
    fig = plt.figure(figsize=(col*2, row*2))
    gs = gridspec.GridSpec(row, col)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(64, 64, 3))
    return img


# In[3]:


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var) * epsilon


# In[4]:


def vae_loss(img_input, x_decoded_mean):
    img_input = K.flatten(img_input)
    x_decoded_mean = K.flatten((x_decoded_mean+1)/2)
    xent_loss = K.mean((img_input - x_decoded_mean)**2)
    lam = 5e-4
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss * lam)


# In[5]:


def xent_loss(img_input, x_decoded_mean):
    img_input = K.flatten(img_input)
    x_decoded_mean = K.flatten((x_decoded_mean+1)/2)
    #xent_loss = img_rows * img_cols * metrics.binary_crossentropy(img_input, x_decoded_mean)
    xent_loss = K.mean((img_input - x_decoded_mean)**2)
    return K.mean(xent_loss)


# In[6]:


def kl_loss(img_input, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(kl_loss)


# In[7]:


#data_dir = './hw4_原始/hw4_data'
data_dir = sys.argv[1]
output_dir = sys.argv[2]


# In[8]:


train_dir = os.path.join(data_dir, 'train')
train_data_x = [file for file in os.listdir(train_dir) if file.endswith('.png')]
train_data_x.sort()
train_x=[]
for i in train_data_x:
    train_x.append(imread(os.path.join(train_dir,i)))
train_x = np.array(train_x)

num_train = train_x.shape[0]


# In[9]:


test_dir = os.path.join(data_dir, 'test')
test_data_x = [file for file in os.listdir(test_dir) if file.endswith('.png')]
test_data_x.sort()
test_x=[]
for i in test_data_x:
    test_x.append(imread(os.path.join(test_dir,i)))
test_x = np.array(test_x)

num_test = test_x.shape[0]


# In[10]:


y_data=[]
with open(os.path.join(data_dir, 'train.csv'), newline='') as file:
    reader = csv.reader(file, delimiter=',', quotechar='|')
    for row in reader:
        y_data.append(row[8])
train_y = np.array(y_data[1:], dtype=float)


# In[11]:


y_data=[]
with open(os.path.join(data_dir, 'test.csv'), newline='') as file:
    reader = csv.reader(file, delimiter=',', quotechar='|')
    for row in reader:
        y_data.append(row[8])
test_y = np.array(y_data[1:], dtype=float)


# In[24]:


(img_rows, img_cols) = (64, 64)
latent_dim = 1024

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
#vae.summary()

# encoder from learned model
encoder = Model(img_input, z_mean)
#encoder.summary()

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
#decoder.summary()


# In[13]:


#with open ('./hw4_原始/vae_save/models_vae_tanh_mse_5e-5/history.p', 'rb') as file:
with open ('vae_history.p', 'rb') as file:
    history = pickle.load(file)
mse = history['xent_loss']
KLD = history['kl_loss']
fig, ax = plt.subplots(1,2, figsize=(20,8))
ax[0].plot(range(1, len(KLD)+1), KLD)
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel('Training epochs', fontsize=16)
ax[0].set_title('KLD', fontsize=20)
ax[1].plot(range(1, len(mse)+1), mse)
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel('Training epochs', fontsize=16)
ax[1].set_title('MSE', fontsize=20)
plt.savefig(os.path.join(output_dir,'fig1_2.jpg'), bbox_inches='tight')
plt.close(fig)


# In[15]:


np.random.seed(66)
#vae.load_weights('./hw4_原始/vae_save/models_vae_tanh_mse_5e-6/vae.h5')
vae.load_weights('vae.h5')
sam = np.random.random_integers(num_train, size=(10))
predict = vae.predict(train_x[sam])
predict = (np.array(predict)+1.0)/2
images = np.concatenate((train_x[sam], predict))
fig = plot(images, 2, 10)
plt.savefig(os.path.join(output_dir,'fig1_3.jpg'), bbox_inches='tight')
plt.close()


# In[16]:


np.random.seed(66)
#decoder.load_weights('./hw4_原始/vae_save/models_vae_tanh_mse_5e-5/generator.h5')
decoder.load_weights('vae_generator.h5')
noise = np.random.normal(0.66, 0.77, (32, latent_dim))
predict = decoder.predict(noise)
predict = (np.array(predict)+1.0)/2
fig = plot(predict, 4, 8)
plt.savefig(os.path.join(output_dir,'fig1_4.jpg'), bbox_inches='tight')
plt.close()


# In[17]:


#encoder.load_weights('./hw4_原始/vae_save/models_vae_tanh_mse_5e-5/encoder.h5')
encoder.load_weights('vae_encoder.h5')
predict = encoder.predict(test_x)
predict_2d = TSNE(n_components=2, perplexity=50, learning_rate=500, n_iter = 500).fit_transform(predict) 

bool_attr = np.array(test_y) == 1.0
fig = plt.figure(figsize=(8, 8))
attr0 = plt.scatter(x=predict_2d[bool_attr, 0],
                    y=predict_2d[bool_attr, 1],
                    color='b',
                    alpha=0.5)
attr1 = plt.scatter(x=predict_2d[~bool_attr, 0],
                    y=predict_2d[~bool_attr, 1],
                    color='r',
                    alpha=0.5)
plt.legend((attr0, attr1), ('Male', 'Female'), fontsize=16, ncol=1, loc=2)
plt.title('Result', fontsize=20)
plt.savefig(os.path.join(output_dir, 'fig1_5.jpg'), bbox_inches='tight')
plt.close(fig)


# In[32]:


def gan_build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    G = Sequential()

    G.add(Reshape((2, 2, 256), input_shape=(latent_size,)))
    
    # upsample to (8, 8, ...)
    G.add(Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())
    
    # upsample to (8, 8, ...)
    G.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())
    
    # upsample to (16, 16, ...)
    G.add(Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())

    # upsample to (32, 32, ...)
    G.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())

    G.add(Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            activation='tanh', kernel_initializer='glorot_normal'))


    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,))

    fake_image = G(latent)
    #G.summary()
    return Model(latent, fake_image)


# In[19]:


#with open ('./hw4_原始/gan_save/gan-history.p', 'rb') as file:
with open ('gan_history.p', 'rb') as file:
    history = pickle.load(file)
generator_loss_train = history['train']['generator']
discriminator_loss_train = history['train']['discriminator']
discriminator_loss_real = history['test']['discriminator_loss_real']
discriminator_loss_fake = history['test']['discriminator_loss_fake']

fig, ax = plt.subplots(1,2, figsize=(20,8))
ax[0].plot(range(1, len(generator_loss_train)+1), generator_loss_train, label='Generator')
ax[0].plot(range(1, len(discriminator_loss_train)+1), discriminator_loss_train, label='Discriminator')
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel('Training epochs', fontsize=16)
ax[0].set_title('Training Loss', fontsize=20)
ax[0].legend(loc="upper left", fontsize=20)
ax[1].plot(range(1, len(discriminator_loss_real)+1), discriminator_loss_real, label='Real')
ax[1].plot(range(1, len(discriminator_loss_fake)+1), discriminator_loss_fake, label='Fake')
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel('Training epochs', fontsize=16)
ax[1].set_title('Real image loss vs Fake image loss', fontsize=20)
ax[1].legend(loc="upper right", fontsize=20)

plt.savefig(os.path.join(output_dir, 'fig2_2.jpg'), bbox_inches='tight')
plt.close(fig)


# In[20]:


latent_size = 1024
generator = gan_build_generator(latent_size)

np.random.seed(777)
#generator.load_weights('./hw4_原始/gan_save/params_generator_epoch_000027.h5')
generator.load_weights('gan_generator.h5')
noise = np.random.normal(0.65, 0.3, (32, latent_size))
generated_images = generator.predict(noise, verbose=0)
generated_images = (generated_images+1.0)/2.0
np.clip(generated_images, 0, 1, out=generated_images)
fig = plot(generated_images, 4, 8)
plt.savefig(os.path.join(output_dir, 'fig2_3.jpg'), bbox_inches='tight')
plt.close()


# In[41]:


def acgan_build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    G = Sequential()
    
    G.add(Reshape((2, 2, -1), input_shape=(latent_size+256,)))

    # upsample to (8, 8, ...)
    G.add(Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())
    
    # upsample to (16, 16, ...)
    G.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())
    
    # upsample to (32, 32, ...)
    G.add(Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())

    # upsample to (64, 64, ...)
    G.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    G.add(Activation('relu'))
    G.add(BatchNormalization())

    G.add(Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            activation='tanh', kernel_initializer='glorot_normal'))


    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,))

    # this will be our label
    label_input = Input(shape=(1,))
    emb_label = Embedding(2, 256, embeddings_initializer='glorot_normal')(label_input) 
    emb_label = Reshape((256,))(emb_label)

    # hadamard product between z-space and a class conditional embedding
    h = concatenate([latent, emb_label], axis=1)

    fake_image = G(h)
    #G.summary()
    return Model([latent, label_input], fake_image)


# In[22]:


#with open ('./hw4_原始/acgan_save/acgan-history.p', 'rb') as file:
with open ('acgan_history.p', 'rb') as file:
    history = pickle.load(file)
generator_loss_train = history['train']['generator']
discriminator_loss_train = history['train']['discriminator']
discriminator_loss_real = history['test']['discriminator_loss_real']
discriminator_loss_fake = history['test']['discriminator_loss_fake']

fig, ax = plt.subplots(1,2, figsize=(20,8))
ax[0].plot(range(1, len(generator_loss_train)+1), generator_loss_train, label='Generator')
ax[0].plot(range(1, len(discriminator_loss_train)+1), discriminator_loss_train, label='Discriminator')
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].set_xlabel('Training epochs', fontsize=16)
ax[0].set_title('Training Loss', fontsize=20)
ax[0].legend(loc="upper left", fontsize=20)
ax[1].plot(range(1, len(discriminator_loss_real)+1), discriminator_loss_real, label='Real')
ax[1].plot(range(1, len(discriminator_loss_fake)+1), discriminator_loss_fake, label='Fake')
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].set_xlabel('Training epochs', fontsize=16)
ax[1].set_title('Real image loss vs Fake image loss', fontsize=20)
ax[1].legend(loc="upper right", fontsize=20)

plt.savefig(os.path.join(output_dir, 'fig3_2.jpg'), bbox_inches='tight')
plt.close(fig)


# In[23]:


latent_size = 512 
generator = acgan_build_generator(latent_size)

np.random.seed(778) #777
#generator.load_weights('./hw4_原始/acgan_save/params_generator_epoch_000043.h5')
generator.load_weights('acgan_generator.h5')
noise = np.random.normal(0.1, 0.5, (10, latent_size))

labels_0 = np.array([0] * 10).reshape(-1,)
labels_1 = np.array([1] * 10).reshape(-1,)

# get a batch to display
generated_images_0 = generator.predict([noise, labels_0], verbose=0)
generated_images_0 = (generated_images_0+1.0)/2.0
np.clip(generated_images_0, 0, 1, out=generated_images_0)

generated_images_1 = generator.predict([noise, labels_1], verbose=0)
generated_images_1 = (generated_images_1+1.0)/2.0
np.clip(generated_images_1, 0, 1, out=generated_images_1)

generated_images = np.concatenate((generated_images_0, generated_images_1), axis=0)

fig = plot(generated_images, 2, 10)
plt.savefig(os.path.join(output_dir, 'fig3_3.jpg'), bbox_inches='tight')
plt.close()

