
# coding: utf-8

# In[1]:


from collections import defaultdict
import csv
import pickle
import time
from matplotlib.pyplot import imread
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[2]:



def plot(samples, n_row, n_col):
    fig = plt.figure(figsize=(n_col*2, n_row*2))
    gs = gridspec.GridSpec(n_row, n_col)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(64, 64, 3))
    return fig


# In[3]:


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    D = Sequential()

    D.add(Conv2D(32, kernel_size=(5, 5), padding='same', input_shape=(64, 64, 3)))
    #D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    D.add(Conv2D(64, kernel_size=(5, 5), padding='same', strides=2))
    #D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    D.add(Conv2D(128, kernel_size=(5, 5), padding='same', strides=2))
    #D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    D.add(Conv2D(256, kernel_size=(5, 5), padding='same', strides=2))
    #D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))

    D.add(Conv2D(512, kernel_size=(5, 5), padding='same', strides=2))
    #D.add(BatchNormalization())
    D.add(LeakyReLU(0.2))    
    D.add(Flatten())    
    
    image = Input(shape=(64, 64, 3))
    features = D(image)
    
    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)


    return Model(image, fake)


# In[4]:


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    G = Sequential()

    G.add(Reshape((2, 2, 256), input_shape=(latent_size,)))
    
    # upsample to (8, 8, ...)
    G.add(Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    #G.add(BatchNormalization())
    G.add(Activation('relu'))
    G.add(BatchNormalization())
    
    # upsample to (8, 8, ...)
    G.add(Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    #G.add(BatchNormalization())
    G.add(Activation('relu'))
    G.add(BatchNormalization())
    
    # upsample to (16, 16, ...)
    G.add(Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    #G.add(BatchNormalization())
    G.add(Activation('relu'))
    G.add(BatchNormalization())

    # upsample to (32, 32, ...)
    G.add(Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='glorot_normal'))
    #G.add(BatchNormalization())   
    G.add(Activation('relu'))
    G.add(BatchNormalization())

    G.add(Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            activation='tanh', kernel_initializer='glorot_normal'))


    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size,))

    fake_image = G(latent)

    return Model(latent, fake_image)


# In[5]:


start_time = time.time()


# In[6]:


print('loading data...')


# In[7]:


train_dir='./hw4_data/train/'
train_data_x = [file for file in os.listdir(train_dir)]
train_data_x.sort()
train_x=[]
for i in train_data_x:
    train_x.append(imread(os.path.join(train_dir,i)))
train_x = np.array(train_x)

num_train = train_x.shape[0]


# In[8]:


test_dir='./hw4_data/test/'
test_data_x = [file for file in os.listdir(test_dir)]
test_data_x.sort()
test_x=[]
for i in test_data_x:
    test_x.append(imread(os.path.join(test_dir,i)))
test_x = np.array(test_x)

num_test = test_x.shape[0]


# In[9]:


print('building model(GAN)...')


# In[10]:


latent_size = 1024

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0001
adam_beta_1 = 0.5

# build the discriminator
print('Discriminator model:')
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='binary_crossentropy')

discriminator.summary()

# build the generator
print('Generator model:')
generator = build_generator(latent_size)

generator.summary()

latent = Input(shape=(latent_size,))

# get a fake image
fake = generator(latent)

# we only want to be able to train generation for the combined model
discriminator.trainable = False
fake = discriminator(fake)
combined = Model(latent, fake)

print('Combined model:')
combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='binary_crossentropy')

combined.summary()


# In[11]:


print('training model...')


# In[ ]:


# batch and latent size taken from the paper
epochs = 61
batch_size = 64

train_history = defaultdict(list)
test_history = defaultdict(list)

run_path = './GAN'

num_batches = int(train_x.shape[0] / batch_size)
    

for epoch in range(1, epochs + 1):

    e_start_time = time.time()
    print('Epoch {}/{}'.format(epoch, epochs))

    

    # we don't want the discriminator to also maximize the classification
    # accuracy of the auxiliary classifier on generated images, so we
    # don't train discriminator to produce class labels for generated
    # images (see https://openreview.net/forum?id=rJXTf9Bxg).

    # To preserve sum of sample weights for the auxiliary classifier,
    # we assign sample weight of 2 to the real images.
    #disc_sample_weight = [np.ones(2 * batch_size), np.concatenate((np.ones(batch_size) * 2, np.zeros(batch_size)))]

    epoch_gen_loss = []
    epoch_disc_loss = []

    for index in range(num_batches):
        # generate a new batch of noise
        noise = np.random.normal(0, 0.8, (batch_size, latent_size))

        # get a batch of real images
        image_batch = train_x[index * batch_size:(index + 1) * batch_size]
        image_batch = (image_batch-0.5)*2

        # generate a batch of fake images, using the generated labels as a conditioner.
        generated_images = generator.predict(noise, verbose=0)

        x = np.concatenate((image_batch, generated_images))

        # use one-sided soft real/fake labels
        # Salimans et al., 2016
        # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
        # real label, fake label
        soft_zero, soft_one = 0, 0.95 
        y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)

        # see if the discriminator can figure itself out...
        for _ in range(1):
            l = discriminator.train_on_batch(x, y)
        epoch_disc_loss.append(l)

        # make new noise. we generate 2 * batch size here such that we have
        # the generator optimize over an identical number of images as the discriminator
        noise = np.random.normal(0, 0.8, (2 * batch_size, latent_size))

        # we want to train the generator to trick the discriminator
        # For the generator, we want all the {fake, not-fake} labels to say not-fake
        trick = np.ones(2 * batch_size) * soft_one
        for _ in range(1):
            ll = combined.train_on_batch(noise, trick)
        epoch_gen_loss.append(ll)


    discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
    generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

    # generate an epoch report on performance       
    train_history['generator'].append(discriminator_train_loss)
    train_history['discriminator'].append(generator_train_loss)      


    # evaluate the testing loss here

    # generate a new batch of noise
    noise = np.random.normal(0, 0.8, (num_test, latent_size))

    # sample some labels from p_c and generate images from them
    generated_images = generator.predict(noise, verbose=False)

    x_real = test_x
    x_fake = generated_images
    y_real = np.array([1] * num_test) 
    y_fake = np.array([0] * num_test) 

    # see if the discriminator can figure itself out...
    discriminator_loss_real = discriminator.evaluate(x_real, y_real, verbose=False)
    discriminator_loss_fake = discriminator.evaluate(x_fake, y_fake, verbose=False)

    # Aqui começa a medir accuracy do discriminator
    y_pred_real = discriminator.predict(x_real)
    y_pred_fake = discriminator.predict(x_fake)
    y_pred_real = np.around(y_pred_real)
    y_pred_fake = np.around(y_pred_fake)
    
    discriminator_accuracy_real = accuracy_score(y_real, y_pred_real)
    discriminator_accuracy_fake = accuracy_score(y_fake, y_pred_fake)

    # generate an epoch report on performance

    test_history['discriminator_loss_real'].append(discriminator_loss_real)
    test_history['discriminator_loss_fake'].append(discriminator_loss_fake)
    test_history['discriminator_accuracy_real'].append(discriminator_accuracy_real)
    test_history['discriminator_accuracy_fake'].append(discriminator_accuracy_fake)

    print('{0:<15s} | {1:4s}'.format('', *discriminator.metrics_names))
    print('-' * 35)
    ROW_FMT = '{0:<15s} | {1:<4.4f}'
    print(ROW_FMT.format('generator', generator_train_loss))
    #print(ROW_FMT.format('generator (test)', test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator', discriminator_train_loss))
    #print(ROW_FMT.format('discriminator (test)', test_history['discriminator'][-1]))    

    print('Testing for hw4 :')
    print('{0:<15s} | {1:6s} | {2:8s} '.format('discriminator', 'loss', 'accuracy'))
    print('-' * 36)
    ROW_FMT = '{0:<15s} | {1:<4.4f} | {2:<4.4f}'
    print(ROW_FMT.format('real image', discriminator_loss_real, discriminator_accuracy_real))
    print(ROW_FMT.format('fake image', discriminator_loss_fake, discriminator_accuracy_fake)) 


    if epoch%2 == 1: 
        # save weights every epoch
        generator.save_weights(os.path.join(run_path, 'params_generator_epoch_{0:06d}.h5'.format(epoch)), True)
        discriminator.save_weights(os.path.join(run_path, 'params_discriminator_epoch_{0:06d}.h5'.format(epoch)), True)

        # generate some digits to display
        num_rows = 32
        noise = np.random.normal(0, 0.8, (num_rows, latent_size))

        # get a batch to display
        generated_images = generator.predict(noise, verbose=0)
        generated_images = (generated_images+1.0)/2.0
        fig = plot(generated_images, 4, 8)
        plt.savefig(os.path.join(run_path, 'plot_epoch_{0:03d}_generated.png'.format(epoch)), bbox_inches='tight')
        '''
        fig,axes = plt.subplots(4, 8, figsize=[64,64])
        plt.subplots_adjust(wspace=0, hspace=0)
        for i,iax in enumerate(axes.flatten()):
            iax.imshow((generated_images[i]+1.0)/2.0, interpolation='nearest')
            iax.axis('off')

        plt.savefig(os.path.join(run_path, 'plot_epoch_{0:03d}_generated.png'.format(epoch)), bbox_inches='tight')
        
        plt.close()
        '''
    print('eopch_{} time: {:.2f}'.format(epoch, (time.time()-e_start_time)/60))       


# In[ ]:


with open('gan-history.p', 'wb') as f:
    pickle.dump({'train' :train_history, 'test' :test_history}, f)
    
print('save gan-history.p')


# In[ ]:


print('total time: {:.2f}'.format((time.time()-start_time)/60))

with open('gan-history.p', 'rb') as f:
    ff = pickle.load(f)
print(ff)