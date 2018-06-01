
# coding: utf-8

# In[1]:


from keras.applications.resnet50 import ResNet50
import numpy as np
import pandas as pd
from reader import readShortVideo
import os
from keras.layers import *
from keras.models import *
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import pickle
from sklearn.manifold.t_sne import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import imread
from sklearn.metrics import accuracy_score


# In[2]:


model = ResNet50(weights='imagenet', include_top=False)


# In[3]:


output_dir = './'


# In[4]:


trimmed_dir = './HW5_data/TrimmedVideos/'
trimmed_train_video_dir = './HW5_data/TrimmedVideos/video/train'
trimmed_valid_video_dir = './HW5_data/TrimmedVideos/video/valid'


# In[5]:


trimmed_train_label = pd.read_csv(os.path.join(trimmed_dir, 'label', 'gt_train.csv'), index_col=None)
trimmed_train_video_name = trimmed_train_label['Video_name'].tolist()
trimmed_train_video_category = trimmed_train_label['Video_category'].tolist()
trimmed_train_action_labels = trimmed_train_label['Action_labels'].tolist()


# In[6]:


trimmed_valid_label = pd.read_csv(os.path.join(trimmed_dir, 'label', 'gt_valid.csv'), index_col=None)
trimmed_valid_video_name = trimmed_valid_label['Video_name'].tolist()
trimmed_valid_video_category = trimmed_valid_label['Video_category'].tolist()
trimmed_valid_action_labels = trimmed_valid_label['Action_labels'].tolist()


# In[7]:


all_train_frames=[]
for video_category, video_name in list(zip(trimmed_train_video_category, trimmed_train_video_name)):
    frames = readShortVideo(trimmed_train_video_dir , video_category, video_name, downsample_factor=12, rescale_factor=1)
    all_train_frames.append(frames)


# In[8]:


all_valid_frames=[]
for video_category, video_name in list(zip(trimmed_valid_video_category, trimmed_valid_video_name)):
    frames = readShortVideo(trimmed_valid_video_dir , video_category, video_name, downsample_factor=12, rescale_factor=1)
    all_valid_frames.append(frames)


# In[9]:


all_train_features = []
for f in all_train_frames:
    features = model.predict(f)
    all_train_features.append(features.reshape(-1, 2048))

all_valid_features = []
for f in all_valid_frames:
    features = model.predict(f)
    all_valid_features.append(features.reshape(-1, 2048))


# In[10]:


with open('TrimmedVideos_features.p', 'wb') as file:
    pickle.dump({'all_train_features':all_train_features, 'all_valid_features':all_valid_features}, file)


# Problem 1 

# In[11]:


P1_train_x = np.array(list(map(lambda l: np.average(l, axis = 0), all_train_features)))
P1_train_y = np.array(to_categorical(trimmed_train_action_labels, 11))

P1_valid_x = np.array(list(map(lambda l: np.average(l, axis = 0), all_valid_features)))
P1_valid_y = np.array(to_categorical(trimmed_valid_action_labels, 11))


# In[18]:


CNN_epochs = 50

inputs = Input(shape=(2048,))

x = Dense(1024, activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
out = Dense(11, activation='softmax')(x)

CNN_model = Model(inputs=inputs, outputs=out)
CNN_model.summary()
CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
CNN_history = CNN_model.fit(P1_train_x, P1_train_y, shuffle=True, epochs=CNN_epochs, validation_split=0.1, callbacks=[EarlyStopping(patience = 5)])
CNN_model.save_weights('CNN_model.h5')


# In[19]:


CNN_predict = CNN_model.predict(P1_valid_x)
CNN_predict = np.argmax(CNN_predict, axis=1).astype(str)


# In[20]:


CNN_validation_score = CNN_model.evaluate(P1_valid_x, P1_valid_y)
print(CNN_validation_score)


# In[21]:


with open('p1_valid.txt', 'w') as file:
    for i in CNN_predict:
        file.write(i+'\n')


# In[22]:


with open('CNN_history.p', 'wb') as file:
    pickle.dump(CNN_history.history, file)


# In[23]:


CNN_val_loss =  CNN_history.history['val_loss'][:-3]
CNN_val_acc =  CNN_history.history['val_acc'][:-3]
CNN_tra_loss =  CNN_history.history['loss'][:-3]
CNN_tra_acc =  CNN_history.history['acc'][:-3]


# In[24]:


fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].plot(range(1, len(CNN_val_loss)+1), CNN_val_loss, label='validation loss')
ax[0].plot(range(1, len(CNN_tra_loss)+1), CNN_tra_loss, label='train loss')
ax[0].set_xlabel('Training epochs', fontsize=16)
ax[0].set_title('CNN-based loss', fontsize=20)
ax[0].legend(loc="upper right", fontsize=10)
ax[1].plot(range(1, len(CNN_val_acc)+1), CNN_val_acc, label='validation accuracy')
ax[1].plot(range(1, len(CNN_tra_acc)+1), CNN_tra_acc, label='train accuracy')
ax[1].set_xlabel('Training epochs', fontsize=16)
ax[1].set_title('CNN-based accuracy', fontsize=20)
ax[1].legend(loc="upper right", fontsize=10)
plt.savefig(os.path.join(output_dir, 'CNN-based.jpg'), bbox_inches='tight')
plt.close(fig)


# Problem 2

# In[25]:


P2_train_x = pad_sequences(all_train_features, maxlen=200)
P2_train_y = np.array(to_categorical(trimmed_train_action_labels, 11))

P2_valid_x = pad_sequences(all_valid_features, maxlen=200)
P2_valid_y = np.array(to_categorical(trimmed_valid_action_labels, 11))


# In[123]:


RNN_epochs = 20
patience = 5

input_shape = P2_train_x[0].shape
inputs = Input(shape=input_shape)

x = Bidirectional(LSTM(512), name = 'lstm_features')(inputs)
x = Dense(256, activation='relu')(x)
out = Dense(11, activation='softmax')(x)

RNN_model = Model(inputs=inputs, outputs=out)
RNN_model.summary()
RNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
RNN_history = RNN_model.fit(P2_train_x, P2_train_y, shuffle=True, epochs=RNN_epochs, validation_split=0.1, callbacks=[EarlyStopping(patience = patience)])
RNN_model.save_weights('RNN_model.h5')


# In[124]:


RNN_predict = RNN_model.predict(P2_valid_x)
RNN_predict = np.argmax(RNN_predict, axis=1).astype(str)


# In[125]:


RNN_validation_score = RNN_model.evaluate(P2_valid_x, P2_valid_y)
print(RNN_validation_score)


# In[126]:


with open('p2_valid.txt', 'w') as file:
    for i in RNN_predict:
        file.write(i+'\n')


# In[127]:


with open('RNN_history.p', 'wb') as file:
    pickle.dump(RNN_history.history, file)


# In[128]:


RNN_val_loss = RNN_history.history['val_loss'][:-3]
RNN_val_acc = RNN_history.history['val_acc'][:-3]
RNN_tra_loss = RNN_history.history['loss'][:-3]
RNN_tra_acc = RNN_history.history['acc'][:-3]


# In[129]:


fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].plot(range(1, len(RNN_val_loss)+1), RNN_val_loss, label='validation loss')
ax[0].plot(range(1, len(RNN_tra_loss)+1), RNN_tra_loss, label='train loss')
ax[0].set_xlabel('Training epochs', fontsize=16)
ax[0].set_title('RNN-based loss', fontsize=20)
ax[0].legend(loc="upper right", fontsize=10)
ax[1].plot(range(1, len(RNN_val_acc)+1), RNN_val_acc, label='validation accuracy')
ax[1].plot(range(1, len(RNN_tra_acc)+1), RNN_tra_acc, label='train accuracy')
ax[1].set_xlabel('Training epochs', fontsize=16)
ax[1].set_title('RNN-based accuracy', fontsize=20)
ax[1].legend(loc="upper right", fontsize=10)
plt.savefig(os.path.join(output_dir, 'RNN-based.jpg'), bbox_inches='tight')
plt.close(fig)


# In[32]:


CNN_features_2d = TSNE(n_components=2).fit_transform(P1_valid_x) 

fig = plt.figure(figsize=(8, 8))
plt.scatter(CNN_features_2d[:,0], CNN_features_2d[:,1], c=trimmed_valid_action_labels, alpha=0.5, cmap="nipy_spectral")

plt.title('CNN video features', fontsize=20)
plt.savefig(os.path.join(output_dir, 'CNN_tsne.jpg'), bbox_inches='tight')
plt.close(fig)


# In[130]:


layer_name = 'lstm_features'
intermediate_layer_model = Model(inputs=RNN_model.input,
                                 outputs=RNN_model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(P2_valid_x)


# In[131]:


RNN_features_2d = TSNE(n_components=2).fit_transform(intermediate_output) 

fig = plt.figure(figsize=(8, 8))
plt.scatter(RNN_features_2d[:,0], RNN_features_2d[:,1], c=trimmed_valid_action_labels, alpha=0.5, cmap="nipy_spectral")

plt.title('RNN video features', fontsize=20)
plt.savefig(os.path.join(output_dir, 'RNN_tsne.jpg'), bbox_inches='tight')
plt.close(fig)


# Problem 3

# In[56]:


fulllength_dir = './HW5_data/FullLengthVideos/'
fulllength_train_video_dir = './HW5_data/FullLengthVideos/videos/train'
fulllength_valid_video_dir = './HW5_data/FullLengthVideos/videos/valid'


# In[70]:


fulllength_train_label_txts = [file for file in os.listdir(os.path.join(fulllength_dir, 'labels', 'train')) if file.endswith('.txt')]
fulllength_train_label_txts.sort()
fulllength_train_action_labels = []
all_train_fullvideo_features = []
for txt in fulllength_train_label_txts:
    with open(os.path.join(fulllength_dir, 'labels', 'train', txt), 'r') as file:
        fulllength_train_action_labels.append([line.rstrip() for line in file])
    fulllength_train_frame_dir = os.path.join(fulllength_train_video_dir, txt[:-4])
    fulllength_train_frame = [file for file in os.listdir(fulllength_train_frame_dir) if file.endswith('.jpg')]    
    fulllength_train_frame.sort()
    video_frames = []
    for frame in fulllength_train_frame:
        video_frames.append(imread(os.path.join(fulllength_train_frame_dir,frame)))
    features = model.predict(np.array(video_frames))
    all_train_fullvideo_features.append(features.reshape(-1, 2048))


# In[63]:


fulllength_valid_label_txts = [file for file in os.listdir(os.path.join(fulllength_dir, 'labels', 'valid')) if file.endswith('.txt')]
fulllength_valid_label_txts.sort()
fulllength_valid_action_labels = []
all_valid_fullvideo_features = []
for txt in fulllength_valid_label_txts:
    with open(os.path.join(fulllength_dir, 'labels', 'valid', txt), 'r') as file:
        fulllength_valid_action_labels.append([line.rstrip() for line in file])
    fulllength_valid_frame_dir = os.path.join(fulllength_valid_video_dir, txt[:-4])
    fulllength_valid_frame = [file for file in os.listdir(fulllength_valid_frame_dir) if file.endswith('.jpg')]    
    fulllength_valid_frame.sort()
    video_frames = []
    for frame in fulllength_valid_frame:
        video_frames.append(imread(os.path.join(fulllength_valid_frame_dir,frame)))
    features = model.predict(np.array(video_frames))
    all_valid_fullvideo_features.append(features.reshape(-1, 2048))


# In[75]:


with open('FullLengthVideos_features.p', 'wb') as file:
    pickle.dump({'all_train_fullvideo_features':all_train_fullvideo_features, 'all_valid_fullvideo_features':all_valid_fullvideo_features}, file)


# In[143]:


Max_time_steps = 400

P3_train_x = []
P3_train_y = []

P3_valid_x = []
P3_valid_y = []

for index, video_frames_feature in enumerate(all_train_fullvideo_features):
    ll = len(video_frames_feature)
    for i in range(0,ll-Max_time_steps,200):
        P3_train_x.append(video_frames_feature[i:min(i+Max_time_steps, ll)])
        P3_train_y.append(fulllength_train_action_labels[index][i:min(i+Max_time_steps, ll)])
        
for index, video_frames_feature in enumerate(all_valid_fullvideo_features):
    ll = len(video_frames_feature)
    for i in range(0,ll-Max_time_steps,200):
        P3_valid_x.append(video_frames_feature[i:min(i+Max_time_steps, ll)])
        P3_valid_y.append(fulllength_valid_action_labels[index][i:min(i+Max_time_steps, ll)])


# In[144]:


P3_train_x = pad_sequences(P3_train_x, maxlen=400)
P3_train_y = pad_sequences(P3_train_y, maxlen=400)
P3_train_y = np.array(to_categorical(P3_train_y, 11))

P3_valid_x = pad_sequences(P3_valid_x, maxlen=400)
P3_valid_y = pad_sequences(P3_valid_y, maxlen=400)
P3_valid_y = np.array(to_categorical(P3_valid_y, 11))


# In[145]:


epochs = 20

input_shape = (Max_time_steps, 2048)#P3_train_x[0].shape
inputs = Input(shape=input_shape)

x = Bidirectional(LSTM(512, return_sequences=True))(inputs)
x = Dense(256, activation='relu')(x)
out = Dense(11, activation='softmax')(x)

seq2seq_model = Model(inputs=inputs, outputs=out)
seq2seq_model.load_weights('RNN_model.h5')
seq2seq_model.summary()

seq2seq_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
seq2seq_history = seq2seq_model.fit(P3_train_x, P3_train_y, shuffle=True, epochs=epochs, validation_split=0.1, callbacks=[EarlyStopping(patience = patience)])
seq2seq_model.save_weights('seq2seq_model.h5')


# In[148]:


seq2seq_val_loss = seq2seq_history.history['val_loss'][:-3]
seq2seq_val_acc = seq2seq_history.history['val_acc'][:-3]
seq2seq_tra_loss = seq2seq_history.history['loss'][:-3]
seq2seq_tra_acc = seq2seq_history.history['acc'][:-3]


# In[149]:


fig, ax = plt.subplots(1,2,figsize=(20,8))
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].plot(range(1, len(seq2seq_val_loss)+1), seq2seq_val_loss, label='validation loss')
ax[0].plot(range(1, len(seq2seq_tra_loss)+1), seq2seq_tra_loss, label='train loss')
ax[0].set_xlabel('Training epochs', fontsize=16)
ax[0].set_title('seq2seq loss', fontsize=20)
ax[0].legend(loc="upper right", fontsize=10)
ax[1].plot(range(1, len(seq2seq_val_acc)+1), seq2seq_val_acc, label='validation accuracy')
ax[1].plot(range(1, len(seq2seq_tra_acc)+1), seq2seq_tra_acc, label='train accuracy')
ax[1].set_xlabel('Training epochs', fontsize=16)
ax[1].set_title('seq2seq accuracy', fontsize=20)
ax[1].legend(loc="upper right", fontsize=10)
plt.savefig(os.path.join(output_dir, 'seq2seq.jpg'), bbox_inches='tight')
plt.close(fig)


# In[146]:


seq2seq_predict = seq2seq_model.predict(P3_valid_x)
seq2seq_predict = np.argmax(seq2seq_predict, axis=2).astype(str)


# In[147]:


seq2seq_validation_score = seq2seq_model.evaluate(P3_valid_x, P3_valid_y)
print(seq2seq_validation_score)


# In[195]:


for video_frames_feature, video_name, labels in zip(all_valid_fullvideo_features, fulllength_valid_label_txts, fulllength_valid_action_labels):
    ll = len(video_frames_feature)
    output_labels = []
    r = list(range(0,ll,200))
    ind=0
    for i in r:
        predict = seq2seq_model.predict(pad_sequences(np.expand_dims(np.array(video_frames_feature[i:min(i+Max_time_steps, ll)]), axis=0), maxlen=400))
        if ind == 0:
            ind+=300
            output_labels.extend(np.argmax(predict[0][:300], axis=1).astype(str))
        elif ind>ll:
            pass
        else:
            ind+=200
            if ind>=ll:
                output_labels.extend(np.argmax(predict[0][-(ll-ind+200):], axis=1).astype(str))
            else:
                output_labels.extend(np.argmax(predict[0][100:300], axis=1).astype(str))

    print(len(output_labels), len(labels))
    print(accuracy_score(output_labels, labels))
    with open(video_name, 'w') as file:
        for y in output_labels:
            file.write(y+'\n')


# In[151]:


with open('seq2seq_history.p', 'wb') as file:
    pickle.dump(seq2seq_history.history, file)

