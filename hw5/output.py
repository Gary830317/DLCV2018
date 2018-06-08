
# coding: utf-8

# In[8]:


from keras.layers import *
from keras.models import *
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.sequence import pad_sequences
import os
import sys
from reader import readShortVideo
import pandas as pd
from matplotlib.pyplot import imread


# In[ ]:


model = ResNet50(weights='imagenet', include_top=False)


# In[ ]:


if sys.argv[1] == 'hw5_p1.sh':
    
    trimmed_valid_video_dir = sys.argv[2] #'./HW5_data/TrimmedVideos/video/valid/'
    trimmed_label = sys.argv[3] #'./HW5_data/TrimmedVideos/label/gt_valid.csv'
    output_dir = sys.argv[4] #'./'
    
    trimmed_valid_label = pd.read_csv(trimmed_label, index_col=None)
    trimmed_valid_video_name = trimmed_valid_label['Video_name'].tolist()
    trimmed_valid_video_category = trimmed_valid_label['Video_category'].tolist()
    
    all_valid_features = []
    for video_category, video_name in list(zip(trimmed_valid_video_category, trimmed_valid_video_name)):
        frames = readShortVideo(trimmed_valid_video_dir , video_category, video_name, downsample_factor=12, rescale_factor=1)
        features = model.predict(frames)
        all_valid_features.append(features.reshape(-1, 2048))
        
    P1_valid_x = np.array(list(map(lambda l: np.average(l, axis = 0), all_valid_features)))

    inputs = Input(shape=(2048,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(256, activation='relu')(x)
    out = Dense(11, activation='softmax')(x)
    CNN_model = Model(inputs=inputs, outputs=out)

    CNN_model.load_weights('CNN_model.h5')
    
    CNN_predict = CNN_model.predict(P1_valid_x)
    CNN_predict = np.argmax(CNN_predict, axis=1).astype(str)
    
    with open(os.path.join(output_dir, 'p1_valid.txt'), 'w') as file:
        for i in CNN_predict:
            file.write(i+'\n')
    


# In[ ]:


if sys.argv[1] == 'hw5_p2.sh':
    
    trimmed_valid_video_dir = sys.argv[2] #'./HW5_data/TrimmedVideos/video/valid/'
    trimmed_label = sys.argv[2] #'./HW5_data/TrimmedVideos/label/gt_valid.csv'
    output_dir = sys.argv[4] #'./'
    
    trimmed_valid_label = pd.read_csv(trimmed_label, index_col=None)
    trimmed_valid_video_name = trimmed_valid_label['Video_name'].tolist()
    trimmed_valid_video_category = trimmed_valid_label['Video_category'].tolist()
    
    all_valid_features = []
    for video_category, video_name in list(zip(trimmed_valid_video_category, trimmed_valid_video_name)):
        frames = readShortVideo(trimmed_valid_video_dir , video_category, video_name, downsample_factor=12, rescale_factor=1)
        features = model.predict(frames)
        all_valid_features.append(features.reshape(-1, 2048))
        
    P2_valid_x = pad_sequences(all_valid_features, maxlen=200)
    
    input_shape = P2_valid_x[0].shape
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(512), name = 'lstm_features')(inputs)
    x = Dense(256, activation='relu')(x)
    out = Dense(11, activation='softmax')(x)
    RNN_model = Model(inputs=inputs, outputs=out)

    RNN_model.load_weights('RNN_model.h5')
    
    RNN_predict = RNN_model.predict(P2_valid_x)
    RNN_predict = np.argmax(RNN_predict, axis=1).astype(str)
    
    with open(os.path.join(output_dir, 'p2_valid.txt'), 'w') as file:
        for i in RNN_predict:
            file.write(i+'\n')


# In[ ]:


if sys.argv[1] == 'hw5_p3.sh':
    fulllength_valid_video_dir = sys.argv[2]#'./HW5_data/FullLengthVideos/videos/valid'
    output_dir = sys.argv[3] #'./'

    fulllength_valid_video_list = [file for file in os.listdir(fulllength_valid_video_dir)]
    fulllength_valid_video_list.sort()
    all_valid_fullvideo_features = []
    for video in fulllength_valid_video_list:
        fulllength_valid_frame_dir = os.path.join(fulllength_valid_video_dir, video)
        fulllength_valid_frame = [file for file in os.listdir(fulllength_valid_frame_dir) if file.endswith('.jpg')]    
        fulllength_valid_frame.sort()
        video_frames = []
        for frame in fulllength_valid_frame:
            video_frames.append(imread(os.path.join(fulllength_valid_frame_dir,frame)))
        features = model.predict(np.array(video_frames))
        all_valid_fullvideo_features.append(features.reshape(-1, 2048))
        
    Max_time_steps = 400    
    
    input_shape = (Max_time_steps, 2048)#P3_train_x[0].shape
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(512, return_sequences=True))(inputs)
    x = Dense(256, activation='relu')(x)
    out = Dense(11, activation='softmax')(x)

    seq2seq_model = Model(inputs=inputs, outputs=out)
    seq2seq_model.load_weights('seq2seq_model.h5')

    
    for video_frames_feature, video_name in zip(all_valid_fullvideo_features, fulllength_valid_video_list):
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

        with open(video_name+'.txt', 'w') as file:
            for y in output_labels:
                file.write(y+'\n')

