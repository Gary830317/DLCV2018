
# coding: utf-8

# In[2]:


import os
import cv2
import sys
import numpy as np
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications.resnet50 import ResNet50


# In[3]:


np.random.seed(100)

shots = [1, 5, 10] 
train_novel_dir = sys.argv[1] #'task2_dataset/novel'
test_dir = sys.argv[2] #'test'
output_dir = sys.argv[3] #'./'


im_length, im_width = 224, 224


# In[4]:


label_dir = [file for file in os.listdir(train_novel_dir) if file.startswith('class')]
label_dir.sort()

test_image = []
images = [file for file in os.listdir(test_dir) if file.endswith('.png')]
images = sorted(images, key=lambda image: int(image[:-4]))
for image in images:
    test_image.append(cv2.resize((cv2.imread(os.path.join(test_dir,image))/255).astype('float32'),(im_length, im_width),interpolation=cv2.INTER_CUBIC))

test_image = np.array(test_image)

test_label_dic = np.array([str(label[-2:]) for label in label_dir])


# In[5]:


model = ResNet50(include_top=True, weights=None, classes=80)

layer_name = 'flatten_1' # modify the layer name every time you run the program
extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


# In[6]:


inputs = Input(shape=(im_length, im_width, 3))
x = extractor(inputs)
out = Dense(20, activation='softmax')(x)
mlp2 = Model(inputs = inputs, outputs = out)

mlp2.summary()


# In[ ]:


for shot in shots:
    mlp2.load_weights('{}_shot_Resnet.h5'.format(shot))
    
    y_pred = mlp2.predict(test_image)
    y_pred = y_pred.argmax(axis=-1)
    predicted_label = np.array(test_label_dic)[y_pred]
    
    with open(os.path.join(output_dir,'{}_shot.csv'.format(shot)),'w') as file:
        file.write('image_id,predicted_label')
        file.write('\n')
        for index, l in enumerate(predicted_label):
            file.write('{},{}'.format(index,l))
            file.write('\n')

    print('{}_shot.csv saved'.format(shot))

