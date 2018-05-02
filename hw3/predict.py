
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
from model import *


# In[19]:


def to_image(predict, dir_, output_dir):
    label2rgb = np.array([[0.,255.,255.], [255.,255.,0.], [255.,0.,255.], [0.,255.,0.], [0.,0.,255.], [255.,255.,255.], [0.,0.,0.]])
    for idx, i in enumerate(predict):
        mask=label2rgb[np.argmax(i, axis=2)]
        k=os.path.split(test_data_x[idx])[1]
        scipy.misc.imsave(os.path.join(output_dir, '{}_mask.png'.format(k[:4])).replace('\r', ''), mask)


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

