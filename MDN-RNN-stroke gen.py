

import keras
from context import * # imports the MDN layer 
import numpy as np
import random
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D 
#get_ipython().run_line_magic('matplotlib', 'inline')

# Only for GPU use:
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
from keras import backend as K
K.set_session(sess)


# ### First download and process the dataset.

# In[3]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[5]:


Ty = 300
Tx = 300
m = len(strokes)


# In[6]:


def pad_stroke(stroke,Ty):
    _npads = Ty - stroke.shape[0] 
    padded_stroke = np.vstack ([ stroke,np.zeros((_npads,3)) ])
    #padded_strokes.shape
    return padded_stroke

def tranc_stroke(stroke, Ty):
    if stroke.shape[0] >= Ty:
        stroke = stroke[:Ty,]
    elif stroke.shape[0] < Ty:
        stroke = pad_stroke(stroke,Ty)
    return stroke

new_strokes = np.array(list(map(lambda x: tranc_stroke(x, Ty+1), strokes)))


# In[7]:


input_strokes = new_strokes[:,:Ty,:]
output_strokes = new_strokes[:,1:Ty+1,:]


# In[8]:


Xoh = input_strokes
Yoh = output_strokes






# Training Hyperparameters:
#SEQ_LEN = 30
SEQ_LEN = Tx
BATCH_SIZE = 32
HIDDEN_UNITS = 256
EPOCHS = 1
SEED = 2345  # set random seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
OUTPUT_DIMENSION = 3
NUMBER_MIXTURES = 10

# Sequential model
model = keras.Sequential()

# Add two LSTM layers, make sure the input shape of the first one is (?, 30, 3)
model.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(None,SEQ_LEN,OUTPUT_DIMENSION), return_sequences=True))
model.add(keras.layers.LSTM(HIDDEN_UNITS))

# Here's the MDN layer, need to specify the output dimension (3) and number of mixtures (10)
model.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))

# Now we compile the MDN RNN - need to use a special loss function with the right number of dimensions and mixtures.
model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())

# Let's see what we have:
model.summary()


# ## Process the Data and Train the Model
# 
# - Chop up the data into slices of the correct length, generate `X` and `y` for the training process.
# - Very similar process to the previous RNN examples!
# - We end up with 330000 examples - a pretty healthy dataset.

# In[8]:




# Fit the model
history = model.fit(Xoh, Yoh, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[keras.callbacks.TerminateOnNaN()])


# In[15]:


model.save('kanji_mdnrnn_model.h5')  # creates a HDF5 file 'my_model.h5'


