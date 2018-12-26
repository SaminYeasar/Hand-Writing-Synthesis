
# coding: utf-8

# In[11]:


import keras
import mdn
import numpy as np
import random
import matplotlib.pyplot as plt


# Only for GPU use:
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


# In[12]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[29]:


Ty = 300
Tx = 300
m = len(strokes)


# In[30]:


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


# In[31]:


input_strokes = new_strokes[:,:Ty,:]
output_strokes = new_strokes[:,1:Ty+1,:]


# In[32]:


Xoh = input_strokes
Yoh = output_strokes


# In[34]:


"""
plt.plot(Xoh[0].T[1], Xoh[0].T[2])
plt.title("Raw values (diffs) for one training example")
plt.show()
"""


# In[35]:


"""
plt.plot(Xoh[0].T[1].cumsum(), -1 * Xoh[0].T[2].cumsum())
plt.title("Accumulated values for one training example")
plt.show()
"""


# In[36]:


# Training Hyperparameters:
SEQ_LEN = 30
#SEQ_LEN = Tx
BATCH_SIZE = 30
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

#model.add(keras.layers.Dense(N_HIDDEN, batch_input_shape=(None, 1), activation='relu'))
# Here's the MDN layer, need to specify the output dimension (3) and number of mixtures (10)
model.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))

# Now we compile the MDN RNN - need to use a special loss function with the right number of dimensions and mixtures.
model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())

# Let's see what we have:
model.summary()


# In[37]:


# Functions for slicing up data
def slice_sequence_examples(sequence, num_steps):
    xs = []
    for i in range(len(sequence) - num_steps - 1):
        example = sequence[i: i + num_steps]
        xs.append(example)
    return xs

def seq_to_singleton_format(examples):
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs,ys)

# Prepare training data as X and Y.
slices = []
for seq in Xoh:
    slices +=  slice_sequence_examples(seq, SEQ_LEN+1)
X, y = seq_to_singleton_format(slices)

X = np.array(X)
y = np.array(y)

print("Number of training examples:")
print("X:", X.shape)
print("y:", y.shape)


# In[ ]:


# Fit the model
EPOCHS = 1
history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[keras.callbacks.TerminateOnNaN()])


# In[ ]:


model.save_weights('mdn_stroke2stroke_epoch{}.h5'.format(EPOCHS))  # creates a HDF5 file 'my_model.h5'


# In[ ]:


################################################



################################################

