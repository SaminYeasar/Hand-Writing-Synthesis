#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
import tensorflow as tf
gpu_fraction = 0.1
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
"""


# In[ ]:


"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
"""


# In[ ]:


"""
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
"""


# In[5]:


import tensorflow as tf
import keras
import mdn
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda,MaxPooling1D,Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.backend import clip
import keras.backend as K
import numpy as np
#from keras.preprocessing.text import Tokenizer
#from faker import Faker
#import random
#from tqdm import tqdm
#from babel.dates import format_date
#from at_nmt_utils import *
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys 

sys.path.insert(0,'..')
from utils import plot_stroke


# In[7]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[8]:


with open('sentences.txt') as f:
    texts = f.readlines()


# In[9]:



chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .#'  # for other char in texts


# unique contains all the unique characters in the file
unique = sorted(set(chars))

# creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(unique)}
idx2char = {i:u for i, u in enumerate(unique)}

num_char = len(char2idx)


# In[10]:


########## gives the best representation so far
stroke_len = 300
char_len =int(stroke_len/25)


# In[11]:
def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s

def check_char (char2idx,val):
    result = []
    if char2idx.get(val)!=None:
        result = char2idx[val]
    elif char2idx.get(val)==None :
        result = char2idx['#']
    return result

def str2num(texts):
    input_text = []
    for f in range(len(texts)):
        inps = texts[f]   
        data = list(map( lambda val: check_char (char2idx,val),inps ))
        input_text.append(data)
        #np.concatenate((a, b), axis=0)
    return np.vstack(input_text)

def pad_texts(text, char_len):  
    pads = char_len - len(text)
    for i in range(int(pads)):
        text = text+str(' ')
    return text

def tranc_text(texts, char_len):
    for i in range (len(texts)):
        if len(texts[i]) > char_len:
            texts[i] = texts[i][0:int(char_len)]

        elif len(texts[i]) < char_len:
            texts[i] = pad_texts(texts[i],char_len)
    return texts 


# In[12]:


texts = tranc_text(texts, char_len)
n_texts = str2num(texts)
C = np.array(list(map(lambda x: to_categorical(x, num_classes=len(char2idx)), n_texts)))


# In[13]:


Tx = stroke_len
Ty = stroke_len

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
    #return input_stroke,output_stroke
    return stroke

new_strokes = np.array(list(map(lambda x: tranc_stroke(x, Ty+1), strokes)))
Xoh = new_strokes[:,:Ty,:]
Yoh = new_strokes[:,1:Ty+1,:]


# In[14]:


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(100,kernel_initializer='normal', activation = "relu")
densor2 = Dense(1,kernel_initializer='normal', activation = "relu")
pooling = MaxPooling1D(pool_size=25, strides=25, padding="same")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)
dropout = Dropout(0.2)

# In[15]:


def one_step_attention(a, s_prev,C):
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator ([s_prev,a]) # (?,500,600)
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)     # (?,500,100)
   # e = clip(e,-10,10)
    
    e = pooling(e)          # (?,20,100)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)   # (?,20,1)
   # energies = clip(enegies,-10,10) 
   # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)   # (?,20,1)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    #context = dotor([alphas,a])
    context = dotor([alphas,C])  # context = (?,1,55); alpha = (?,20,1) , alpha = (?,20,55)
    return context


# In[16]:


n_a = 128  #bi-directional in total ends up having 300 variables
n_s = 256
output_dim = 3
n_mix = 10
input_feat_size = Xoh.shape[2]   #3
output_feat_size = Yoh.shape[2]   #3

post_activation_LSTM_cell = LSTM(n_s, return_state = True)
#output_layer = Dense(len(machine_vocab), activation=softmax)
mix_model = mdn.MDN(output_dim, n_mix)
#output_layer = Dense(3, activation = "sigmoid")



# In[18]:


def model(Tx, Ty, n_a, n_s, input_feat_size, output_feat_size, char_len, num_char):


    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, input_feat_size))
    C = Input(shape=(char_len, num_char))   # one hot encoded vector
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True),input_shape=(Tx, input_feat_size))(X)
    #a = clip(a,-100,100)
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s, C)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context,initial_state= [s, c])
       # s = clip(s,-100,100)
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        #out = output_layer(s)
        
        out = mix_model(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    
    model = Model(inputs=[X,C,s0,c0], outputs=outputs)

    ### END CODE HERE ###
    
    return model


# In[28]:



#char_len = total number of characters in input text C
# num_char = number of possible characters
model = model(Tx, Ty, n_a, n_s, input_feat_size, input_feat_size,char_len, num_char)


# In[29]:


model.summary()


# In[30]:


opt = Adam(lr=0.005, decay=0.01, beta_1=0.9, beta_2=0.999)
#model.compile(optimizer=opt,
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

model.compile(loss=mdn.get_mixture_loss_func(output_dim, n_mix), optimizer=opt)


# In[19]:


m = Xoh.shape[0]  # no of examples we have for training
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


# In[ ]:


BATCH_SIZE = 32
EPOCHS = 10

filepath="seq2seq-LSTM-50-weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[15]:

history = model.fit([Xoh,C, a0, c0], outputs, validation_split=0.33, batch_size=BATCH_SIZE, epochs=EPOCHS,callbacks=callbacks_list)


#history = model.fit([Xoh,C, s0, c0], outputs, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[keras.callbacks.TerminateOnNaN()])


# In[ ]:


model.save('Attention_mdn_batch10_epoch32.h5')  # creates a HDF5 file 'my_model.h5'

