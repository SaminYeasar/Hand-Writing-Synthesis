#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import keras
#import mdn
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda ,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import h5py


# In[2]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys 
import os
sys.path.insert(0,'..')
from utils import plot_stroke


# In[3]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[4]:


with open('sentences.txt') as f:
    texts = f.readlines()


# In[5]:


chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .#'  # for other char in texts


# unique contains all the unique characters in the file
unique = sorted(set(chars))

# creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(unique)}
idx2char = {i:u for i, u in enumerate(unique)}

num_char = len(char2idx)


# In[6]:


########## gives the best representation so far
stroke_len = 300
char_len = int(stroke_len/25)


# In[7]:


Ty = 300   #output data length
Tx = char_len   #input data length
no_examples = len(strokes)  #no of examples


print("No of examples",no_examples)
print("Output data length",Ty)
print("Input data length",Tx)


# In[8]:


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


# In[9]:


texts = tranc_text(texts, char_len)
n_texts = str2num(texts)
C = np.array(list(map(lambda x: to_categorical(x, num_classes=len(char2idx)), n_texts)))
input_feat_size = C.shape[2]


# In[10]:


print("Input Data Shape",C.shape)
print("Input feature shape",input_feat_size)


# In[11]:


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
output_strokes = np.array(list(map(lambda x: tranc_stroke(x, Ty), strokes)))
output_feat_size = output_strokes.shape[2]


# In[12]:


print("Output Data Shape",output_strokes.shape)
print("Output feature shape",output_feat_size)


# In[13]:


def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s


# In[14]:


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, kernel_initializer='random_uniform', activation = "tanh")
densor2 = Dense(1, kernel_initializer='random_uniform' , activation = "tanh")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)


# In[15]:


def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator ([s_prev,a])
    
    e =  densor1(concat) 
    e = BatchNormalization()(e)
    
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas,a])
    return context


# In[16]:


n_a = 128
n_s = 256
post_activation_LSTM_cell = LSTM(n_s,activation='tanh',recurrent_dropout=0.2,return_state = True)
out_densor1 = Dense(100, kernel_initializer='random_uniform', activation = "tanh")
out_densor2 = Dense(100, kernel_initializer='random_uniform' , activation = "tanh")
out_densor3 = Dense(3, kernel_initializer='random_uniform' , activation = "tanh")
output_layer = Dense(3)


# In[17]:


def model(Tx, Ty, n_a, n_s, input_feat_size, output_feat_size):
    X = Input(shape=(Tx, input_feat_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    a = Bidirectional(LSTM(n_a,activation='tanh',recurrent_dropout=0.2,return_sequences=True),input_shape=(Tx, input_feat_size))(X)
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context,initial_state= [s, c])
        
        output_l1 =  out_densor1(s) 
        output_l1 = BatchNormalization()(output_l1)
        
        output_l2 = out_densor2(output_l1) 
        output_l2 = BatchNormalization()(output_l2)
        
        out = out_densor3(output_l2)
        #out = output_layer(s)
        outputs.append(out)
    model = Model(inputs=[X,s0,c0], outputs=outputs)
    return model


# In[18]:


model = model(Tx, Ty, n_a, n_s, input_feat_size, output_feat_size)


# In[ ]:





# In[19]:


# All parameter gradients will be clipped to
# a maximum value of 100 and
# a minimum value of -100.
opt = Adam(lr=0.005, decay=0.01, beta_1=0.9, beta_2=0.999,clipvalue=100)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[20]:


checkpoint_path = "AttentionModel-Text2Stroke.h5"
if os.path.isfile(checkpoint_path) == True:
    model.load_weights('AttentionModel-Text2Stroke.h5')


# In[21]:


s0 = np.zeros((no_examples, n_s))
c0 = np.zeros((no_examples, n_s))
outputs = list(output_strokes.swapaxes(0,1))


# In[ ]:


#filepath="AttentionModel-LSTM-NN-strok-gen-weights.{epoch:02d}-{val_loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]


# In[ ]:


model.fit([C, s0, c0], outputs, epochs=100, batch_size=32)
print('training completed')
model.save('AttentionModel-Text2Stroke.h5')


# In[ ]:




