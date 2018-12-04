#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""
import tensorflow as tf
gpu_fraction = 0.1
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
"""


# In[6]:


"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
"""


# In[16]:


"""
import tensorflow as tf
tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) 
print
"""


# In[10]:


#tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[12]:


#tf.device


# In[7]:


"""
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
"""


# In[2]:


#!pip install tensorflow_probability


# In[1]:


import os
import tensorflow as tf
import keras
#import mdn
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda,MaxPooling1D ,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from keras.callbacks import ModelCheckpoint
#from keras.preprocessing.text import Tokenizer
#from at_nmt_utils import *
import matplotlib.pyplot as plt


# In[2]:


#%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys 

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
char_len


# In[7]:


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


# In[8]:


texts = tranc_text(texts, char_len)
n_texts = str2num(texts)
C = np.array(list(map(lambda x: to_categorical(x, num_classes=len(char2idx)), n_texts)))


# In[9]:


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


# In[10]:


def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s


# In[11]:


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(100, activation = "tanh")
densor2 = Dense(1, activation = "relu")
pooling = MaxPooling1D(pool_size=25, strides=25, padding="same")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


# In[12]:


def one_step_attention(a, s_prev,C):
    s_prev = repeator(s_prev)
    concat = concatenator ([s_prev,a]) # (?,500,600)
    e =  densor1(concat) 
    e = pooling(e)          # (?,20,100)
    e2 = BatchNormalization()(e)
    energies = densor2(e2)   # (?,20,1)
    alphas = activator(energies)   # (?,20,1)
    #context = dotor([alphas,a])
    context = dotor([alphas,C])  # context = (?,1,55); alpha = (?,20,1) , alpha = (?,20,55)
    return context


# In[13]:


n_a = 128  #bi-directional in total ends up having 300 variables
n_s = 256
output_dim = 3
n_mix = 10
input_feat_size = Xoh.shape[2]   #3
output_feat_size = Yoh.shape[2]   #3

#post_activation_LSTM_cell = LSTM(n_s, return_state = True)
#output_layer = Dense(len(machine_vocab), activation=softmax)
#mix_model = mdn.MDN(output_dim, n_mix)
#output_layer = Dense(3, activation = "sigmoid")

post_activation_LSTM_cell = LSTM(n_s,activation='tanh',recurrent_dropout=0.2,return_state = True)
out_densor1 = Dense(100, kernel_initializer='random_uniform', activation = "tanh")
out_densor2 = Dense(100, kernel_initializer='random_uniform' , activation = "tanh")
out_densor3 = Dense(3, kernel_initializer='random_uniform' , activation = "tanh")


# In[14]:


def model(Tx, Ty, n_a, n_s, input_feat_size, output_feat_size, char_len, num_char):
    X = Input(shape=(Tx, input_feat_size))
    C = Input(shape=(char_len, num_char))   # one hot encoded vector
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    

    outputs = []
    #a = Bidirectional(LSTM(n_a, return_sequences=True),input_shape=(Tx, input_feat_size))(X)
    a = Bidirectional(LSTM(n_a,activation='tanh',recurrent_dropout=0.2,return_sequences=True),input_shape=(Tx, input_feat_size))(X)
    for t in range(Ty):
        context = one_step_attention(a, s, C)
        s, _, c = post_activation_LSTM_cell(context,initial_state= [s, c])
        #out = mix_model(s)
        #out = output_layer(s)
        output_l1 =  out_densor1(s) 
        output_l1 = BatchNormalization()(output_l1)
        
        output_l2 = out_densor2(output_l1) 
        output_l2 = BatchNormalization()(output_l2)
        
        out = out_densor3(output_l2)
        outputs.append(out)    
    model = Model(inputs=[X,C,s0,c0], outputs=outputs)
    return model


# In[15]:



#char_len = total number of characters in input text C
# num_char = number of possible characters
model = model(Tx, Ty, n_a, n_s, input_feat_size, input_feat_size,char_len, num_char)


# In[22]:


model.summary()


# In[23]:


#opt = Adam(lr=0.005, decay=0.01, beta_1=0.9, beta_2=0.999)
#model.compile(optimizer=opt,
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

#model.compile(loss=mdn.get_mixture_loss_func(output_dim, n_mix), optimizer=keras.optimizers.Adam())


# In[16]:


opt = Adam(lr=0.005, decay=0.01, beta_1=0.9, beta_2=0.999,clipvalue=100)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])


# In[17]:


m = Xoh.shape[0]  # no of examples we have for training
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


# In[32]:


#filepath="AttentionModel-NN-writing-gen-weights.{epoch:02d}-{val_loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]


# In[ ]:


#model.fit([Xoh,C, s0, c0], outputs, validation_split=0.33, epochs=20, batch_size=32,callbacks=callbacks_list)
#print('training completed')
#model.save('AttentionModel-NN-writing-gen-weights_batch32_epoch20.h5')  # creates a HDF5 file 'my_model.h5'


# In[18]:


checkpoint_path = "results/AttentionModel-NN-wrting-generate/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True,verbose=1)


#model.fit(train_images, train_labels,  epochs = 10, 
#          validation_data = (test_images,test_labels),
#          callbacks = [cp_callback])  # pass callback to training


# In[19]:
if os.path.isfile(checkpoint_path) == True:
	print('found pre-weights')
	model.load_weights(checkpoint_path)
else:
	 print('no pre-trained weight found')
model.fit([Xoh,C, s0, c0], outputs, validation_split=0.33,
          epochs=20, batch_size=32,
          callbacks=[cp_callback])


# In[ ]:




