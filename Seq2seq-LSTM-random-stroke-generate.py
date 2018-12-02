#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import keras
import mdn
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda , Reshape
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint

#from faker import Faker
import random
#from tqdm import tqdm
#from babel.dates import format_date
#from at_nmt_utils import *
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys 
import os
sys.path.insert(0,'..')
#from utils import plot_stroke


# In[3]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[4]:


Tx=Ty=300
m = len(strokes)


# In[5]:


def pad_stroke(stroke,Ty):
	npads = Ty - stroke.shape[0]
	padded_stroke = np.vstack ([ stroke,np.zeros((_npads,3)) ])
	#padded_strokes.shape
	return padded_stroke

def tranc_stroke(stroke, Ty):
	#input_stroke = []
	#output_stroke = []
	if stroke.shape[0] >= Ty:
		stroke = stroke[:Ty,]
	elif stroke.shape[0] < Ty:
		stroke = pad_stroke(stroke,Ty)
	return stroke

new_strokes = np.array(list(map(lambda x: tranc_stroke(x, Ty+1), strokes)))


# In[6]:


Xoh = new_strokes[:,:Ty,:]           # input_strokes
Yoh = new_strokes[:,1:Ty+1,:]       # output_strokes


# In[7]:


#print('shape of X:', Xoh.shape)
#print('number of training examples:', Xoh.shape[0])
#print('Tx (length of sequence):', Xoh.shape[1])
#print('Shape of Y (m*Ty*ny):', Yoh.shape)     #have not swapt value            


# In[8]:


n_a = 256
n_x = n_y = 3


# In[9]:

output_dim = 3 
n_mix = 10
#reshapor = Reshape((1, 78))                        
reshapor = Reshape((1, n_y))
LSTM_cell = LSTM(n_a,recurrent_dropout=0.2,return_state = True)      
#densor = Dense(n_y, activation='softmax')    
#densor = Dense(n_y)

mix_model = mdn.MDN(output_dim, n_mix)

# In[10]:


def stroke_learn_model(Tx, n_a, n_x):
	# Define the input of your model with a shape
	X = Input(shape=(Tx, n_x))
	# Define a0, initial hidden state for the decoder LSTM
	a0 = Input(shape=(n_a,), name='a0')
	c0 = Input(shape=(n_a,), name='c0')
	a = a0
	c = c0
	outputs = []
	for t in range(Tx):
		# select the "t"th time step vector from X.
		x = Lambda(lambda x: X[:,t,:])(X)
		# reshape x to be (1, n_values)
		x = reshapor(x)
		# one step of the LSTM_cell
		a, _, c = LSTM_cell(x,initial_state=[a,c])
		# densor to the hidden state output of LSTM_Cell
		out = mix_model(a)
	#out = mix_model(a)
		outputs.append(out)
	# Create model instance
	model = Model(inputs=[X,a0,c0], outputs=outputs)
	return model


# In[11]:

model = stroke_learn_model(Tx = Tx , n_a = n_a, n_x = n_x)


# In[12]:


opt = Adam(clipvalue=100)
#model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

model.compile(loss=mdn.get_mixture_loss_func(output_dim,n_mix), optimizer=opt)

# In[13]:


a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
outputs = list(Yoh.swapaxes(0,1))


# In[14]:



#outputs[0].shape


# In[5]:


filepath="seq2seq-MDN-300timlen-weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[15]:


#model.fit([Xoh, a0, c0], outputs, monitor='val_loss', epochs = 20, batch_size = 30,callbacks=callbacks_list)
model.fit([Xoh, a0, c0], outputs, validation_split=0.33, epochs=100, batch_size=32,callbacks=callbacks_list)

model.save('seq2seq_mdn_model.h5')
# In[19]:

"""
def stroke_inference_model(LSTM_cell, densor, n_x, n_a=50 , Ty=300 ):
	# Define the input of your model with a shape 
	x0 = Input(shape=(1, n_x))
	# Define a0, initial hidden state for the decoder LSTM
	a0 = Input(shape=(n_a,), name='a0')
	c0 = Input(shape=(n_a,), name='c0')
	a = a0
	c = c0
	x = x0
	outputs = []
	for t in range(Ty):
		#Perform one step of LSTM_cell (≈1 line)
		a, _, c = LSTM_cell(x,initial_state=[a,c])
		#print(a)
		# Dense layer to the hidden state output of the LSTM_cell 
		out = densor(a)
		#print(out)
		outputs.append(out)
		#x = Lambda(one_hot)(out)
		x = RepeatVector(1)(out)
		#print(x)
	# Create model instance with the correct "inputs" and "outputs" 
	inference_model = Model(inputs=[x0,a0,c0], outputs=outputs)
	return inference_model


# In[20]:


inference_model = stroke_inference_model(LSTM_cell, densor, n_x = n_x, n_a = n_a, Ty = Ty)


# In[21]:


x_initializer = np.zeros((1, 1, 3))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


# In[22]:


pred = inference_model.predict([x_initializer, a_initializer, c_initializer])


# In[57]:


gen_strokes = np.array(pred).swapaxes(0,1)
np.savetxt("seq2seq_gen_strokes.csv", result, delimiter=",")


# In[20]:



#def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
#                       c_initializer = c_initializer):
#    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
#    indices = np.argmax(np.array(pred),axis=2)
#    results = to_categorical(indices,num_classes=None)
#    return results, indices
#results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
"""

