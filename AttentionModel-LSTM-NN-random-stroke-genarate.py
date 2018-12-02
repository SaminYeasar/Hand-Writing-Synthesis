#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda ,Dropout,BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


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


Ty = 300   #input data length
Tx = 300   #output data length
no_examples = len(strokes)  #no of examples


# In[5]:


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


# In[6]:


Xoh = new_strokes[:,:Ty,:]
Yoh = new_strokes[:,1:Ty+1,:]


# In[7]:


input_feat_size = Xoh.shape[2]
output_feat_size = Yoh.shape[2]


# In[8]:


#print(Xoh.shape)


# In[9]:


def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s


# In[10]:


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, kernel_initializer='random_uniform', activation = "tanh")
densor2 = Dense(1, kernel_initializer='random_uniform' , activation = "tanh")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)


# In[11]:


def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator ([s_prev,a])
    
    e =  densor1(concat) 
    e = BatchNormalization()(e)
    
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas,a])
    return context


# In[12]:


n_a = 128
n_s = 256
post_activation_LSTM_cell = LSTM(n_s,activation='tanh',recurrent_dropout=0.2,return_state = True)
out_densor1 = Dense(100, kernel_initializer='random_uniform', activation = "tanh")
out_densor2 = Dense(100, kernel_initializer='random_uniform' , activation = "tanh")
out_densor3 = Dense(3, kernel_initializer='random_uniform' , activation = "tanh")
output_layer = Dense(3)


# In[13]:


def model(Tx, Ty, n_a, n_s, input_feat_size, output_feat_size):
    X = Input(shape=(Tx, output_feat_size))
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


# In[14]:


model = model(Tx, Ty, n_a, n_s, input_feat_size, output_feat_size)


# In[ ]:


model.summary()


# In[ ]:


# All parameter gradients will be clipped to
# a maximum value of 100 and
# a minimum value of -100.
opt = Adam(lr=0.005, decay=0.01, beta_1=0.9, beta_2=0.999,clipvalue=100)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['accuracy'])


# In[ ]:


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


# In[ ]:


filepath="AttentionModel-LSTM-NN-strok-gen-weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[ ]:


model.fit([Xoh, s0, c0], outputs, validation_split=0.33, epochs=100, batch_size=32,callbacks=callbacks_list)
print('training completed')
model.save('AttentionModel-LSTM-NN-strok-gen.h5')


# In[ ]:


#model.load_weights("Attention-LSTM-100.Maxlen-400.weights.01-1119.31.hdf5")


# In[ ]:


#source = np.zeros((1,400,3))


# In[ ]:


"""
result = []
for i in range(100):    
    prediction = model.predict([source, s0, c0])
    source = np.array(prediction).swapaxes(0,1)
    result.append(source)
    
"""
#np.savetxt("gen_strokes.csv", result, delimiter=",")


# In[ ]:


#result[-1]


# In[ ]:


#predicted_strokes = np.array(prediction).swapaxes(0,1)


# In[ ]:


#predicted_strokes


# In[ ]:


#result[-1][0][1:2,0] = 1


# In[ ]:


#result[-1][0]


# In[ ]:


#plot_stroke(result[-1])


# In[ ]:


#plt.plot(-1 *result[-1][0][:,1])


# In[ ]:




