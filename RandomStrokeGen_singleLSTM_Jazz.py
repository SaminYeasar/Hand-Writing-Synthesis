
# coding: utf-8

# # Using the Jazz model from coursera

# In[1]:


import tensorflow as tf
import numpy as np
import mdn
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras


# In[2]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[3]:


Tx = 150   #output data length
no_examples = len(strokes)  #no of examples


# In[4]:


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
new_strokes = np.array(list(map(lambda x: tranc_stroke(x, Tx+1), strokes)))


# In[5]:


Xoh = new_strokes[:,:Tx,:]
Yoh = new_strokes[:,1:Tx+1,:]


# In[6]:


input_feat_size = Xoh.shape[2]
output_feat_size = Yoh.shape[2]


# In[7]:


print('shape of input data:', Xoh.shape)
print('number of training examples:', Xoh.shape[0])
print('Tx (length of sequence):', Xoh.shape[1])
print('Number of Input Features:', input_feat_size)
print('Shape of output data:', Yoh.shape)


# In[8]:


n_a = 256
output_dim = 3
n_mix = 10
reshapor = Reshape((1, input_feat_size))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
#densor = Dense(n_values, activation='softmax')     # Used in Step 2.D
mixture_model = mdn.MDN(output_dim, n_mix)


# In[9]:


def stroke_gen(Tx, n_a, input_feat_size):
    # Define the input of your model with a shape 
    X = Input(shape=(Tx, input_feat_size))  # shape=(?, Tx, features space ) = (? , 300 , 3)
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')     # shape=(?, n_a) = (? , 256)
    c0 = Input(shape=(n_a,), name='c0')     # shape=(?, n_a) = (? , 256)
    a = a0
    c = c0
    
    ### START CODE HERE ### 
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda x: X[:,t,:])(X)                           # shape=(?, feature_space) = (? , 3)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)                                             # shape=(?, 1, feature_space) = (? ,1, 3)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x,initial_state=[a,c])   
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        #out = densor(a)
        out = mixture_model (a)             # shape= (?, 70) for 10 mixtures and intended output features 3
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs=[X,a0,c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return model


# In[10]:


model = stroke_gen(Tx = Tx , n_a = n_a , input_feat_size = input_feat_size)


# In[12]:


#opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(loss=mdn.get_mixture_loss_func(output_dim, n_mix), optimizer=keras.optimizers.Adam())


# In[13]:


model.summary()


# In[14]:


a0 = np.zeros((no_examples, n_a))
c0 = np.zeros((no_examples, n_a))
outputs = list(Yoh.swapaxes(0,1))


# In[1]:


EPOCHS = 10
model.fit([Xoh, a0, c0], outputs, batch_size = 32, epochs=EPOCHS)
print('training has completed')


# In[ ]:


print('saving the model for {} number of epochs'.format(EPOCHS))
model.save('Random-Stroke2Stroke_{}.hdf5'.format(EPOCHS))

