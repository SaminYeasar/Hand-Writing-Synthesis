
# coding: utf-8

# In[1]:


import keras
from context import * # imports the MDN layer 
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
get_ipython().run_line_magic('matplotlib', 'inline')

# Only for GPU use:
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)


# ### First download and process the dataset.

# In[2]:


strokes = np.load('strokes.npy',encoding='bytes')


# In[3]:


Ty = 300
Tx = 300
m = len(strokes)


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

new_strokes = np.array(list(map(lambda x: tranc_stroke(x, Ty+1), strokes)))


# In[5]:


input_strokes = new_strokes[:,:Ty,:]
output_strokes = new_strokes[:,1:Ty+1,:]


# In[6]:


Xoh = input_strokes
Yoh = output_strokes


# In[7]:


Xoh.shape


# In[23]:


plt.plot(Xoh[0].T[1], Xoh[0].T[2])
plt.title("Raw values (diffs) for one training example")
plt.show()


# In[25]:


plt.plot(Xoh[0].T[1].cumsum(), -1 * Xoh[0].T[2].cumsum())
plt.title("Accumulated values for one training example")
plt.show()


# In[8]:


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


# In[9]:


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


# In[63]:


# Fit the model
EPOCHS = 100
history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[keras.callbacks.TerminateOnNaN()])


# In[21]:


model.save_weights('mdn_stroke2stroke_epoch{}.h5'.format(EPOCHS))  # creates a HDF5 file 'my_model.h5'


# In[ ]:


################################################



################################################

'''
# In[17]:


plt.figure()
plt.plot(history.history['loss'])
plt.show()


# ## Try out the model! Generate some Kanji!
# 
# We need to create a decoding model with batch size 1 and sequence length 1.

# In[10]:


# Decoding Model
# Same as training model except for dimension and mixtures.

decoder = keras.Sequential()
decoder.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(1,1,OUTPUT_DIMENSION), return_sequences=True, stateful=True))
decoder.add(keras.layers.LSTM(HIDDEN_UNITS, stateful=True))
decoder.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))
decoder.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())
decoder.summary()

decoder.load_weights('mdn_stroke2stroke_epoch1.h5') # load weights independently from file


# ## Generating drawings
# 
# - First need some helper functions to view the output.

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def zero_start_position():
    """A zeroed out start position with pen down"""
    out = np.zeros((1, 1, 3), dtype=np.float32)
    out[0, 0, 2] = 1 # set pen down.
    return out

def generate_sketch(model, start_pos, num_points=100):
     return None

def cutoff_stroke(x):
    return np.greater(x,0.5) * 1.0

def plot_sketch(sketch_array):
    """Plot a sketch quickly to see what it looks like."""
    sketch_df = pd.DataFrame({'x':sketch_array.T[1],'y':sketch_array.T[2],'z':sketch_array.T[0]})
    sketch_df.x = sketch_df.x.cumsum()
    sketch_df.y = -1 * sketch_df.y.cumsum()
    # Do the plot
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    #ax1.scatter(sketch_df.x,sketch_df.y,marker='o', c='r', alpha=1.0)
    # Need to do something with sketch_df.z
    ax1.plot(sketch_df.x,sketch_df.y,'r-')
    plt.show()


# ## SVG Drawing Function
# 
# Here's Hardmaru's Drawing Functions from _write-rnn-tensorflow_. Big hat tip to Hardmaru for this!
# 
# Here's the source: https://github.com/hardmaru/write-rnn-tensorflow/blob/master/utils.py
# 

# In[13]:


# Hardmaru's Drawing Functions from write-rnn-tensorflow
# Big hat tip
# Here's the source:
# https://github.com/hardmaru/write-rnn-tensorflow/blob/master/utils.py

import svgwrite
from IPython.display import SVG, display

def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)

def draw_strokes(data, factor=1, svg_filename='sample.svg'):
    min_x, max_x, min_y, max_y = get_bounds(data, factor)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    dwg = svgwrite.Drawing(svg_filename, size=dims)
    dwg.add(dwg.rect(insert=(0, 0), size=dims, fill='white'))

    lift_pen = 1

    abs_x = 25 - min_x
    abs_y = 25 - min_y
    p = "M%s,%s " % (abs_x, abs_y)

    command = "m"

    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        lift_pen = data[i, 2]
        p += command + str(x) + "," + str(y) + " "

    the_color = "black"
    stroke_width = 1

    dwg.add(dwg.path(p).stroke(the_color, stroke_width).fill("none"))

    dwg.save()
    display(SVG(dwg.tostring()))


# In[27]:


params[0].shape


# In[34]:


p


# In[35]:


p.reshape((3,))


# In[57]:


sketch2 = np.array(sketch2)
sketch2


# In[59]:


cutoff_stroke(sketch2.T[2])


# In[55]:


sketch


# In[14]:


# Predict a character and plot the result.
temperature = 2.5 # seems to work well with rather high temperature (2.5)

p = zero_start_position()
sketch = [p.reshape(3,)]

for i in range(400):
    params = decoder.predict(p.reshape(1,1,3))
    p = mdn.sample_from_output(params[0], OUTPUT_DIMENSION, NUMBER_MIXTURES, temp=temperature)
    sketch.append(p.reshape((3,)))
sketch2 = sketch
sketch = np.array(sketch)


decoder.reset_states()

sketch.T[2] = cutoff_stroke(sketch.T[2])
draw_strokes(sketch, factor=0.5)
#plot_sketch(sketch)


# In[15]:


temperature = 2.5 # seems to work well with rather high temperature (2.5)

p = zero_start_position()
sketch = [p.reshape(3,)]

for i in range(400):
    params = decoder.predict(p.reshape(1,1,3))
    p = mdn.sample_from_output(params[0], OUTPUT_DIMENSION, NUMBER_MIXTURES, temp=temperature)
    sketch.append(p.reshape((3,)))
sketch2 = sketch
sketch = np.array(sketch)


# In[17]:


sketch.shape


# In[18]:


plt.plot(sketch.T[1].cumsum(), -1 * sketch.T[2].cumsum())
plt.title("Accumulated values for one training example")
plt.show()
'''


