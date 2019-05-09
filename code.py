
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# ### All the parameters required for the network can be tweaked below

# In[ ]:


### Hyperparamets babyyy, change accordingly
Channel_uses = 4
Epochs = 10000
Noise_variance = 1e-4
Pert_variance = 1e-4
Batch_size = 1024
# init_losses_vec = np.ones(128)


# ### TX loss function 
# The policy function for the transmitter is similar to that of a cross-entropy between the noisy loss feedback (l) and the J(w,$\theta$) function value
# 
# Loss = -$\sum_{i=1}^n$($l_i$ * J($w_i$,$\theta$))

# In[ ]:


def tx_loss(y_true, y_pred): 
#     loss = - y_true*keras.backend.log(y_pred)

    return -y_true*y_pred
     


# ### Perturbation
# 
# After we get the output from the transmitter network, we then add the perturbation matrix as mentioned in the paper. We write a  function for this purpose and then make a custom layer like functionality using the `keras.layers.lambda` functionality

# In[ ]:


def perturbation(d):
    W = tf.keras.backend.random_normal(shape = (2*Channel_uses,),
    mean=0.0,stddev=Pert_variance**0.5,dtype=None,seed=None)
    d = ((1-Pert_variance)**0.5)*d + W
    return d


# ###  Tx model

# In[ ]:


def blah(y):
    w = y[:,y.shape[-1]//2:] - y[:,:y.shape[-1]//2]
    print('hehehe', w.shape)
    t = -keras.backend.sum(w*w)
#     t = keras.backend.exp(-t/Pert_variance**2)/(np.pi*Pert_variance**2)**Channel_uses
    return t


# In[ ]:


# tx layers
tx_input = keras.layers.Input((1,), name='tx_input')
x = keras.layers.BatchNormalization()(tx_input)
x = keras.layers.Dense(units=10*Channel_uses, activation='elu', name='tx_10')(x)
x = keras.layers.Dense(units=2*Channel_uses, activation='elu', name='tx_out')(x)
xp = keras.layers.Lambda(perturbation, name='Xp')(x)
concat = keras.layers.concatenate([x,xp], axis=1)
policy = keras.layers.Lambda(blah)(concat)
print(concat.shape)


# We define the entire graph but for simplicity sake, we also define a sub-model for getting the internediate layer outputs.
# 
# To be even more precise, we add perturbation after we get the Tx layer output. So, to get the perturbation matrix out, we define a full model and another proxy model (which shares weights with the full model) which return without perturbation matrix effects.
# 
# We then subtract these two layers to get the value of W (perturbation matrix) for a given batch/sample
# 
# (Note that we had to take this roundabout method to get W because Keras can't return two tensors for a said layer)

# In[ ]:


tx_model = keras.models.Model(inputs=tx_input, outputs=concat)


# In[ ]:


tx_model.summary()


# In[ ]:


pert_model = keras.models.Model(inputs=tx_input, outputs=policy)


# In[ ]:


pert_model.compile(loss=tx_loss, optimizer='sgd')
pert_model.summary()


# ### Rx model
# 
# In the said RX model, we are taking the Perturbed input, adding channel effects and then passing on for estimation. 

# In[ ]:


rx_input = keras.layers.Input((2*Channel_uses,), name='rx_input')
# channel layer
y = keras.layers.Lambda(lambda x: x+keras.backend.random_normal(
        shape = (2*Channel_uses,), mean=0.0, stddev=Noise_variance**0.5), name='channel')(rx_input)

y = keras.layers.Dense(2*Channel_uses, activation='relu', name='rx_2')(y)
y = keras.layers.Dense(10*Channel_uses, activation='relu', name='rx_10')(y)
pred = keras.layers.Dense(1, activation='relu', name='rx_output')(y)


# In[ ]:


rx_model = keras.models.Model(inputs=rx_input, outputs=pred)
rx_model.summary()


# In[ ]:


rx_model.compile(loss='mse', optimizer='sgd')


# In[ ]:


data_numbers = np.random.uniform(0,1,(Batch_size,))
y = tx_model.predict(data_numbers)
print(y.shape)
XP = y[:,y.shape[-1]//2:]
estimated_vector  = np.squeeze(rx_model.predict(XP))
print(estimated_vector.shape, data_numbers.shape)


# In[ ]:


l = (estimated_vector-data_numbers)**2
l_hat = rx_model.predict(tx_model.predict(data_numbers)[:,2*Channel_uses:])


# In[ ]:


pert_model.fit(data_numbers, l_hat, batch_size=Batch_size, epochs=1)


# ### Training
# 
# Training this entire network is done as discussed in the paper -
# 1. Generate a batch of numbers sampled from Uniform random variable from [0,1]
# 2. Pass the numbers through Tx and then Rx
# 3. Get a loss vectors for the said batch of numbers
# 4. Train the Rx network on MSE with SGD
# 5. Feed back the loss vector to Tx using the same pair of Tx and Rx to incorporate noise into the loss vector
# 6. Use policy function, the loss vector and train the Tx for the same batch of numbers
# 7. Back to step 1

# In[ ]:


for i in range(Epochs):
    data_numbers = np.random.uniform(0,1,(Batch_size,))
    y = tx_model.predict(data_numbers)
    XP = y[:,y.shape[-1]//2:]
    estimated_vector= np.squeeze(rx_model.predict(XP))
    l = (estimated_vector-data_numbers)**2
    l_hat = rx_model.predict(tx_model.predict(data_numbers)[:,2*Channel_uses:])
    pert_model.fit(data_numbers, l_hat, batch_size=Batch_size, epochs=1, verbose=0)
#     print("Tx-done")
    rx_model.fit(XP, data_numbers, batch_size=Batch_size, epochs=1)
#     print("Rx-done")


# ### Prediction phase
# Note that the network is predicting numbers with a quite low error margin (+- 1e-2)
# This is in case of continous numbers 
# Say we feed numbers sampled from PAM (discrete numbers) and set our prediction rules as a floor or ceiling function, this model easily achieves 95% accuracy
# 
# This is all achieved even though there is a noisy feedback of losses from Tx to Rx

# In[ ]:


data_numbers = np.random.uniform(0,1,(10,))
y = tx_model.predict(data_numbers)
XP = y[:,y.shape[-1]//2:]
estimated_vector= np.squeeze(rx_model.predict(XP))
print(data_numbers)
print(estimated_vector)
# l = (estimated_vector-data_numbers)**2
# l_hat = rx_model.predict(tx_model.predict(data_numbers)[:,2*Channel_uses:])
# pert_model.fit(data_numbers, l_hat, batch_size=Batch_size, epochs=1)
# print("Tx-done")
# rx_model.fit(XP, data_numbers, batch_size=Batch_size, epochs=1)
# print("Rx-done")


# ### Post implimentation tid-bits
# 
# Please note that we had to make some chnages from the original discussed implimentation and theory to attain some numerical stability and to dodge NaN losses
# 
# 1. In the J(w,$\theta$) function, we have a part involving exp(|w|) and some constants. Where as the loss involved $L_i$ * log(J(w,$\theta$)). This causes numerical instability in case the J function goes negative or is very very small due to exp() and then log. To prevent this, we ignored the constants (as they dont affect gradient terms while differentiating) and removed the exp() and log() terms all-together
# 
# 2. Author assumed two pairs of Tx-Rx with shared weights. We used one for both purposes as it is symmetric
