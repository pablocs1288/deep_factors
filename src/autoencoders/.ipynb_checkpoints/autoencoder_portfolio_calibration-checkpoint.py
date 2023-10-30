import copy

import pandas as pd
import numpy as np

import src.utils.returns_preprocessing as preproc

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from sklearn.preprocessing import StandardScaler  

from tensorflow.python.ops import math_ops

## avaliable custom loss functions
def custom_matmult_rmse_loss(y_true, y_pred, input_tensor):
    # this optimization function is trouble as works with portfolio returns and not log??
    # normalize this objective function to emulate a linear encoder such as the PCA
    
    # normalizing weights the same way as with PCA
    neg_mask = K.cast(input_tensor < 0, K.floatx())
    pos_mask = K.cast(input_tensor >= 0, K.floatx())
    neg_sum = K.sum(neg_mask * input_tensor)
    pos_sum = K.sum(pos_mask * input_tensor)
    input_tensor = K.switch(input_tensor >= 0, input_tensor / pos_sum, input_tensor / neg_sum)
    
    # calculating the returns
    y_pred_dot = K.dot(y_pred, K.transpose(input_tensor)) 
    loss = K.sqrt(K.mean(K.square(y_pred_dot - y_true))) 
    return loss


### training and fit parts!
def autoencoder_portfolio_calibration_fit(x, y_target, network_params):
    
    # input autoencoder params
    encoding_dim = network_params['encoding_dim']
    l2_penalty = network_params['l2_penalty']
    optimizer =  network_params['optimizer']
    loss = network_params['loss']
    epochs = network_params['epochs']
    batch_size = network_params['batch_size']
    shuffle = network_params['shuffle']
    hidden_layers_activation = network_params['hidden_layers_activation']
    output_activation = network_params['output_activation']
    custom_loss_function = network_params['custom_loss_function']
    batch_normalization = network_params['batch_normalization']
    scale_data = network_params['scale_data']
    verbose = network_params['verbose']
    
    target = Input(shape=(1,)) # target tensor -> y_target
    input_tensor = Input(shape=(x.shape[1],))
    
    encoded = Dense(encoding_dim, activation=hidden_layers_activation, kernel_regularizer=regularizers.l2(l2_penalty))(input_tensor)
    if batch_normalization:
        encoded = BatchNormalization()(encoded)
        
    decoded = Dense(x.shape[1], activation= output_activation, kernel_regularizer=regularizers.l2(l2_penalty))(encoded)
    #if batch_normalization: # not sure if this goes here as it is a softmax layer
    #    encoded = BatchNormalization()(encoded)
    
    if custom_loss_function is None:
        custom_loss_function = custom_matmult_rmse_loss
 
    # construct and compile deep learning routine
    deep_learner = Model([input_tensor, target], decoded)
    deep_learner.add_loss(custom_loss_function(target, decoded, input_tensor))
    deep_learner.compile(optimizer=optimizer, loss = None, run_eagerly=True)
    
    if scale_data:
        ss = StandardScaler() # Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
        ss.fit(x)
        x = ss.transform(x)
    
    deep_learner.fit([x, y_target] , None, shuffle=shuffle, epochs=epochs, batch_size = batch_size, verbose = verbose)
    
    final_model = Model(deep_learner.input[0], deep_learner.output) # this make inferences compatible
    return final_model



    
def portfolio_returns_predict(x, model, return_log = True):
    "Get the models weights to estimate returns"
    
    # weights
    normalized_weights = copy.deepcopy(model.predict(x))
    overall_normalized_weights = []
    for i in range(len(normalized_weights[0])):
        overall_normalized_weights.append(np.mean(normalized_weights[:,i]))
    # receive x as log-returns, so we transform it to simple returns to apply the dot product
    x = x.apply(lambda x: preproc.log_to_simple_returns(x))
    x = np.array(x)
    returns = []
    for i in range(np.shape(x)[0]):
        returns.append(np.dot(overall_normalized_weights, np.transpose(x[i])))
    
    if return_log:
        return preproc.simple_to_log_returns(pd.Series(returns)), overall_normalized_weights
    
    return returns, overall_normalized_weights


def two_norm_diff(y, y_hat):
    return np.linalg.norm((y_hat - y))



###############################################
### Short version!!
def autoencoder_short_portfolio_calibration_fit(x, y_target, network_params):
    
    # input autoencoder params
    encoding_dim = network_params['encoding_dim']
    l2_penalty = network_params['l2_penalty']
    optimizer =  network_params['optimizer']
    loss = network_params['loss']
    epochs = network_params['epochs']
    batch_size = network_params['batch_size']
    shuffle = network_params['shuffle'] 
    hidden_layers_activation = network_params['hidden_layers_activation']


    input_tensor = Input(shape=(x.shape[1],))
    encoded = Dense(encoding_dim, activation=hidden_layers_activation, kernel_regularizer=regularizers.l2(l2_penalty))(input_tensor)
    decoded = Dense(x.shape[1], activation= 'tanh', kernel_regularizer=regularizers.l2(l2_penalty))(encoded)
    return_ = Dense(1, activation= 'linear', kernel_regularizer=regularizers.l2(l2_penalty))(decoded)
    
    deep_learner = Model(input_tensor, return_)
    deep_learner.compile(optimizer='sgd', loss='mean_squared_error')
    
    ss = StandardScaler()# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
    ss.fit(x)
    x = ss.transform(x)
    
    deep_learner.fit(x, y_target, shuffle=shuffle, epochs=epochs, batch_size = batch_size)    # fit the model
    return deep_learner



def short_portfolio_predict(x, model):
    weights = K.eval(model.layers[2].weights[1])
    returns = x.apply(lambda row: np.dot(weights,  np.transpose(row)), axis = 1)
    return np.array(returns)


#weights = K.eval(model.layers[2].weights[1]) # is not a single weight
#weights = weights / len(weights)
#returns = x.apply(lambda row: np.dot(weights,  np.transpose(row)), axis = 1)