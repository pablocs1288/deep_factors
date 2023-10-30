import copy

import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler  
from tensorflow.python.ops import math_ops



def autoencoder_market_projection_fit(x, network_params):
    
    # input autoencoder params
    encoding_dim = network_params['encoding_dim']
    l2_penalty = network_params['l2_penalty']
    optimizer =  network_params['optimizer']
    loss = network_params['loss']
    epochs = network_params['epochs']
    batch_size = network_params['batch_size']
    shuffle = network_params['shuffle']
    hidden_layers_activation = network_params['hidden_layers_activation'] # relu
    output_activation = network_params['output_activation'] # Columbia -> tanh with a batch normalization layer in sequence 
    batch_normalization = network_params['batch_normalization']
    scale_data = network_params['scale_data']
    verbose = network_params['verbose']
    
    # 1st layer - input
    input_img = Input(shape=(x.shape[1],), )
    
    # 2nd layer - encoding
    encoded = Dense(round(x.shape[1]/2), activation=hidden_layers_activation, kernel_regularizer=regularizers.l2(l2_penalty))(input_img)
    if batch_normalization:
        encoded = BatchNormalization()(encoded)
    # 3rd layer - encoding
    encoded = Dense(encoding_dim, activation=hidden_layers_activation, kernel_regularizer=regularizers.l2(l2_penalty))(encoded)
    if batch_normalization:
        encoded = BatchNormalization()(encoded)
    # 4th layer - decoding
    decoded = Dense(round(x.shape[1]/2), activation = hidden_layers_activation,kernel_regularizer = regularizers.l2(l2_penalty))(encoded)
    #if batch_normalization:
    #    decoded = BatchNormalization()(decoded)
    # 5th layer - output    
    decoded = Dense(x.shape[1], activation= output_activation, kernel_regularizer=regularizers.l2(l2_penalty))(decoded)
    #if batch_normalization:
    #    decoded = BatchNormalization()(decoded)
    
   
    model = Model(input_img, decoded)
    model.compile(optimizer=optimizer, loss=loss)
    
    if scale_data:
        ss = StandardScaler() # Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
        ss.fit(x)
        x = ss.transform(x) 
    
    model.fit(x, x, shuffle=shuffle, epochs=epochs, batch_size = batch_size, verbose=verbose)
    
    return model

def _predict(x, model):
    reconstruct = model.predict(x)
    return reconstruct


def communal_information(x, model):
    reconstruction = _predict(x, model)
    communal_information = []
    for i in range(0, x.shape[1]):
        diff = np.linalg.norm((x.iloc[:,i] - reconstruction[:,i])) # 2 norm difference
        communal_information.append(float(diff))
    ranking = np.array(communal_information).argsort()
    info = []
    for stock_index in ranking:
        info.append((stock_index, communal_information[stock_index], x.iloc[:,stock_index].name))
        
    communal_info = pd.DataFrame(info)
    communal_info.columns = ['index', 'rmse', 'ticker']
    return communal_info