# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 18:02:17 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.layers import Add, Conv2D, Input, Lambda, Activation, Conv2DTranspose
from keras.models import Model
from keras.layers import Conv3D, ZeroPadding3D, BatchNormalization, Multiply
from keras.layers import LeakyReLU, concatenate, Reshape, Softmax, MaxPool2D
from IPython.display import SVG
from keras.utils import plot_model

def SelfAttention(x, channels):
    x_shape = K.shape(x)
    # self attention
    f = Conv2D(filters=channels//8, kernel_size=1, strides=1, padding='same')(x)
    f = MaxPool2D(pool_size=(2, 2))(f)
    g = Conv2D(filters=channels//8, kernel_size=1, strides=1, padding='same')(x)
    # flatten hw * (1/8)c matmul (1/8)c * (1/4)hw -> hw * (1/4)hw
    shape = (K.shape(g)[0], -1, K.shape(g)[-1])
    g = Lambda(tf.reshape, arguments={'shape':shape})(g)
    shape = (K.shape(f)[0], -1, K.shape(f)[-1])
    f = Lambda(tf.reshape, arguments={'shape':shape})(f)
    s = Lambda(tf.matmul, arguments={'b':f, 'transpose_b':True})(g)
    # attention map
    beta = Softmax()(s) 
    h = Conv2D(filters=channels//2, kernel_size=1, strides=1, padding='same')(x)
    h = MaxPool2D(pool_size=(2, 2))(h)
    shape = (K.shape(h)[0], -1, K.shape(h)[-1])
    h = Lambda(tf.reshape, arguments={'shape':shape})(h)
    # hw * (1/4)hw matmul (1/4)hw * (1/2)c -> hw * (1/2)c
    o = Lambda(tf.matmul, arguments={'b':h, 'transpose_b':False})(beta)
    # gamma
    gamma = K.variable(0.0)
    shape = (x_shape[0], x_shape[1], x_shape[2], channels//2)
    o = Lambda(tf.reshape, arguments={'shape':shape})(o)
    # xch * scale ** 2
    o = Conv2D(filters=channels, kernel_size=1, strides=1, padding='same')(o)
    o = Lambda(tf.multiply, arguments={'y':gamma})(o)
    x = Add()([o, x])
    return x
    
    
