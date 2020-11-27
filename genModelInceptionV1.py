import numpy as np 
import keras

from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense,concatenate
from keras.layers import Conv2D,GlobalAveragePooling2D,AveragePooling2D, MaxPool2D

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

input_layer = keras.Input(shape=(224, 224, 3))
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

x = Conv2D(16, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(16, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(48, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=16,
                     filters_3x3_reduce=24,
                     filters_3x3=32,
                     filters_5x5_reduce=4,
                     filters_5x5=8,
                     filters_pool_proj=8,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=32,
                     filters_3x3_reduce=32,
                     filters_3x3=48,
                     filters_5x5_reduce=8,
                     filters_5x5=24,
                     filters_pool_proj=16,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=48,
                     filters_3x3_reduce=16,
                     filters_3x3=52,
                     filters_5x5_reduce=4,
                     filters_5x5=12,
                     filters_pool_proj=16,
                     name='inception_4a')


x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(256, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(1, activation='sigmoid', name='auxilliary_output_1')(x1)

x = inception_module(x,
                     filters_1x1=40,
                     filters_3x3_reduce=103,
                     filters_3x3=56,
                     filters_5x5_reduce=4,
                     filters_5x5=16,
                     filters_pool_proj=16,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=32,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=4,
                     filters_5x5=16,
                     filters_pool_proj=16,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=28,
                     filters_3x3_reduce=36,
                     filters_3x3=72,
                     filters_5x5_reduce=8,
                     filters_5x5=16,
                     filters_pool_proj=16,
                     name='inception_4d')


x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(256, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(1, activation='sigmoid', name='auxilliary_output_2')(x2)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=40,
                     filters_3x3=80,
                     filters_5x5_reduce=8,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=40,
                     filters_3x3=80,
                     filters_5x5_reduce=8,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=96,
                     filters_3x3_reduce=48,
                     filters_3x3=96,
                     filters_5x5_reduce=12,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

x = Dropout(0.4)(x)

x = Dense(10, activation='softmax', name='output')(x)
model = keras.Model(input_layer, [x, x1, x2], name='inception_v1')
model.summary()
