import numpy as np 
import keras
import glob
import math
import os
import time
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense,concatenate
from keras.layers import Conv2D,GlobalAveragePooling2D,AveragePooling2D, MaxPool2D
from keras.callbacks import EarlyStopping,TensorBoard,ModelCheckpoint,CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle
#read folder

Human=[cv2.resize(cv2.imread(file),(64,64)) for file in glob.glob('Human/*jpg')]
NonHuman=[cv2.resize(cv2.imread(file),(64,64)) for file in glob.glob('Non-Human/*jpg')]

imageHuman=[cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in Human]
imageNonHuman=[cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in NonHuman]
# numbers of image
numbersOfHuman=len(imageHuman)
numbersOfNonHuman=len(imageNonHuman)

# devide data
x_train=imageHuman[0:math.ceil(0.5*numbersOfHuman)]+imageNonHuman[0:math.ceil(0.5*numbersOfNonHuman)]
x_val=imageHuman[math.ceil(0.5*numbersOfHuman):math.ceil(0.8*numbersOfHuman)]+imageNonHuman[math.ceil(0.5*numbersOfNonHuman):math.ceil(0.8*numbersOfNonHuman)]
x_test=imageHuman[math.ceil(0.8*numbersOfHuman):numbersOfHuman]+imageNonHuman[math.ceil(0.8*numbersOfNonHuman):numbersOfNonHuman]


# reshape data
X_train=np.reshape(x_train,(len(x_train),64,64,1))
X_val=np.reshape(x_val,(len(x_val),64,64,1))
X_test=np.reshape(x_test,(len(x_test),64,64,1))

# label
numOfHumanTrain=math.ceil(0.5*numbersOfHuman)
numOfNonHumanTrain=math.ceil(0.5*numbersOfNonHuman)
Y_train=np.concatenate((np.ones((numOfHumanTrain,)),np.zeros((numOfNonHumanTrain),)),axis=0)

numOfHumanVal=math.ceil(0.3*numbersOfHuman)-1
numOfNonHumanVal=math.ceil(0.3*numbersOfNonHuman)-1
Y_val=np.concatenate((np.ones((numOfHumanVal,)),np.zeros((numOfNonHumanVal),)),axis=0)

numOfHumanTest=math.ceil(0.2*numbersOfHuman)-1
numOfNonHumanTest=math.ceil(0.2*numbersOfNonHuman)-1
Y_test=np.concatenate((np.ones((numOfHumanTest,)),np.zeros((numOfNonHumanTest),)),axis=0)

# Shuffle data
X_train,Y_train=shuffle(X_train,Y_train,random_state=1)
X_val,Y_val=shuffle(X_val,Y_val,random_state=1)
X_test,Y_test=shuffle(X_test,Y_test,random_state=1)
# Creat model
# function creates inception model
def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

# create initial input
input_layer = keras.Input(shape=(64, 64, 1))

# 1st layer
x = Conv2D(16, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

# 2nd layer
x = Conv2D(16, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(48, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)
# 3r layer
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

# 4th layer
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

# 5th layer
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
# model.summary()

# checkpointer
Checkpointer=ModelCheckpoint(
        filepath=os.path.join('data','checkpoint','CNN'+'-'+\
            '.{epoch:0.3d}-{val_loss:05d}.hd5f'),
        verbose=1,
        save_best_only=True
)
# csv_logger
timestamp=time.time()
csv_logger=CSVLogger(os.path.join('data','logs','CNN'+'training'+\
    str(timestamp)+'.logs'))

# tensorboard
tb=TensorBoard(log_dir=os.path.join('data','logs','CNN'))

# earlystopping
early_stopper=EarlyStopping(patience=10)

epochs=100
batch_Size=16

# Data Augmentation
train_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=Tru,zoom_range=.1)
val_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=Tru,zoom_range=.1)
test_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=Tru,zoom_range=.1)

train_generator.fit(X_train)
test_generator.fit(X_test)
val_generator.fit(X_val)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
H=model.fit_generator(train_generator.flow(X_train, Y_train, batch_size=batch_size), epochs = epochs, validation_data = val_generator.flow(X_val, Y_val, batch_size=batch_size),  callbacks=[tb, early_stopper,csv_logger, checkpointer], verbose=1)

score=model.evaluate(X_test,Y_test,verbose=0)
model.save('InceptionV1.h5')