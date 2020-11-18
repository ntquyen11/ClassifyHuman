import numpy as np
import cv2 
import time
import matplotlib.pyplot as plt 
import math
import glob
import os
from  keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten,Activation 
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger



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

# creat model
model=Sequential()

# 1st convolution layer
model.add(Conv2D(24,(11,11),input_shape=(64,64,1),strides=(4,4),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

# 2nd convolution layer
model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

# 3rd convolution layer
model.add(Conv2D(96,kernel_size=(3,3),strides=(1,1),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 4th convolution layer
model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5th convolution layer
model.add(Conv2D(24,kernel_size=(3,3),strides=(1,1),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

# passing it to a fully connected layer
model.add(Flatten())

# 1st fully connected layer
model.add(Dense(1024,input_shape=(64,64,1,)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 2nd fully connecte('relu'))

model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 3rd fully connected layer
model.add(Dense(250))
model.add(BatchNormalization())
model.add(Activation('relu'))


# output layer
model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

# model.summary
# model.summary()
# losss
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# # fit data
from keras.preprocessing.image import ImageDataGenerator
train_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=True,zoom_range=.1)
test_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=True,zoom_range=.1)
val_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=True,zoom_range=.1)

train_generator.fit(X_train)
val_generator.fit(X_val)
test_generator.fit(X_test)

from keras.callbacks import ReduceLROnPlateau

#Model Checkpoint
checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'checkpoints', 'CNN' + '-' + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

# 
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', 'CNN' + '-' + 'training-' + \
        str(timestamp) + '.log'))

# TensorBoard
tb = TensorBoard(log_dir=os.path.join('data', 'logs', 'CNN'))
# Early stopping
early_stopper = EarlyStopping(patience=10)

batch_size=16
epochs=100
learn_rate=0.001
H=model.fit_generator(train_generator.flow(X_train, Y_train, batch_size=batch_size), epochs = epochs, steps_per_epoch = X_train.shape[0]//batch_size, validation_data = val_generator.flow(X_val, Y_val, batch_size=batch_size), validation_steps = 250, callbacks=[tb, early_stopper,csv_logger, checkpointer], verbose=1)

# evaluate
score=model.evaluate(X_test,Y_test,verbose=0)
model.save('classifyHuman.h5')