import numpy as np 
import keras
import glob
import cv2
import math
import os
import time
from keras.models import Sequential
from keras.layers import Dropout,MaxPooling2D,Flatten,Dense,concatenate
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
model=Sequential()
# Layer 1
model.add(Conv2D(8,(3,3),activation='relu',input_shape=(64,64,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
# Layer 2
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(Conv2D(16,(1,1),activation='relu',padding='same'))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
# Layer 3
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(Conv2D(32,(1,1),activation='relu',padding='same'))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
# Layer 4
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(1,1),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Conv2D(64,(1,1),activation='relu',padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='same'))
# Layer 5
model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,(1,1),activation='relu',padding='same'))
model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
model.add(Conv2D(128,(1,1),activation='relu',padding='same'))
model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
model.add(Conv2D(250,(1,1),activation='relu',padding='same'))
model.add(GlobalAveragePooling2D())
# Flatten
model.add(Flatten())

model.add(Dense(1,activation='sigmoid'))
# model.summary()

# checkpointer
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


epochs=100
batch_size=16

# Data Augmentation
train_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=True,zoom_range=.1)
val_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=True,zoom_range=.1)
test_generator=ImageDataGenerator(rotation_range=2,horizontal_flip=True,zoom_range=.1)

train_generator.fit(X_train)
test_generator.fit(X_test)
val_generator.fit(X_val)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
H=model.fit_generator(train_generator.flow(X_train, Y_train, batch_size=batch_size), epochs = epochs, steps_per_epoch = X_train.shape[0]//batch_size, validation_data = val_generator.flow(X_val, Y_val, batch_size=batch_size), validation_steps = 250, callbacks=[tb, early_stopper,csv_logger, checkpointer], verbose=1)

score=model.evaluate(X_test,Y_test,verbose=0)
model.save('Darknet19.h5')