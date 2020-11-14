import numpy as np
import cv2 
from  keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten,Activation 
from keras.utils import np_utils
import matplotlib.pyplot as plt 
import glob
import os
import math


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
# convolution
model.add(Conv2D(64, (3, 3), activation='sigmoid', input_shape=(64,64,1)))
# convolution
model.add(Conv2D(32, (3, 3), activation='sigmoid'))
#maxpooling
model.add(MaxPooling2D(pool_size=(2,2)))
#flatten
model.add(Conv2D(32,(3,3),activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
#dense
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
# losss
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# fit data
H = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),batch_size=32, epochs=10, verbose=1)
numberEpoch=10
fig=plt.figure()
plt.plot(np.arange(0,numberEpoch),H.history['loss'],label='training loss')
plt.plot(np.arange(0,numberEpoch),H.history['val_loss'],label='validation loss')
plt.plot(np.arange(0,numberEpoch),H.history['accuracy'],label='accuracy')
plt.plot(np.arange(0,numberEpoch),H.history['val_accuracy'],label='validation accuracy')

plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')

# evaluate
score=model.evaluate(X_test,Y_test,verbose=0)
print(score)

plt.imshow(X_test[0].reshape(64,64), cmap='gray')
y_predict = model.predict(X_test[0].reshape(1,64,64,1))
print('Gia tri du doan: ', np.argmax(y_predict))
plt.show()