import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#put all picture address and reaction to a list
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)





#split picture and record to different list
def get_img(samples):
    images = []
    reactions = []
    for sample in samples[1:]:
        if sample[3]=='steering':
            continue
        reaction = list(map(float,sample[3:]))
        if abs(reaction[0])>0.08: #set up a penulty if turning larger than that I will append a flipped picture to data set
            image = cv2.imread(sample[0])
            images.append(image)
            images.append(cv2.flip(image,1))
            reactions.append(reaction[0])
            reactions.append(reaction[0]*-1)
    return images,reactions
            
images,reactions = get_img(samples)
reactions = np.array(reactions)
images = np.array(images)




        
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,Cropping2D
from keras.layers import Dropout, Flatten, Dense,Activation,Lambda
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint  

#build the model.
##the model structure comes from Nvedia
model = Sequential()
model.add(Lambda(lambda x: x/255 -0.5,input_shape=(160,320,3)))#normalize data
model.add(Cropping2D(cropping=((70,25),(0,0))))# cut the part of all picture that may useless for driving.
model.add(Conv2D(24,5,activation='relu',strides = (2,2),padding= 'same'))
model.add(Conv2D(36,5,activation='relu',strides = (2,2),padding= 'same'))
model.add(Conv2D(48,5,activation='relu',strides = (2,2),padding= 'same'))
model.add(Conv2D(64,3,activation='relu',strides = (2,2),padding= 'same'))
model.add(Conv2D(64,3,activation='relu',strides = (2,2),padding= 'same'))
model.add(Flatten())
model.add(Dense(100, use_bias=False))
model.add(Dense(50, use_bias=False))
model.add(Dense(10, use_bias=False))
model.add(Dense(1))
model.summary()

model.compile(optimizer='Nadam', loss='mse')
# Use Nadam optimizer which is like adam optimizer but normally perform better.
epochs = 5
batch_s = 32

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)
model.fit(images,reactions, 
          validation_split = 0.2,
          epochs=epochs, batch_size=batch_s, callbacks=[checkpointer], verbose=1,shuffle=True)
model.save('model.h5')



