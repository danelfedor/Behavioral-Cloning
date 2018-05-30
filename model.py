import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#put all picture address and reaction to a list
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]

#split picture and record to different list
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                if abs(float(batch_sample[3]))>0.08: #set up a penulty if turning larger than that I will append a flipped picture to data set
                    name = r'G:\Users\Hu Wenqi\Dropbox\Self-Driving\CarND-Behavioral-Cloning-P3\data\IMG\\' + batch_sample[0].split('\\')[-1]
                    image = cv2.imread(name)
                    print(name)
                    images.append(image)
                    images.append(cv2.flip(image,1))
                    angles.append(float(batch_sample[3]))
                    angles.append(float(batch_sample[3])*-1)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
        
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D,Cropping2D
from keras.layers import Dropout, Flatten, Dense,Activation,Lambda
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint  
from keras.layers.advanced_activations import LeakyReLU
#build the model.
##the model structure comes from Nvedia
model = Sequential()
model.add(Lambda(lambda x: x/255 -0.5,input_shape=(160,320,3)))#normalize data
model.add(Cropping2D(cropping=((70,25),(0,0))))# cut the part of all picture that may useless for driving.
model.add(BatchNormalization())
model.add(Conv2D(24,5,strides = (2,2),activation='relu'))
model.add(Conv2D(36,5,strides = (2,2),activation='relu'))
model.add(Conv2D(48,5,strides = (2,2),activation='relu'))
model.add(Conv2D(64,3,activation='elu'))
model.add(Conv2D(64,3,activation='elu'))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(100,activation='elu'))
model.add(BatchNormalization())
model.add(Dense(50,activation='elu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='elu'))
model.add(Dense(1))
model.summary()
model.compile(optimizer='Nadam', loss='mse')

# Use Nadam optimizer which is like adam optimizer but normally perform better.
epochs = 5
model.compile(optimizer='Nadam', loss='mse')
checkpointer = ModelCheckpoint(filepath='weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)/32,callbacks=[checkpointer],
                    validation_data=validation_generator,nb_val_samples=len(validation_samples)/32, nb_epoch=epochs)
model.load_weights('weights.best.from_scratch.hdf5')
model.save('model.h5')