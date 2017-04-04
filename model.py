# -*- encoding: utf-8 -*-

import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import math
import pickle
import os

# hyper parameters
left_right_correction = 0.3

def defaultReader(sample):
    img = cv2.imread(sample['image'])
    steering = sample['steering']
    return img, steering

def leftReader(sample):
    img = cv2.imread(sample['image'])
    steering = sample['steering']
    return img, steering + left_right_correction

def rightReader(sample):
    img = cv2.imread(sample['image'])
    steering = sample['steering']
    return img, steering - left_right_correction

def fliplrReader(sample):
    img = cv2.imread(sample['image'])
    steering = sample['steering']
    return np.fliplr(img), -steering

data_base = 'data'
sample_dirs = os.listdir(data_base)
sample_dirs = [data_base + "/" + d for d in sample_dirs]
samples = []
for d in sample_dirs:
    with open(d + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # replace path prefix
            center, left, right = line[0], line[1], line[2]
            center = d + "/IMG/" + center.split("\\")[-1]
            left   = d + "/IMG/" + left.split("\\")[-1]
            right  = d + "/IMG/" + right.split("\\")[-1]
            samples.append({
                'image': center,
                'steering': float(line[3]),
                'reader': defaultReader
            })
            samples.append({
                'image': center,
                'steering': float(line[3]),
                'reader': fliplrReader
            })
            samples.append({
                'image': left,
                'steering': float(line[3]),
                'reader': leftReader
            })
            samples.append({
                'image': right,
                'steering': float(line[3]),
                'reader': rightReader
            })

train_samples, valid_samples = train_test_split(samples, test_size = 0.2)
num_train_samples = len(train_samples)
num_valid_samples = len(valid_samples)

print("Number of train samples:", num_train_samples)
print("Number of valid samples:", num_valid_samples)

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for batch_start in range(0, num_samples, batch_size):
            batch = samples[batch_start:batch_start + batch_size]

            X,y = [],[]
            for sample in batch:
                img,steering = sample['reader'](sample)
                X.append(img)
                y.append(steering)
            X = np.array(X)
            y = np.array(y)
            X, y = shuffle(X, y)
            yield (X, y)

# model definition - LeNet
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# model.add(Conv2D(6, (5, 5)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(16, (5, 5)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(400))
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='adam')

# model definition - from nvidia
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), strides = (2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides = (2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides = (2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
          
batch_size = 64
steps_train = int(math.ceil(num_train_samples / batch_size))
steps_valid = int(math.ceil(num_valid_samples / batch_size))
train_generator = generator(train_samples, batch_size)
valid_generator = generator(valid_samples, batch_size)

# train model
history = model.fit_generator(train_generator, steps_train,
                              validation_data = valid_generator,
                              validation_steps = steps_valid,
                              epochs = 10)

# save model
model.save('model.h5')
