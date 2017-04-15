# -*- encoding: utf-8 -*-
import argparse
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import math
import pickle
import os
import random

parser = argparse.ArgumentParser(description = "Model trainer")
parser.add_argument('--model', default="model", metavar='MODEL', help='name of the output model.')
parser.add_argument('--epochs', default="10", type=int, metavar='EPOCHS', help='number of epochs to train.')
args = parser.parse_args()

# hyper parameters
# batch size
batch_size = 64

image_size = (160, 320, 3)
cropping_tb = (65, 25)
cropping_lr = (0, 0)

def readImg(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

def imageReadFn(path):
    return lambda : readImg(path)

def flippedImageReadFn(path):
    return lambda : np.fliplr(readImg(path))

# preparing training and validating samples
data_base = 'data'
sample_dirs = os.listdir(data_base)
sample_dirs = [data_base + "/" + d for d in sample_dirs]
samples_all = []
for d in sample_dirs:
    with open(d + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # replace path prefix
            center, left, right = line[0], line[1], line[2]
            center = d + "/IMG/" + center.split("\\")[-1]
            left   = d + "/IMG/" + left.split("\\")[-1]
            right  = d + "/IMG/" + right.split("\\")[-1]
            steering = float(line[3])

            # image from center camera
            samples_all.append({
                'imageFn': imageReadFn(center),
                'steering':  steering
            })
            # image from center camera, flipped
            samples_all.append({
                'imageFn': flippedImageReadFn(center),
                'steering': -steering
            })

samples = []
samples_all = shuffle(samples_all)
print(len(samples_all))

for sample in samples_all:
    steering = sample['steering']
    samples.append(sample)

# splitting data points into training set and validation set
train_samples, valid_samples = train_test_split(samples, test_size = 0.2)
num_train_samples = len(train_samples)
num_valid_samples = len(valid_samples)

print("Number of train samples:", num_train_samples)
print("Number of valid samples:", num_valid_samples)

def generator(samples, batch_size = 32):
    """Batch generator feeding to the network.
    """
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for batch_start in range(0, num_samples, batch_size):
            batch = samples[batch_start:batch_start + batch_size]

            X,y = [],[]
            for sample in batch:
                img, steering = sample['imageFn'](), sample['steering']
                X.append(img)
                y.append(steering)
            X = np.array(X)
            y = np.array(y)
            X, y = shuffle(X, y)
            yield (X, y)

model_name = "model"
if args.model:
    model_name = args.model
    
# model definition - NVidia architecture
model = Sequential()
# cropping
model.add(Cropping2D(cropping=(cropping_tb, cropping_lr), input_shape=image_size))
# normalizing image ( to [-0.5, 0.5] )
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, (5, 5), strides = (2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides = (2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides = (2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
    
steps_train = int(math.ceil(num_train_samples / batch_size))
steps_valid = int(math.ceil(num_valid_samples / batch_size))
train_generator = generator(train_samples, batch_size)
valid_generator = generator(valid_samples, batch_size)

# train model
print("Start training model {}".format(model_name))
history = model.fit_generator(train_generator, steps_train,
                              validation_data = valid_generator,
                              validation_steps = steps_valid,
                              epochs = args.epochs)

# save model
model.save(model_name + ".h5")
print("Model saved to {}".format(model_name + ".h5"))

print(history.history)
