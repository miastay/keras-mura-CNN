#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:16:05 2019

@author: ryan
"""
import numpy as np
import pandas as pd
from skimage.io import imread
from matplotlib import pyplot as plot
#to measure program time
import time
start_time = time.time()

train_locs = pd.io.parsers.read_csv("./networkdata/wrist_train_image_paths.csv")
test_locs = pd.io.parsers.read_csv("./networkdata/wrist_valid_image_paths.csv")
batch_size = 5
train_size = 110
valid_size = 40
num_epochs = 3
img_width = 256
img_height = 256
blocks = 5

def initialize_data() :
    print("Initializing")
    '''Instantiates test data'''
    

    X_train = list()
    Y_train = list()
    i = 0
    for row in train_locs['Location'] :
        img = imread(row, as_gray = True)
        if(img.dtype != "uint8") :
            img *= 255
            img = img.astype("uint8")
        if len(img[0]) > img_width :
            Y_train.append([1, 0] if ('positive' in row) else [0, 1])
            img = np.reshape(img, (512, img.shape[1], 1))
            index = int((len(img[0])-img_width)/2) +1
            end_index = img_width-index
            fin = np.full((img_width,img_height,1), 0)
            for r in range(img_height) :
                for c in range(index, end_index) :
                    fin.itemset((r,c,0), img.item((r,c-index,0)))
            X_train.append(fin)
            i += 1
            if(i >= train_size) :
                break
    
    '''Instantiates validation data'''
    
    X_test = list()
    Y_test = list()
    i = 0
    for row in test_locs['Locations'] :
        img = imread(row, as_gray = True)
        if(img.dtype != "uint8") :
            img *= 255
            img = img.astype("uint8")
        if len(img[0]) < 499 :
            Y_test.append([1, 0] if ('positive' in row) else [0, 1])
            img = np.reshape(img, (512, img.shape[1], 1))
            index = int((512-len(img[0]))/2) +1
            end_index = 512-index
            fin = np.full((512,512,1), 0)
            for r in range(512) :
                for c in range(index, end_index) :
                    fin.itemset((r,c,0), img.item((r,c-index,0)))
            X_test.append(fin)
            i += 1
            if(i >= valid_size) :
                break
    
    '''code up until this point functions correctly'''
    
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)
    
    print("Data initialized in %s seconds." % (time.time() - start_time))
    
    return X_train, Y_train, X_test, Y_test
    

import keras.backend as K
start_time = time.time()

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))
    
def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Activation, BatchNormalization, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

def train_network(X_train, Y_train, X_test, Y_test, batch_size, num_epochs) :
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    #Y_train = keras.utils.to_categorical(Y_train, num_classes=2)
    #Y_test = keras.utils.to_categorical(Y_test, num_classes=2)
    print(Y_train.shape, ",", Y_test.shape)
    
    data_generator = ImageDataGenerator(rotation_range=90,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        featurewise_center=True,
                                        featurewise_std_normalization=True,
                                        horizontal_flip=True)
    
    data_generator.fit(X_train)
    '''code appears to work through this point'''
    
    '''rework following code as needed'''
    
    for i in range(len(X_test)):
        X_test[i] = data_generator.standardize(X_test[i])
        
    model = Sequential()
    for index in range(blocks) :
        model.add(Conv2D(2*index, (3, 3), padding='same', input_shape=X_train.shape[1:]))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*index, (3, 3), padding='same'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        print(index)
        if(index % 10 == 0) :
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
    #model.add(Dropout(0.2))
    #add in CAM layering
    #
    #ensure that flattening layer is correctly positioned
    model.add(Flatten())
    #model.add(GlobalAveragePooling2D())
    #model.add(Lambda(global_average_pooling, output_shape=global_average_pooling_shape))
    model.add(Dense(2, activation='softmax', input_shape=(2,)))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    hist = model.fit_generator(
        generator=data_generator.flow(x=X_train,
                                      y=Y_train,
                                      batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=num_epochs,
        validation_data=(X_test, Y_test),
        workers=4)
    
    print("Network completed in %s seconds." % (time.time() - start_time))
    return model, hist
    
def save_model_as_JSON(model, savepath, weightpath) :
    json_string = model.to_json()
    save = open(savepath, 'w')
    save.write(json_string)
    save.close
    model.save_weights(weightpath)
    
def load_model_from_JSON(modelpath, weightpath) :
    from keras.models import model_from_json
    json_file = open(modelpath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weightpath)
    return loaded_model

def run_class_activation(model, params, img) :
#model = load_model_from_JSON("./networkdata/model.json", "./networkdata/weights.h5")
    print(model)
    from keras.models import Model
    import scipy as sp
    weights = model.layers[-1].get_weights()[0]
    print(weights.shape)
    cam_model = Model(inputs=model.input, outputs=(model.layers[-6].output,model.layers[-1].output))
    '''
    X_train = params[0]
    Y_train = params[1]
    X_test = params[2]
    Y_test = params[3]
    '''
    features,results = cam_model.predict(img)
    print(features.shape)
    
    img_features = features[0,:,:,:]
    height = img.shape[1] / img_features.shape[0]
    width = img.shape[2] / img_features.shape[1]
    
    cam_features = sp.ndimage.zoom(img_features, (height, width, 1), order=2)
    predict = np.argmax(results[0])
    cam_features = img_features
    
    plot.figure(facecolor="white")
    cam_weights = weights[:,predict]
    cam_output = np.dot(cam_features,cam_weights)
    
    plot.imshow(np.squeeze(img[0], -1), alpha=0.5)
    plot.imshow(cam_output, cmap = 'jet', alpha=0.5)
    
    plot.show()

    
def get_image(filepath) : 
    return imread(filepath, as_gray = True)

def run_cam_visualization() :
    from vis.visualization import visualize_cam 
    from matplotlib import pyplot as plot
    plot.imshow(visualize_cam(model, 17, filter_indices=None, seed_input=get_image("./networkdata/testimg.png"), penultimate_layer_idx=None,
    backprop_modifier=None, grad_modifier=None))
    
params = initialize_data()
(model, history) = train_network(params[0], params[1], params[2], params[3], batch_size, num_epochs)
save_model_as_JSON(model, "./networkdata/model.json", "./networkdata/weights.h5")

import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



