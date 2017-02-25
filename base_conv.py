from __future__ import division,print_function

import os, json, sys
import os.path
from glob import glob
from shutil import copyfile
import numpy as np
import pandas as pd

#import modules
from utils import *
from vgg16 import Vgg16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

def submit(preds, test_batches, filepath):
    def do_clip(arr, mx): 
        return np.clip(arr, (1-mx)/9, mx)

    def img_names(filenames):
        df = pd.DataFrame(filenames, columns=['img'])
        df.loc[:, 'img'] = df['img'].str.replace('unknown/', '')
        df.loc[:, 'img'] = df['img'].str.replace('own/', '')
        return df

    def preprocess(subm):
        # make classes
        classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
        submission = pd.DataFrame(subm, columns=classes)
        return submission
    
    # make submission dataframe
    df_img_names = img_names(test_batches.filenames)
    subm = do_clip(preds,0.93)
    submission = preprocess(subm)
    submission = pd.concat([df_img_names, submission], axis=1)

    print(submission.head())
    print("saving to csv: " + filepath)
    submission.to_csv(filepath, index=False, compression='gzip')

def push_to_kaggle(filepath):
    command = "kg submit -c state-farm-distracted-driver-detection " + filepath
    os.system(command)

class BaseConvModel():
    
    def __init__(self, path='data/'):
        self.model = self.create_model()
        self.path = path
        self.train_batches = self.create_batches(path+'train/')
        self.val_batches = self.create_batches(path+'valid/')
        self.test_batches = self.create_batches(path+'test/')

    def get_conv_layers(self):
        vgg = Vgg16()
        model=vgg.model
        last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
        conv_layers = model.layers[:last_conv_idx+1]
        return conv_layers
        
    def get_fc_layers(self, input_shape):
        return [
                MaxPooling2D(input_shape=input_shape),
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.),
                Dense(512, activation="relu"),
                Dropout(0.),
                Dense(10, activation="softmax")
        ]

    def create_batches(self, path):
        batch_size = 64
        target_size = (224, 224)
        return image.ImageDataGenerator().flow_from_directory(path, 
            target_size = target_size,
            batch_size = batch_size,
            class_mode = 'categorical',
            shuffle = False
        )

    def create_model(self):
        # conv
        conv_layers = self.get_conv_layers()
        conv_output_shape = conv_layers[-1].output_shape[1:]
        # dense
        fc_layers = self.get_fc_layers(conv_output_shape)
        layers = conv_layers + fc_layers
        # model
        model = Sequential(layers)
        optimizer = Adam(lr=0.001)
        # optimizer = RMSprop(lr=0.00001, rho=0.7)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        nb_epoch = 5

        self.model.fit_generator(
            self.train_batches, 
            samples_per_epoch = self.train_batches.nb_sample,
            nb_epoch = nb_epoch,
            validation_data = self.val_batches, 
            nb_val_samples = self.val_batches.nb_sample)

        self.model.save_weights(self.path+'models/base_conv_weights.h5')

    def test(self):
        return self.model.predict_generator(self.test_batches, self.test_batches.nb_sample)

if __name__ == "__main__":
    path = 'data/'
    bcm = BaseConvModel(path=path)
    print(bcm.model.summary())
    print("===============================================================")
    print("training model....")
    bcm.train()
    print("===============================================================")

    print("testing model....")
    preds = bcm.test()
    test_preds_filepath = path + 'results/base_test_preds.dat'
    save_array(test_preds_filepath, preds)

    print("preparing submission...")
    submission_filepath = path+'results/base_conv_subm.gz'
    submit(preds, bcm.test_batches, submission_filepath)
    
    print("push to kaggle...")
    push_to_kaggle(submission_filepath)
