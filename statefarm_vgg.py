from __future__ import division,print_function

import os, json, sys
import os.path
from glob import glob
from shutil import copyfile
import numpy as np

#import modules
from utils import *
from vgg16bn import Vgg16BN
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

# pathing
path = "data/"
# path = "data/sample/"  # use smaller sample 
results_path = path + 'results/'
test_path = path + 'test/'

# training settings
batch_size = 16
nb_epoch = 5
data_augment_size = 3

# classes and labels
(val_classes, trn_classes, val_labels, trn_labels, 
    val_filenames, filenames, test_filenames) = get_classes(path)

# create batches
batches = get_batches(path+'train', batch_size=batch_size, shuffle=False)
val_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)
test_batches = get_batches(path+'test', batch_size=batch_size, shuffle=False)

# The 10 classes to predict are:
# c0: safe driving
# c1: texting - right
# c2: talking on the phone - right
# c3: texting - left
# c4: talking on the phone - left
# c5: operating the radio
# c6: drinking
# c7: reaching behind
# c8: hair and makeup
# c9: talking to passenger


##
## Build convolution layers with VGG
## We can use pre-trained VGG conv layers and predict output, and use as input to dense layer. 
def conv_model():
    vgg = Vgg16()
    model=vgg.model
    last_conv_idx = [i for i,l in enumerate(model.layers) if type(l) is Convolution2D][-1]
    conv_layers = model.layers[:last_conv_idx+1]
    return Sequential(conv_layers)
    # # load vgg with convolution layers only
    # vgg = Vgg16BN(include_top=False)
    # vgg.model.pop()  # pop last maxpooling layer
    # # freeze convolution layers
    # for l in vgg.model.layers:
    #     l.trainable = False
    # return vgg.model

def model_output_shape(model):
    return model.layers[-1].output_shape[1:]

def get_conv_feats(conv_model):
    if os.path.isdir(path+'results/conv_feat.dat') & os.path.isdir(path+'results/conv_val_feat.dat'):
        return load_conv_feats(conv_model)
    else:
        return calc_conv_feats(conv_model)

def calc_conv_feats(conv_model):
    print("calculating convolution features")
    conv_feat = conv_model.predict_generator(batches, batches.nb_sample)
    conv_val_feat = conv_model.predict_generator(val_batches, val_batches.nb_sample)
    # conv_test_feat = conv_model.predict_generator(test_batches, test_batches.nb_sample)
    
    # save arrays
    print("saving to file....")
    save_array(path+'results/conv_feat.dat', conv_feat)
    save_array(path+'results/conv_val_feat.dat', conv_val_feat)
    # save_array(path+'results/conv_test_feat.dat', conv_test_feat)    
    # return (conv_feat, conv_val_feat, conv_test_feat)
    return (conv_feat, conv_val_feat)

def load_conv_feats(conv_model):
    print("loading convolution features from file...")
    conv_feat = load_array(path+'results/conv_feat.dat')
    conv_val_feat = load_array(path+'results/conv_val_feat.dat')
    return (conv_feat, conv_val_feat)

##
## Dense layer with batch norm
## We feed output of convolution layers into dense layer. 
def dense_layers(p=0.8, input_shape=(512, 14, 14)):
    return [
        # MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        MaxPooling2D(input_shape=input_shape), 
        Flatten(),
        Dropout(p/2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p/2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
        ]

def dense_model(input_shape=(512, 14, 14), use_da=False):
    if use_da:
        print("using dense_da_layers")
        dl = dense_da_layers(input_shape=input_shape)
    else:
        dl = dense_layers(input_shape=input_shape)
    dense_model = Sequential(dl)

    optimizer = Adam(lr=0.0001)
    # optimizer = RMSprop(lr=0.00001, rho=0.7)
    dense_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return dense_model

def train(model, conv_feat, _labels, conv_val_feat, _val_labels=val_labels, nb_epoch=5):
    # train dense model using conv features as input
    model.fit(conv_feat, _labels, batch_size=batch_size, nb_epoch=nb_epoch, 
             validation_data=(conv_val_feat, _val_labels))
    model.save_weights(path+'models/conv_weights.h5')
    return model


## 
## add image augementation
##

def get_da_conv_feats(conv_model):
    if os.path.isdir(path+'results/da_conv_feat.dat') & os.path.isdir(path+'results/conv_val_feat.dat'):
        return load_da_conv_feats(conv_model)
    else:
        print("features not found, calculating new...")
        return calc_da_conv_feats(conv_model)

def calc_da_conv_feats(conv_model):
    print("generating data-augmented batches...")
    # image augmentation generator
    gen_t = image.ImageDataGenerator(rotation_range=15, height_shift_range=0.05, 
        shear_range=0.1, channel_shift_range=20, width_shift_range=0.1)
    da_batches = get_batches(path+'train', gen_t, batch_size=batch_size, shuffle=False)

    print("calculating data-augmented conv features")
    # generate data-augmented conv features
    da_conv_feat = conv_model.predict_generator(da_batches, da_batches.nb_sample * data_augment_size)

    # save arrays
    print("saving to file....")
    save_array(path+'results/_da_conv_feat.dat', da_conv_feat)

    # release memory
    del da_conv_feat
    del da_batches
    del gen_t

    # concatenate with original conv features
    print("concatenating....")
    conv_feat = load_array(path+'results/conv_feat.dat')
    da_conv_feat = load_array(path+'results/_da_conv_feat.dat')
    da_conv_feat = np.concatenate([da_conv_feat, conv_feat])

    # save arrays
    print("saving to file....")
    save_array(path+'results/da_conv_feat.dat', da_conv_feat)
    return da_conv_feat

def load_da_conv_feats(conv_model):
    print("loading data-augmented convolution features from file...")
    conv_feat = load_array(path+'results/da_conv_feat.dat')
    conv_val_feat = load_array(path+'results/conv_val_feat.dat')
    return (conv_feat, conv_val_feat)

def get_da_trn_labels():
    da_trn_labels = np.concatenate([trn_labels]*(data_augment_size+1))
    return da_trn_labels

def dense_da_layers(p=0.8, input_shape=(512, 14, 14)):
    return [
        MaxPooling2D(input_shape=input_shape), 
        Flatten(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(p),
        Dense(10, activation='softmax')
        ]

##
## testing
##
def calc_test_conv_feats(conv_model):
    print("calculating test convolution features")
    conv_test_feat = conv_model.predict_generator(test_batches, test_batches.nb_sample)
    # save arrays
    print("saving to file....")
    save_array(path+'results/conv_test_feat.dat', conv_test_feat)
    return conv_test_feat

def load_test_conv_feats():
    print("loading test convolution features from file...")
    conv_test_feat = load_array(path+'results/conv_test_feat.dat')
    return conv_test_feat
    
def get_test_conv_feats(conv_model):
    if os.path.isdir(path+'results/conv_test_feat.dat'):
        return load_test_conv_feats()
    else:
        return calc_test_conv_feats(conv_model)

def test(model, conv_test_feat):
    preds = model.predict(conv_test_feat, batch_size=batch_size*2)
    return preds

def submit(preds):
    def do_clip(arr, mx): 
        return np.clip(arr, (1-mx)/9, mx)

    subm = do_clip(preds,0.93)
    subm_name = path+'results/subm.gz'
    # make classes
    classes = sorted(batches.class_indices, key=batches.class_indices.get)
    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'img', [a[4:] for a in test_filenames])
    print(submission.head())
    print("saving to csv: " + subm_name)
    submission.to_csv(subm_name, index=False, compression='gzip')


##
## main scripts
##
def run_basic():
    cm = conv_model()
    print("build convolution model from VGG conv layers")
    print(cm.summary())
    print("===============================================")

    conv_output_shape = model_output_shape(cm)
    model = dense_model(input_shape=conv_output_shape)
    print("build dense model with shape: %s" % (conv_output_shape,))
    print(model.summary())
    print("===============================================")

    print("get conv layer features")
    conv_feat, conv_val_feat = get_conv_feats(cm)

    print("training for %d epochs..." % nb_epoch)
    model = train(model, conv_feat, trn_labels, conv_val_feat, val_labels, nb_epoch)

    # clear memory
    del conv_feat
    del conv_val_feat
    del cm 
    return model 

def precalc_da_conv_feat():
    cm = conv_model()
    print("build convolution model from VGG conv layers")
    print(cm.summary())
    print("===============================================")
    print("pre-calculating data-augmented conv layer features")
    calc_da_conv_feats(cm)

def precalc_test_conv_feat():
    cm = conv_model()
    print("pre-calculating test conv features")
    calc_test_conv_feats(cm)

def run_augmented():
    cm = conv_model()
    print("build convolution model from VGG conv layers")
    print(cm.summary())
    print("===============================================")

    conv_output_shape = model_output_shape(cm)
    model = dense_model(input_shape=conv_output_shape, use_da=True)
    print("build dense data-augmented model with shape: %s" % (conv_output_shape,))
    print(model.summary())
    print("===============================================")

    print("get conv layer features")
    conv_da_feat, conv_val_feat = get_da_conv_feats(cm)

    print("training for %d epochs..." % nb_epoch)
    da_trn_labels = get_da_trn_labels()
    model = train(model, conv_da_feat, da_trn_labels, conv_val_feat, val_labels, nb_epoch)

    print("saving model weights... ")
    model.save_weights(path+'models/da_model.h5')

    # clear memory
    del conv_da_feat
    del conv_val_feat
    del cm 

    return model

def run_test(dense_mdl):
    conv_test_feat = load_array(path+'results/conv_test_feat.dat')
    print("running test....")
    preds = test(dense_mdl, conv_test_feat)

    # clear memory
    del conv_test_feat
    del dense_mdl
    print("preparing submission....")
    submit(preds)

def run_test_only():
    print("loading model")
    layers = dense_da_layers()
    model = Sequential(layers)
    model.load_weights(path+'models/da_model.h5')
    
    print("loading conv test features")
    conv_test_feat = load_array(path+'results/conv_test_feat.dat')

    print("running test....")
    preds = test(model, conv_test_feat)

    # clear memory
    del conv_test_feat
    del model
    print("preparing submission....")
    submit(preds)


# main
# precalc_da_conv_feat()
precalc_test_conv_feat()
dense_mdl = run_basic()
run_test(dense_mdl)

# da run
# run_augmented()
# run_test_only()
