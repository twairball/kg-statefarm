from __future__ import division,print_function

import os, json, sys
from glob import glob
from shutil import copyfile
import numpy as np
import pandas as pd

from utils import *
from vgg16 import Vgg16

import errno
def mkdir_p(path):
    """ 'mkdir -p' in Python """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def setup_datasets():
    # create validation dataset
    DATA_HOME_DIR = os.getcwd() + '/data'

    def filepath(x):
        return x['classname'] + '/' + x['img']

    #Create directories
    for folder in ['valid', 'results', 'models',
        'sample/train', 'sample/test', 'sample/valid', 'sample/results', 'sample/models']:

        folder_path = DATA_HOME_DIR + '/' + folder
        mkdir_p(folder_path)

    # driver images
    df_imgs = pd.read_csv('data/driver_imgs_list.csv')

    # create class subdirectories for valid, and sample
    for c in df_imgs['classname'].unique():
        mkdir_p(DATA_HOME_DIR + '/valid/' + c)
        mkdir_p(DATA_HOME_DIR + '/sample/train/' + c)
        mkdir_p(DATA_HOME_DIR + '/sample/valid/' + c)

    # randomly pick 3 for validation 
    subjects = np.random.permutation(df_imgs['subject'].unique())
    val_subjs = subjects[0:3]
    df_vals = df_imgs[df_imgs['subject'].isin(val_subjs)]
    df_train = df_imgs[~df_imgs['subject'].isin(val_subjs)]

    # move validation subjects to validation
    df_vals.apply(lambda x: 
        os.rename(
            DATA_HOME_DIR + '/train/' + filepath(x), 
            DATA_HOME_DIR + '/valid/' + filepath(x)), 
        axis=1
    )

    # randomly copy training to sample
    sample_train = df_train.groupby(['subject', 'classname']).head(30)
    sample_train.apply(lambda x: 
        copyfile(
            DATA_HOME_DIR + '/train/' + filepath(x), 
            DATA_HOME_DIR + '/sample/train/' + filepath(x)), 
        axis=1
    )

    # randomly copy validation to sample 
    sample_vals = df_vals.groupby(['subject', 'classname']).head(30)
    sample_vals.apply(lambda x: 
        copyfile(
            DATA_HOME_DIR + '/valid/' + filepath(x), 
            DATA_HOME_DIR + '/sample/valid/' + filepath(x)), 
        axis=1
    )
