#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import math
import joblib
import glob
import re


import cv2

import numpy as np
import numpy.random as rd

from settings import settings

import os


# -------------------------------------------
#   Load pkl files or ".jpg" & ".csv" files
# -------------------------------------------
def data_loader(test=False):
    """
    Read wav files or Load pkl files
	"""

    ##  Sort function for file name
    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

	# Load settings
    args = settings()

    # Make folder
    if not os.path.exists(args.model_save_path):    # Folder of model
        os.makedirs(args.model_save_path)

    if not os.path.exists(args.pkl_path):           # Folder of train pkl
        os.makedirs(args.pkl_path)

    # File name
    if not test:
        image_names = args.train_data_path + '/*.bmp'
        eval_names  = args.train_data_path + '/*.txt'
        pkl_image   = args.pkl_path + '/train_image.pkl'
        pkl_eval    = args.pkl_path + '/train_eval.pkl'

    else:
        image_names = args.test_data_path + '/*.bmp'
        eval_names  = args.test_data_path + '/*.txt'
        pkl_image   = args.pkl_path + '/test_image.pkl'
        pkl_eval    = args.pkl_path + '/test_eval.pkl'

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   No pkl files
    ##    -> Read images & assesment values, and Create pkl files
    ##  ~~~~~~~~~~~~~~~~~~~
    if not (os.access(pkl_image, os.F_OK) and os.access(pkl_eval, os.F_OK)):

        ##  Read Image files
        print(' Load bmp file...')

        # Get image data
        image_data = []
        for image_file in sorted(glob.glob(image_names), key=numericalSort):
            image_data.append(cv2.resize(cv2.imread(image_file), args.size).transpose(2,0,1))

        image_data = np.array(image_data)

        # Get evaluation data
        eval_data = []
        for image_file in sorted(glob.glob(eval_names), key=numericalSort):
            eval_data = np.expand_dims(np.loadtxt(glob.glob(eval_names)[0], delimiter=',', dtype='float'), axis=1)

        ##  Create Pkl files
        print(' Create Pkl file...')
        with open(pkl_image, 'wb') as f:        # Create clean pkl file
            joblib.dump(image_data, f, protocol=-1, compress=3)

        with open(pkl_eval, 'wb') as f:         # Create noisy pkl file
            joblib.dump(eval_data, f, protocol=-1, compress=3)

    else:
        with open(pkl_image, 'rb') as f:        # Load image pkl file
            print(' Load Image Pkl...')
            image_data = joblib.load(f)

        with open(pkl_eval, 'rb') as f:         # Load evaluation pkl file
            print(' Load Evaluation Pkl...')
            eval_data = joblib.load(f)

    return image_data, eval_data


class create_batch:
    """
    Creating Batch Data for training
    """

    ## 	Initialization
    def __init__(self, image, mos, batches, test=False):

        # Data Shaping
        self.image  = image.copy()
        self.mos    = mos.copy()

        # Random index ( for data scrambling)
        ind = np.array(range(image.shape[0]-1))
        if not test:
            rd.shuffle(ind)

        # Parameters
        self.i = 0
        self.batch = batches
        self.iter_n = math.ceil(image.shape[0] / batches)               # Batch num for each 1 Epoch
        self.rnd = np.r_[ind, np.random.choice(self.iter_n*batches-len(ind))] # Reuse beggining of data when not enough data

    def shuffle(self):
        self.i = 0
        rd.shuffle(self.rnd)

    def __iter__(self):
        return self

    ## 	Pop batch data
    def __next__(self):
        index = self.rnd[ self.i * self.batch : (self.i + 1) * self.batch ]   # Index of extracting data
        self.i += 1
        return self.image[index], self.mos[index]                   # Image & MOS
