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
        self.image  = image
        self.mos    = mos

        # Index
        ind = np.array(range(image.shape[0]-1))
	
        # Parameters
        self.batch = batches
        self.iter = math.ceil(image.shape[0]/batches)                # Batch num for each 1 Epoch
        self.rnd = np.r_[ind,ind] 				     # Reuse beggining of data when not enough data

    def shuffle(self):
	rd.shuffle(self.rnd)
	
    ## 	Pop batch data
    def next(self, i):
        index = self.rnd[ i * self.batch : (i + 1) * self.batch ]   # Index of extracting data
        return self.image[index], self.mos[index]                   # Image & MOS


class create_batch_test:
    """
    Creating Batch Data for test
    """

    ## 	Initialization
    def __init__(self, image, mos, start_frame=None, stop_frame=None):

        # Processing range
        if start_frame is None:             # Start frame position
            start_frame  = 0
        if stop_frame is None:              # Stop frame position
            stop_frame   = image.shape[0]

        # Parameters
        self.image  = np.array(image[start_frame:stop_frame])
        self.mos    = np.array(mos[start_frame:stop_frame])
