#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from six.moves import range

import os
import time
import numpy as np
from scipy import stats
from tqdm import tqdm

#   NNabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context  # GPU

#   Image
import cv2

#   Originals
from settings import settings
import data as dt

# -------------------------------------------
#   Network for IQA
# -------------------------------------------
def network(input, scope="", test=False):
    """
    Define Convolutional DNN
       input:   input image ( size must be 64 by 64 )
       test:    test or not
    """

    # not used
    def resblock(x, n=256,name=''):
        r = x
        with nn.parameter_scope('block1-'+name):
            r = PF.convolution(r, n, kernel=(5, 5), pad=(2, 2), stride=(1, 1))
            r = PF.batch_normalization(r, batch_stat=not test)
            r = F.relu(r)
        with nn.parameter_scope('block2-'+name):
            r = PF.convolution(r, n, kernel=(5, 5), pad=(2, 2), stride=(1, 1))
        return F.concatenate(r, x, axis=1)

    def convblock(x, n, f_size=9, name=''):
        r = PF.convolution(x, n, kernel=(f_size, f_size), pad=(f_size//2, f_size//2), stride=(1, 1), name=name)
        return F.relu(r)

    with nn.parameter_scope(scope):

        # Convolution : 3,64,64 -> 32,64,64 -> 32,32,32
        c1 = convblock(input, 64, name='cnv1')
        c1 = F.max_pooling(c1, (2,2), (2,2))

        # Convolution : 32,32,32 -> 32,32,32 -> 32,16,16
        c2 = convblock(c1, 32, name='cnv2')
        c2 = F.max_pooling(c2, (2,2), (2,2))

        # Convolution : 32,16,16 -> 64,16,16 -> 64,8,8
        c3 = convblock(c2, 16, f_size=5, name='cnv3')
        c3 = F.max_pooling(c3, (2, 2), (2, 2))

        # Convolution : 64,8,8 -> 64,8,8 -> 64,4,4
        c4 = convblock(c2, 16, f_size=3, name='cnv4')
        c4 = F.max_pooling(c3, (2, 2), (2, 2))

        c3 = F.leaky_relu(PF.affine(c3, (128,), name='Affine1'), 0.01)
        c4 = PF.affine(c3, (1,), name='Affine2')

    return c4


# -------------------------------------------
#   Training
# -------------------------------------------
def train(args):
    """
    Training
    """

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable
    nn.clear_parameters()                                   #   Clear
    Input   = nn.Variable([args.batch_size, 3, 64, 64])     #   Input
    Trues   = nn.Variable([args.batch_size, 1])             #   True Value

    #   Network Definition
    Name    = "CNN"                                         #   Name of scope which includes network models (arbitrary)
    Output  = network(Input, scope=Name)                    #   Network & Output
    Output_test = network(Input, scope=Name, test=True)

    #   Loss Definition
    Loss    = F.mean(F.absolute_error(Output, Trues) )      #   Loss Function (Squared Error)
    Loss_test = F.mean(F.absolute_error(Output_test, Trues))

    #   Solver Setting
    solver = S.AMSBound(args.learning_rate)         #   Adam is used for solver
    with nn.parameter_scope(Name):                  #   Get updating parameters included in scope
        solver.set_parameters(nn.get_parameters())

    #   Training Data Setting
    image_data, mos_data = dt.data_loader()
    batches = dt.create_batch(image_data, mos_data, args.batch_size)
    del image_data, mos_data

    #   Test Data Setting
    image_data, mos_data = dt.data_loader(test=True)
    batches_test = dt.create_batch(image_data, mos_data, args.batch_size)
    del image_data, mos_data


    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Learning
    ##  ~~~~~~~~~~~~~~~~~~~
    print('== Start Training ==')

    bar = tqdm(total=args.epoch-args.retrain, leave=False)
    bar.clear()
    loss_disp = None
    SRCC = None

    #   Load data
    if args.retrain > 0:
        with nn.parameter_scope(Name):
            print('Retrain from {0} Epoch'.format(args.retrain))
            nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.retrain)))
            solver.set_learning_rate(args.learning_rate/np.sqrt(args.retrain))

    ##  Training
    for i in range(args.retrain,args.epoch):

        bar.set_description_str('Epoch {0}:'.format(i+1), refresh=False)
        if (loss_disp is not None) and (SRCC is not None):
            bar.set_postfix_str('Loss={0:.5f},  SRCC={1:.4f}'.format(loss_disp, SRCC), refresh=False)
        bar.update(1)

        #   Shuffling
        batches.shuffle()
        batches_test.shuffle()

        ##  Batch iteration
        for j in range(batches.iter_n):

            #  Load Batch Data from Training data
            Input.d, Trues.d    = next(batches)

            #  Update
            solver.zero_grad()                      #   Initialize
            Loss.forward(clear_no_need_grad=True)   #   Forward path
            Loss.backward(clear_buffer=True)        #   Backward path
            solver.weight_decay(0.00001)             #   Weight Decay for stable update
            solver.update()

        ## Progress
        # Get result for Display
        Input.d, Trues.d = next(batches_test)
        Loss_test.forward(clear_no_need_grad=True)
        Output_test.forward()
        loss_disp = Loss_test.d
        SRCC, _ = stats.spearmanr(Output_test.d, Trues.d)

        # Display text
        # disp(i, batches.iter_n, Loss_test.d)

        ## Save parameters
        if ((i + 1) % args.model_save_cycle) == 0 or (i + 1) == args.epoch:
            bar.clear()
            with nn.parameter_scope(Name):
                nn.save_parameters(os.path.join(args.model_save_path, 'network_param_{:04}.h5'.format(i+1)))


# -------------------------------------------
#   Test
# -------------------------------------------
def test(args):
    """
    Training
    """

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable
    nn.clear_parameters()  # Clear
    Input = nn.Variable([1, 3, 64, 64])  # Input
    Trues = nn.Variable([1, 1])          # True Value

    #   Network Definition
    Name = "CNN"  # Name of scope which includes network models (arbitrary)
    Output_test = network(Input, scope=Name, test=True)     # Network & Output
    Loss_test = F.mean(F.absolute_error(Output_test, Trues))  # Loss Function (Squared Error)

    #   Load data
    with nn.parameter_scope(Name):
        nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.epoch)))

    # Training Data Setting
    image_data, mos_data = dt.data_loader(test=True)
    batches = dt.create_batch(image_data, mos_data, 1)
    del image_data, mos_data

    truth = []
    result = []
    for j in range(batches.iter_n):
        Input.d, tures = next(batches)
        Loss_test.forward(clear_no_need_grad=True)
        result.append(Loss_test.d)
        truth.append(tures)

    result = np.array(result)
    truth = np.squeeze(np.array(truth))

    # Evaluation of performance
    mae = np.average(np.abs(result - truth))
    SRCC, p1 = stats.spearmanr(truth, result)    # Spearman's Correlation Coefficient
    PLCC, p2 = stats.pearsonr(truth, result)

    #   Display
    print("\n Model Parameter [epoch={0}]".format(args.epoch))
    print(" Mean Absolute Error with Truth: {0:.4f}".format(mae))
    print(" Speerman's Correlation Coefficient: {0:.3f}".format(SRCC))
    print(" Pearson's Linear Correlation Coefficient: {0:.3f}".format(PLCC))


# -------------------------------------------
#   Real-Time Demo
# -------------------------------------------
def demo(args):
    """
    Training
    """

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Arbitrary Parameters
    ##  ~~~~~~~~~~~~~~~~~~~
    #   Evaluation Settings
    Frame_per_calc  = 10                        #   No. of Frames per Calculation Cycle
    magnification   = 1.2                       #   Fine tuning for score value

    #   Video Settings
    frame_rate = 25.0                           #   Frame per Second
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  #     Video file format (mp4)

    #   Display Settings
    font_type   = cv2.FONT_HERSHEY_SIMPLEX      #   Font Type : Hershey fonts
    front_color = (0, 0, 0)                     #   Font Color : front
    back_color  = (255, 255, 255)               #   Font Color : background
    position    = (30, 50)                      #   Context Position


    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Initial settings
    ##  ~~~~~~~~~~~~~~~~~~~

    #   Input Variable
    nn.clear_parameters()                           #   Clear
    Input   = nn.Variable([1, 3, 64, 64])#   Input

    #   Network Definition
    Name    = "CNN"                                 #   Name of scope which includes network models (arbitrary)
    Output  = network(Input, scope=Name, test=True) #   Network & Output

    #   Load data
    with nn.parameter_scope(Name):
        print(args.epoch)
        nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.epoch)))

    #   Video Device
    deviceID = 0
    cap = cv2.VideoCapture(deviceID)

    ##  ~~~~~~~~~~~~~~~~~~~
    ##   Real-time IQA
    ##  ~~~~~~~~~~~~~~~~~~~
    #   Get video information
    _, frame = cap.read()           #   Capture video at once
    height = frame.shape[0]         #   Video size : height
    width = frame.shape[1]          #   Video size : width
    if height > width:
        trim_height = round(abs(height - width) / 2)
        trim_width = 0
    else:
        trim_height = 0
        trim_width = round(abs(height - width) / 2)

    #   Temporary Parameters for calculation
    cnt         = 0
    result      = []
    result_ave  = 0
    video_coding = 0

    while (True):

        # Capture from Video device
        ret, frame = cap.read()

        #   Waiting keyboad input
        key = cv2.waitKey(40) & 0xFF

        #   Resizing Image
        frame_trim = frame[trim_height:height-trim_height,trim_width:width-trim_width,:]   # Trimming so as to be square size
        frame_resize = cv2.resize(frame_trim, (64, 64), interpolation = cv2.INTER_AREA).transpose(2, 0, 1)  # Resize (*,*,3) -> (3,64,64)

        #   Processing
        Input.d = np.expand_dims(frame_resize, axis=0)  # Add axis to match input (3,64,64) -> (1,3,64,64)
        Output.forward()

        #   Storing Score
        score = np.max([min([Output.d[0][0] / 9 * 100, 100]), 0] )
        result.append(score)

        #   Averaging Score
        if cnt > Frame_per_calc:

            #   Average Storing Score
            result_ave = (np.average(np.array(result)))
            result_ave = np.max([np.min([magnification * result_ave, 100]), 0])  # fine tuning

            #   Just for check
            # print('  IQA Value  :: {0:.1f}/{1}'.format(result_ave, 100))

            # Initialization
            cnt = 0
            result = []

        cnt += 1

        # v : Start to save video
        if key == ord('v'):
            writer = cv2.VideoWriter('result/video.mp4', fmt, frame_rate, (width, height))
            video_coding = 1

        # t : Stop to save video
        if key == ord('t'):
            video_coding = 0
            try:
                writer.release()
            except:
                pass

        # q : Exit
        if key == ord('q'):
            try:
                writer.release()
            except:
                pass
            break

        #   Display image
        txt_ = 'Score : {0:.0f}%'.format(result_ave)
        cv2.putText(frame, txt_, position, font_type, 1.2, back_color, 5, cv2.LINE_AA)
        cv2.putText(frame, txt_, position, font_type, 1.2, front_color, 1, cv2.LINE_AA)
        Img_disp = cv2.resize(frame, (round(width * 1.5), round(height * 1.5)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('frame', Img_disp)

        #   Save Video
        if video_coding:
            writer.write(frame)

    #   Finish Capturing
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':


    Demo = False         # If demo, set "True".

    Test = False         # If test, set "True". Otherwise training, set "False".


    if not Demo:
        # GPU connection
        ctx = get_extension_context('cudnn', device_id=0)
        nn.set_default_context(ctx)
        #   Train
        if not Test:
            train(settings())
        else:
            test(settings())
    else:
        #   Demo
        demo(settings())
