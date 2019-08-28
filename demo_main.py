#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from six.moves import range

import os
import time
import numpy as np

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
#   Sub Class for Displaying Progress in Console
# -------------------------------------------
class display:

    # Remaining Time Estimation
    class time_estimation:

        def __init__(self, epoch, batch_num):
            self.start = time.time()
            self.epoch = epoch
            self.batch = batch_num
            self.all = batch_num * epoch

        def __call__(self, epoch_num, batch_num):
            amount = (batch_num + 1) + epoch_num * self.batch
            remain = (time.time() - self.start) / amount * (self.all - amount)

            hours, mins = divmod(int(time.time() - self.start), 3600)
            mins, sec = divmod(mins, 60)
            hours_e, mins_e = divmod(int(remain), 3600)
            mins_e, sec_e = divmod(mins_e, 60)

            elapse_time = [int(hours), int(mins), int(sec)]
            remain_time = [int(hours_e), int(mins_e), int(sec_e)]

            return elapse_time, remain_time

    def __init__(self, epoch, batch_num):

        self.tm = self.time_estimation(epoch, batch_num)
        self.batch = batch_num

    def __call__(self, epoch, trial, loss):

        elapse_time, remain_time = self.tm(epoch, trial)
        print('  ---------------------------------------------------')
        print('  [ Epoch  # {0},    Trials  # {1}/{2} ]'.format(epoch + 1, trial + 1, self.batch))
        print('    +  Loss                    = {:.4f}'.format(loss))
        print('    -------------------------')
        print('    +  Elapsed Time            : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(elapse_time))
        print('    +  Expected Remaining Time : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(remain_time))
        print('  ---------------------------------------------------')


# -------------------------------------------
#   Network for IQA
# -------------------------------------------
def network(input, scope="", test=False):
    """
    Define Convolutional DNN
       input:   input image ( size must be 64 by 64 )
       test:    test or not
    """

    with nn.parameter_scope(scope):
        c1 = PF.prelu(PF.convolution(input, 10, (8,8), (2,2), (2,2), name='cnv1'))   # Convolution : 64,64,3 -> 2,32,29
        c2 = F.max_pooling(c1, (2,2), (2,2))                                 # Max Pooling : 2,32,29 -> 2,32,14
        #c2 = PF.batch_normalization(c2, batch_stat=not test, name='bn1')  # Batch Normalization
        c4 = PF.prelu(PF.convolution(c2, 16, (8,8), (2,2), (1,1), name='cnv2'))          # Convolution : 2,32,14 -> 16,15,5
        #c4 = PF.batch_normalization(c4, batch_stat=not test, name='bn2')  # Batch Normalization
        c6 = F.max_pooling(c4, (2,2), (2,2))                                 # Max Pooling : 16,15,5 -> 16,15,1
        #c6 = PF.batch_normalization(c5, batch_stat=not test)                        # Batch Normalization
        c7 = PF.prelu(c6, name='PReLU')                                             # PReLU
        c8 = PF.affine(c7, (100,), name='Affine')                                   # Affine : 16,15,1 -> 142
        c9 = F.relu(c8)                                                     # ReLU
        c10= PF.affine(c9, (1,), name='Affine2')                                    # Affine : 142 -> 1

    return c10


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
    Loss    = F.mean(F.absolute_error(Output, Trues) )       #   Loss Function (Squared Error)
    Loss_test = F.mean(F.absolute_error(Output_test, Trues))

    #   Solver Setting
    solver = S.AMSBound(args.learning_rate)                     #   Adam is used for solver
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
    disp = display(args.epoch, batches.iter)

    #   Load data
    if args.retrain > 0:
        with nn.parameter_scope(Name):
            print('Retrain from {0} Epoch',format(args.retrain))
            nn.load_parameters(os.path.join(args.model_save_path, "network_param_{:04}.h5".format(args.retrain)))
            solver.set_learning_rate(args.learning_rate/np.sqrt(args.retrain))

    ##  Training
    for i in range(args.retrain,args.epoch):

        print('')
        print(' =========================================================')
        print('  Epoch :: {0}/{1}'.format(i + 1, args.epoch))
        print(' =========================================================')
        print('')

        ##  Batch iteration
        for j in range(batches.iter):

            #  Load Batch Data from Training data
            Input.d, Trues.d    = batches.next(j)

            #  Update
            solver.zero_grad()                      #   Initialize
            Loss.forward(clear_no_need_grad=True)   #   Forward path
            Loss.backward(8, clear_buffer=True)     #   Backward path
            solver.scale_grad(1/8)                  #   Magic spell
            solver.weight_decay(8*0.001)             #   Weight Decay for stable update
            solver.update()

        ## Progress
        # Get result for Display
        Input.d, Trues.d = batches_test.next(j)
        Loss_test.forward(clear_no_need_grad=True)

        # Display text
        disp(i, j, Loss_test.d)

        ## Save parameters
        if ((i + 1) % args.model_save_cycle) == 0 or (i + 1) == args.epoch:
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

    disp = display(args.epoch, batches.iter)

    result = []
    for j in range(batches.iter):
        Input.d, Trues.d = batches.next(j)
        Loss_test.forward(clear_no_need_grad=True)
        result.append(Loss_test.d)

    result_ave = np.average(np.array(result))
    disp(args.epoch, batches.iter-1, result_ave)



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


    Demo = True         # If demo, set "True".

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
