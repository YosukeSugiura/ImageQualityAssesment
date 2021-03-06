# Image Quality Assesment ( with Demo )

This shows the implementation of Image Quality Assesment using DNN with No Reference.

I prepared training/test/demo system for IQA. In demo, this system captures image by web camera and outputs IQA score in real-time.

You can try it on your local PC as well as goole colaboratory.
But **Trainng is possible only on your PC**.   
(Training on google colab is in the future.)

**[Reference Paper]**  
will appear soon

# Overview

## Network Model

This method is based on Convolutional Neural Network (CNN). The structure is shown in the following figure. In this system, the output value takes continuous values. Several convolutional layers have a role in extracting the underlying image features.
![Structure](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/structure.png)
- Activation Function :   Parametric ReLU (Output :  Linear) 

> Notice : Actually, my network model described in `demo_main.py` is a little different with the abobe one for more improvement.

## Training Parameters

The parameters for training are the followings.

-  Training Dataset : TID2013 color images
-  Loss Function : Mean Squared Error
-  Updating Algorithm :  AMSBound  
-  Learning Rate :  0.00005
-  Batch Size : 100  
-  Epoch : 5000

You can change those parameters from `settings.py`.

***
# Demo
***
# Equipment

- **Web Camera**  
   You need web camera to capture photo or video as the input.
   
# How to Execute

You can choose from two ways to execute DNN-based IQA, on Google Colaboratory or your Local PC.   

 >  **1.  Google Colaboratory**  :  for only Demo (one-shot).  
 > 
 >  **2.  Local PC**  :  for Training / Test / Demo (continuous).  

If you want to run **Only Demo**, I recommend you to use Google Colaboratory.

# 1. On Google Colaboratory

You can try demo on goole colaboratory.
Google colab is a very conbenient platform because all the processes are done on the cloud server. 
I show the procedure to execute.

### Demo

 1.  Access [IQA_demo.ipynb](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) and push `Open In Colab` bottun. Or access [this link](https://colab.research.google.com/github/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) directly.  After log-in Google, you can automatically open source code for demo.
 
 2. Before Running, connect web camera to your device.
 
 3. Run all cells. Captured video is displayed.
 
 4. Push 'Capture' bottun on the console. The system executes one-shot evaluation and then displays the IQA result on the console with the captured picture.  
![demo1](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/demo1.png)

> Notice : If not working, you may have to allow the permission of your browser to access the web camera.

# 2. On local PC

You can also demo on your local environment. 
But in this case you have to install some python pachages. 
Please follow the below.

## Requirements

### Python & CUDA

- Python 3.6
- CUDA 10.1 (Refer to [Official Installation Guide](https://developer.nvidia.com/cuda-downloads))
- cuDNN 7.6

### Python Package
At the biggining, I reccomend you to upgrade pip in command prompt by
```
$ pip install --upgrade pip
```
If you get Access Denied Error by the above, try
```
$ pip install --upgrade pip --user
```
and then install the following packages.  
- numpy
- nnabla (>=1.0.20)  
- nnala-ext-cuda (>=1.0.20)  
   -  Install appropriate CUDA and CuDNN in your PC. If you use nnabla-ext-cuda 1.0.20, install CUDA 10.1 and cuDNN 7.6 from NVIDIA.  [Read more.](https://github.com/sony/nnabla/releases)
- joblib
- cv2
   - You can install it by the following command.
   ```
   $ pip install opencv-python
   ```
## Run ( Training / Test / Demo )

Download all files and run `demo_main.py`.  
According to your purpose, change the following two variables in `demo_main.py` appropriately.  

| Parameters | for Training |for Test|for Demo|
----|----|----|---- 
| `Demo` | False | False | True |
| `Test` | False | True | True/False |

For example, If you want to execute training, please set  
```@python
 Demo = False         
 Test = False         
```

### Demo
 
1. Run `demo_main.py` with demo mode (see above).

2. You can find the continuous IQA score (0% - 100%) in real-time on the display window.

3. If you need, use the following commmands during running:

   - **`v`: Start Video Recording**   
      If you press `v` key, you can start to record the display window as "mp4" video.  
      
   - **`t`: Stop Video Recording**   
      If you press `t` key, you can stop to record the display window as "mp4" video.  
      You can find the saved `video.mp4` file in `result` folder after stop.  
      
   - **`q`: Exit**   
      If you press `q` key, you can exit.

![demo2](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/demo2.png)

> Notice : You can see the demo video from [here](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/result).

### Before Training

1. Make folders `data/image_train/` and `data/image_test/` in the same directory with source codes.

2. Download TID2013 dataset from site([Direct link](http://www.ponomarenko.info/tid2013/tid2013.rar)).   
   TAMPERE IMAGE DATABASE 2013 TID2013, version 1.0  
   http://www.ponomarenko.info/tid2013.htm
   
3. Unzip `tid2013.tar`.

4. Resize all images to 64 x 64.

5. Choose somes image files from `tid2013/distorted_images/` as train data and move them to `data/image_train/`.

6. Copy `mos.txt` in `tid2013/` and paste to `data/image_train/`, and Remove some values which is not corresponding to the selected train data.

7. Do 5. and 6. again for test data.

8. Run demo_main.py with Train mode.

# Notice

### Trained Network Model
The trained network is already stored in params folder ( You can see from [here](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/params). ).
By default, you do not need to execute training. 

If you want to use the trained network, open `settings.py` and please set
```@python
self.epoch = 5000   
```

### Fine Tuning 

1. In demo, IQA value is very sensitive to the out of focus.  
  If you want to gain the score in demo, adjust camera focus or change the distance between the camera and the object. 

2. Inversely, when you add a blur, the IQA socre is obviously degradaded.
