# Image Quality Assesment ( with Demo )

**Implementation of Image Quality Assesment using DNN with No Reference.**

In demo, this system captures image by web camera and outputs IQA score in real-time.

You can try it on local PC as well as goole colaboratory.
But **trainng is possible only on your PC**.   
(Training on google colab is in the future.)

**[Reference Paper]**  
will appear soon

# Overview

## Network Model

This method is based on Convolutional Neural Network (CNN). The structure is shown in the following figure. In this system, the output value takes continuous values. Several convolutional layers have a role in extracting the underlying image features.
![Structure](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/structure.png)
- Activation Function :   Parametric ReLU (Output :  Linear) 

> Actually, my network model described in `demo_main.py` is a little different with the abobe one for more improvement.

## Training

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

You can choose two ways to execute DNN-based IQA, on Google Colaboratory or your Local PC.   

 >  **Google Colaboratory**  :  for only Demo (one-shot).  
 > 
 >  **Local PC**  :  for Training / Test / Demo (continuous).  

If you want to run **Only Demo**, I recommend you to use Google Colaboratory.

# 1. On Google Colaboratory

You can demo on goole colaboratory.
Google colab is very conbinient because all the processes are done on the cloud server. 
I show the procedure to execute.

### Demo

 1.  Access [IQA_demo.ipynb](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) and push `Open In Colab` bottun. Or access [this link](https://colab.research.google.com/github/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) directly.  After log-in Google, you can automatically open source code for demo.
 
 2. Before Running, connect web camera to your device.
 
 3. Run all cells. Captured video is displayed.
 
 4. Push 'Capture' bottun on the output console. The system executes one-shot evaluation and then displays the IQA result on the console with the captured picture.  
![demo1](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/demo1.png)

# 2. On local PC

You can also demo on your local environment. 
But in this case you have to install some python pachages. 
Please follow the below.

## Requirements

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
Please change the following variables in `demo_main.py` appropriately according to your purpose.  

| Parameters | for Training |for Test|for Demo|
----|----|----|---- 
| **Demo** | False | False | True |
| **Test** | False | True | True/False |

For example, If you want to execute training, please set  
```@python
 Demo = False         
 Test = False         
```

### Demo
 
When you start demo, the system executes continuous evaluation.
You can find real-time IQA score at upper left of the window.

![demo2](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/demo2.png)

You can see the demo video from [here](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/result).


# Notice

### Training Network Model
The trained network is already stored in params folder. ( You can see from [here](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/params). )
By default, you do not need to execute training.

If you want to use the trained network, check `settings.py` and please set
```@python
self.epoch      = 5000   
```

### Fine Tuning 

In demo, IQA value is very sensitive to the out of focus. 
If you want to gain the score in demo, adjust camera focus or change the distance of the object. 

Inversely, when you add a blur, the IQA socre is obviously degradaded.
