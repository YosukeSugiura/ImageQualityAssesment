# Image Quality Assesment Demo

Implementation of Image Quality Assesment using DNN with No Reference.  
You can try it on local PC as well as goole colaboratory.
But **trainng is possible only on your PC**.   
(Training on google colab is in the future.)

**Reference Paper**  
will appear soon

# Method

## Network Model

This method is based on Convolutional Neural Network (CNN). The structure is shown in the following figure.  
![Structure](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/structure.png)
- Activation Function :   Parametric ReLU (Output :  Linear) 

> Actually, my network model described in `demo_main.py` is a little different with the abobe one for more improvement.

## Training

The parameters for training are the followings.

-  Loss Function : Mean Squared Error
-  Updating Algorithm :  AMSBound  
-  Learning Rate :  0.00005
-  Batch Size : 100  
-  Epoch : 5000

You can change those parameters from `settings.py`.

***
# How to Execute

You can choose two ways to execute DNN-based IQA, on **Google Colaboratory** or **your Local PC**.   

If you want to run **Only Demo**, I recommend you to use Google Colaboratory.

# On Google Colaboratory

You can demo on goole colaboratory.
Google colab is very conbinient because all the processes are done on the cloud server. 
You can open source code on google colab in two ways.  

  - Access [IQA_demo.ipynb](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) 
and push `Open In Colab` bottun.

   - Access [this link](https://colab.research.google.com/github/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) directly. 
   
After log-in Google, you can open source code for demo automatically.


# On local PC

You can demo on your local environment. 
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
## Training & Test & Demo

**The trained network is already stored in params folder.**  
( You can see from [here](https://github.com/YosukeSugiura/ImageQualityAssesment/tree/master/params) . )

If you want to use it, check `settings.py` and please set
```@python
self.epoch      = 5000   
```


Download all files and run `demo_main.py`.  
Please change the following variables in `demo_main.py` appropriately according to your purpose.  

### Training

If you want to execute training, please set  
```@python
 Demo = False         
 Test = False         
```
You can change the training parameters from `settings.py`.

### Test

If you want to execute test, please set  
```@python
 Demo = False         
 Test = True         
```

### Demo

If you want to execute test, please set  
```@python
 Demo = True         
 Test = False
```
