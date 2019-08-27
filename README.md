# Image Quality Assesment Demo

Implementation of Image Quality Assesment using DNN with No Reference.  
You can do demo not only on your local PC but also on goole colaboratory.

**Reference Paper**  
not abailable




***

# On Google Colaboratory

You can demo on goole colaboratory.
Google colab is very conbinient because we do not need to choose the execution environment. 
You can open source code on google colab in two ways.  

  - Access [IQA_demo.ipynb](https://github.com/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb) 
and push `Open In Colab` bottun.

   - You can open iPython code directly from [this link](https://colab.research.google.com/github/YosukeSugiura/ImageQualityAssesment/blob/master/IQA_demo.ipynb). After log-in Google, code for demo would be opened automatically.

# On local PC

You can demo on your local environment. 
But in this case you have to install some python pachages. 
Please follow the below.

## Requirements

At the biggining, I reccomend you to upgrade pip in command prompt by
```
pip install --upgrade pip
```
or, if you get Access Denied Error, try
```
pip install --upgrade pip --user
```
and then install the following packages.  
- numpy
- nnabla (>=1.0.20)  
- nnala-ext-cuda (>=1.0.20)  
   -  Install appropriate CUDA and CuDNN in your PC. If you use nnabla-ext-cuda 1.0.20, install CUDA 10.1 and cuDNN 7.6 from NVIDIA.  [Read more.](https://github.com/sony/nnabla/releases)
- joblib
- cv2
   - You can install it by running the following command in command prompt.
   ```python
   pip install opencv-python
   ```
## Training & Test

The trained network is already prepared in params folder

Download all files and run `demo_main.py`.  
Please change source code according to your purpose.  

### Training

If you want to execute training, please set  
```@python
 Demo = False         
 Test = False         
```

### Test

If you want to execute test, please set  
```@python
 Demo = False         
 Test = True         
```

## Demo

If you want to execute test, please set  
```@python
 Demo = False         
 Test = True         
```
