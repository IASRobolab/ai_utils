# ai_utils
This package is aimed to contain different wrappers for the AI algorithms used in the labs to speed up development and 
improve research quality.

## Algorithms
- **Image segmentation**
  - Yolact++ (Instance segmentation)
  - Mask2Former (Panoptic segmentation)
- **Classification**
  - MMT
- **Pose detection** 
  - Mediapipe (hand pose)


## How to
- [Install ai_utils](#install-ai_utils)
- [Setup and Run](#setup-and-run)
  - [Yolact++](#yolact++-setup)
  - [Mask2Former](#mask2former-setup)
  - [MMT](#mmt-setup)
  - [Mediapipe](#mediapipe-setup)
- [Install](#algorithms-installation)
  - [Yolact++](#yolact++-installation)
  - [Mask2Former](#mask2former-installation)
  - [MMT](#mmt-installation)
  - [Mediapipe](#mediapipe-installation)
- [Find packacge directory path](#find-package-path)

# Install ai_utils

To install ai_utils on your system clone the repo, open a terminal in the main directory and run the following command:
```
python3 -m pip install .
```
Instead if you want to install the package in "editable" or "develop" mode (to prevent the uninstall/install of the 
package at every pkg chjangings) you have can run the following command:

```
python3 -m pip install -e .
```

# Setup and Run

## Yolact++ setup

### Setup
To use Yolact++ in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pth```. \
The weights can be downloaded both from the links in the original Yolact++ repository or on our drive in the
```computer_vision_models/detectors/yolact``` folder.

### Running
To use yolact in  your python code you need to import the YolactInference class:
``` python
    from YolactInference import YolactInference
```
Then you need to create the YolactInference object instance passing some parameters, such as:
- **model_weights**: [_string_] mandatory\
The path of the neural network weights. In general for our ROS convention the weights should be placed in a
 ```YOUR_PKG_DIR_PATH/weights``` folder.
- **display_img**: [_boolean_] default = False\
if True the classification image is displayed on screen.
- **score_threshold**: [_double_] default = 0.5 \
the object with a confidence threshold less than this parameter are discarded and not passed in the output.
``` python
    yolact = YolactInference(model_weights="/YOUR_WEIGHTS_PATH")
```
Finally, use the img_inference() function to evaluate the image with the Neural Network.
``` python
    inference_dict = yolact.img_inference(input_image)
```
**inputs:**

- _input_image_: [numpy array] mandatory\
the image on which to compute the inference
- _classes_: [list] default = None \
This parameter is used as a filter for the classes that we want in the output inference dictionary.

**outputs:**
- _inference_dict_: [dict] \
a dictionary containing the object inferences found on input image divided by class (Key).


## Mask2Former setup

### Setup
To use Mask2Former in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pth``` and then you need also to move some configuration 
files in a ```YOUR_PKG_DIR_PATH/net_config``` folder. \
Both the weights and the config files can be found on our drive in the 
```computer_vision_models/detectors/mask2former``` folder.

### Running
To use Mask2Former in your python code you need to import the Mask2FormerInference class:
``` python
    from Mask2FormerInference import Mask2FormerInference
```
Then you need to create the YolactInference object instance passing some parameters, such as:
- **model_weights**: [_string_] mandatory\
The path of the neural network weights. In general for our ROS convention the weights should be placed in a
 ```YOUR_PKG_DIR_PATH/weights``` folder.
- **config_file**: [_string_] mandatory\
The path of the config file used to load the neural network. In general for our ROS convention the weights should be placed in a
 ```YOUR_PKG_DIR_PATH/net_config``` folder.
- **display_img**: [_boolean_] default = False\          
if True the classification image is displayed on screen. 

``` python
    mask2former = Mask2FormerInference(model_weights="/YOUR_WEIGHTS_PATH", config_file="/YOUR_CONFIG_PATH" )
```
Finally, use the img_inference() function to evaluate the image with the Neural Network.
``` python
    inference_dict = mask2former.img_inference(input_image, classes_wanted)
```
**inputs:**

- _input_image_: [numpy array] mandatory\
the image on which to compute the inference
- _classes_: [list] default = None \
This parameter is used as a filter for the classes that we want in the output inference dictionary.

**outputs:**
- _inference_dict_: [dict] \
a dictionary containing the object inferences found on input image divided by class (Key).

## MMT setup

### Setup
To use MMT in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS```. \
The weights can be found on our drive in the ```computer_vision_models/classifiers/mmt``` folder.

### Running
To use MMT in your python code you need to import the Reidentificator class:
```
    from Reidentificator import Reidentificator
```
Then you need to create the mmt Reidentificator object instance passing some parameters, such as:
- **model_weights**: [_string_] mandatory\
The path of the neural network weights. In general for our ROS convention the weights should be placed in a
 ```YOUR_PKG_DIR_PATH/weights``` folder.
- **class_target**: [_string_] mandatory\
the class string of the object we want to calibrate and then reidentify
- **display_img**: [_boolean_] default = False\          
if True the Reidentification image is displayed on screen.

``` python
    mmt = Reidentificator(model_weights="/YOUR_WEIGHTS_PATH", class_target="person")
```
To use the Reidentificator you need first to calibrate the instance with a calibration phase calling iteratively the 
calibration function until it will return true.
``` python
    while(not calibrate_person(img, inference_output)):
        img = GET_NEW_IMAGE()
        inference_output = GET_INFERENCE_OF_NEW_IMAGE()
``` 
Finally, use the reidentification function to evaluate the image with the Neural Network.
``` python
    mask = reidentify(img, inference_output)
```

**inputs:**

- _img_: [numpy array] mandatory\
the image on which to compute the reidentification.
- _inference_output_: [dict] mandatory \
The dictionary containing the output of a detector (in our case we use yolact).

**outputs:**
- _mask_: [numpy array] \
The mask of the person reidentified.

## Mediapipe setup

### Setup
You don't need to setup anything to use mediapipe hand pose detector, but if you want to classify some gesture you need
to use an classifier (e.g., an SVM classifier) and in this case you should follow the next step.
To use an SVM hand gesture classifier in your code, you need first to move the weights file in a folder called
```weights``` in your package directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS```. \
The weights can be found on our drive in the ```computer_vision_models/classifiers/mediapipe_hand``` folder.

### Running
To use Mediapipe in your python code you need to import the HandPoseInference class:
``` python
    from ai_utils.HandPoseInference import HandPoseInference
```
Then you need to create the mediapipe HandPoseInference object instance passing some parameters, such as:
- **display_img**: [_boolean_] default = False\
When ```True``` print the detected hand points in an image.
- **static_image_mode**: [_boolean_] default = False\
Set ```True``` if you want to use static images otherwise set ```False``` for dynamic ones (moving images)
- **model_complexity**: [_int_] default = 1\
If 0 uses a simpler and faster model. Is 1 uses a more complex and slow but more precise model (speed is almost 
irrelevant on computers).
- **max_num_hands**: [_int_] default = 2\
Number of max detected hands on image.
- **min_detection_confidence**: [_double_] default = 0.3\
Score threshold used to filter out classification with low confidence.
- **min_tracking_confidence**: [_boolean_] default = 0.3\
Same as above but for hand tracking (it is related to hand movements in consecutive images).
- **flip_image**: [_boolean_] default = True\
If True flip images with respect to Y axis. It is used to obtain correct Handedness values because if the input image is
flipped the algorith confuses right and left hands.
- **flatten**: [_boolean_] default = True\
If True the classification output is flattened in a single list composed by 63 values, otherwise a list of 3D points
is returned (size: 21*3)

``` python
    hand_pose = HandPoseInference(display_img=True)
```
To use the mediapipe hand detection use the get_hand_pose function to evaluate the image with the Neural Network..
``` python
    hand_results = hand_pose.get_hand_pose(input_img)
```

**inputs:**

- _img_: [numpy array] mandatory\
the image on which to compute the hands detection.

**outputs:**
- _hands_detected_: [dict] \
A dictionary containing the hands detected on the image. The dict can have only two keys: left and right depending on 
the detected hands handedness. Each key has a list of hands with a format which depends on the instance initialization
parameters

# Algorithms installation
If you want to setup the following algorithm in your local pip you can run the following commands in your bash depending
on what you need to install. \
Be sure that your python build command is upgraded:
``` commandline
  pip install --upgrade build
```

## Yolact++ installation
First of all, clone the custom yolact++ repository, which contains the setup.py file, in a chosen directory (this should
not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows. Python 3
is needed to run Yolact++.

Install the right pytorch dependences:
``` commandline
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 -f https://download.pytorch.org/whl/torch_stable.html
```
To install Yolact++ on your pip environment, activate your environment (or install directly on system if you prefer)
and run:
``` commandline
  cd YOUR_YOLACT_PATH/yolact
  pip install -e .
```

Moreover, to use Yolact++ you need to install DCNv2. There exists two versions of DCNv2 in the Yolact++ repository 
(DCNv2 and DCNv2_latest directories). Choose the version you need to use:
- DCNv2 is used for OLDER GPU architectures (compatible with older pytorch version)
- DCNv2_latest is used for NEWER GPU architectures (compatible with latest pytorch version)

To install it, substitute YOUR_DCNv2_FOLDER with DCNv2 or DCNv2_latest in the following:
``` commandline
  cd YOUR_YOLACT_PATH/yolact/external/YOUR_DCNv2_FOLDER
  pip install -e .
```


## Mask2Former installation
First, clone the custom Mask2Former repository, which contains the setup.py file, in a chosen directory 
(this should not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install Mask2Former on your pip environment, activate your environment (or install directly on system if you prefer).
You need to clone also the detectron2 repository slightly modified by the Robolab Leonardo fellows for Mask2Former.

To setup your environment, run:
```  commandline
  # Detectron2 installation (you need to clone the Robolab repository first)
  cd YOUR_DETECTRON2_PATH/detectron2
  pip install -e .
  pip install git+https://github.com/cocodataset/panopticapi.git
  pip install git+https://github.com/mcordts/cityscapesScripts.git
  
  # Mask2Former installation (you need to clone the Robolab repository first)
  cd YOUR_Mask2Former_PATH/Mask2Former
  pip install -r requirements.txt
  cd mask2former/modeling/pixel_decoder/ops
  sh make.sh
  cd YOUR_Mask2Former_PATH/Mask2Former
  pip install -e .
```


## MMT installation
First of all, clone the custom MMT repository, which contains the setup.py file, in a chosen directory (this should not 
change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install MMT on your pip environment, activate your environment (or install directly on system if you prefer) and run:
``` commandline
  cd YOUR_MMT_PATH/MMT
  python setup.py install
```

## Mediapipe installation

To install Mediapipe on your pip environment, activate your environment (or install directly on system if you prefer) and run:
``` commandline
  pip install mediapipe
```
Google has done a good job; the installation is really simple, isn't it?.

# Find package path
If you are using a standard Cmake package configuration you should save your network weights or config files 
inside your package under a custom directory ```e.g., YOUR_PACKAGE_PATH/weights/YOUR_WEIGHT)```. \
You'll probably need your package path in your code to use the AI algorithms (e.g., to load neural network weights).
If the files are saved as has been declared above you can use the following command to retrieve the package path 
``` python
    import os
    pkg_dir_name = '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-2])
```








