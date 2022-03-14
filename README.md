# ai_utils
This package is aimed to contain different wrappers for the AI algorithms used in the labs

## Algorithms
- **Detectors**
  - Yolact++ 
  - Mask2Former
- **Classifiers**
  - MMT


## How to
- [Setup and Run](#setup-and-run)
  - [Yolact++](#yolact++-setup)
  - [Mask2Former](#mask2former-setup)
  - [MMT](#mmt-setup)
- [Install](#algorithms-installation)
  - [Yolact++](#yolact++-installation)
  - [Mask2Former](#mask2former-installation)
  - [MMT](#mmt-installation)

# Setup and Run

## Yolact++ setup

### Setup
To use Yolact++ in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pth```. \
The weights can be downloaded both from the links in the original Yolact++ repository or on our drive in the
```computer_vision_models/detectors/yolact``` folder.

### Running
To use yolact in  your python code you need to import the YolactInference class:
```
    from YolactInference import YolactInference
```
Then you need to create the YolactInference object instance passing some parameters, such as:
- **model_weights**: [_string_] mandatory\
The path of the neural network weights. In general for our ROS convention the weights should be placed in a
 ```YOUR_PKG_DIR_PATH/weights``` folder.
- **display**: [_boolean_] default = False\
if True the image will be formatted with the inference output on it.
- **score_threshold**: [_double_] default = 0.5 \
the object with a confidence threshold less than this parameter are discarded and not passed in the output.
```
    yolact = YolactInference(model_weights="/YOUR_WEIGHTS_PATH")
```
Finally, use the img_inference() function to evaluate the image with the Neural Network.
```
    out_image, inference_dict = yolact.img_inference(input_image)
```
**inputs:**

- _input_image_: [numpy array] mandatory\
the image on which to compute the inference
- _classes_: [list] default = None \
This parameter is used as a filter for the classes that we want in the output inference dictionary.

**outputs:**
- _out_image_: [numpy array] \
the image with the inference output on it (only if display = True in yolact instantiation).
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
```
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
if True the image will be formatted with the inference output on it.

```
    mask2former = Mask2FormerInference(model_weights="/YOUR_WEIGHTS_PATH", config_file="/YOUR_CONFIG_PATH" )
```
Finally, use the img_inference() function to evaluate the image with the Neural Network.
```
    out_image, inference_dictionary = mask2former.img_inference(input_image, classes_wanted)
```
**inputs:**

- _input_image_: [numpy array] mandatory\
the image on which to compute the inference
- _classes_: [list] default = None \
This parameter is used as a filter for the classes that we want in the output inference dictionary.

**outputs:**
- _out_image_: [numpy array] \
the image with the inference output on it (only if display = True in yolact instantiation).
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
- **display_target**: [_boolean_] default = False\
if True the reidentificator returns an image on which the person reidentified has a white bounding box around him.

```
    mmt = Reidentificator(model_weights="/YOUR_WEIGHTS_PATH", class_target="person")
```
To use the Reidentificator you need first to calibrate the instance with a calibration phase calling iteratively the 
calibration function until it will return true.
```
    while(not calibrate_person(img, inference_output)):
        img = GET_NEW_IMAGE()
        inference_output = GET_INFERENCE_OF_NEW_IMAGE()
``` 
Finally, use the reidentification function to evaluate the image with the Neural Network.
```
    rgb, mask = reidentify(img, inference_output)
```

**inputs:**

- _img_: [numpy array] mandatory\
the image on which to compute the reidentification.
- _inference_output_: [dict] mandatory \
The dictionary containing the output of a detector (in our case we use yolact).

**outputs:**
- _rgb_: [numpy array] \
the image with the inference output on it (only if display_target = True in MMT instantiation).
- _mask_: [numpy array] \
The mask of the person reidentified.

# Algorithms installation
If you want to setup the following algorithm in your local pip you can run the following commands in your bash depending
on what you need to install. \
Be sure that your python build command is upgraded:
```
  pip install --upgrade build
```


## Yolact++ installation
First of all, clone the custom yolact++ repository, which contains the setup.py file, in a chosen directory (this should
not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows. Python 3
is needed to run Yolact++.

To install Yolact++ on your pip environment, activate your environment (or install directly on system if you prefer)
and run:
```
  cd YOUR_YOLACT_PATH/yolact
  python setup.py build develop
```

Moreover, to use Yolact++ you need to install DCNv2. There exists two versions of DCNv2 in the Yolact++ repository 
(DCNv2 and DCNv2_latest directories). Choose the version you need to use:
- DCNv2 is used for OLDER GPU architectures (compatible with older pytorch version)
- DCNv2_latest is used for NEWER GPU architectures (compatible with latest pytorch version)

To install it, substitute YOUR_DCNv2_FOLDER with DCNv2 or DCNv2_latest in the following:
```
  cd YOUR_YOLACT_PATH/yolact/external/YOUR_DCNv2_FOLDER
  python setup.py build develop
```


## Mask2Former installation
First of all, clone the custom Mask2Former repository, which contains the setup.py file, in a chosen directory 
(this should not change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install Mask2Former on your pip environment, activate your environment (or install directly on system if you prefer).
You need to clone also the detectron2 repository slightly modified by the Robolab Leonardo fellows for Mask2Former.

To setup your environment, run:
```
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
  python setup.py build develop
```


## MMT installation
First of all, clone the custom MMT repository, which contains the setup.py file, in a chosen directory (this should not 
change after installation). This repository has been slightly modified by the Robolab Leonardo fellows.

To install MMT on your pip environment, activate your environment (or install directly on system if you prefer) and run:
```
  cd YOUR_MMT_PATH/MMT
  python setup.py install
```
