
# Mask2Former setup

## Setup
To use Mask2Former in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pth``` and then you need also to move some configuration 
files in a ```YOUR_PKG_DIR_PATH/net_config``` folder. \
Both the weights and the config files can be found on our drive in the 
```computer_vision_models/detectors/mask2former``` folder.

## Running
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
