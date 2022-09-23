
# Yolact++ setup

## Setup
To use Yolact++ in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pth```. \
The weights can be downloaded both from the links in the original Yolact++ repository or on our drive in the
```computer_vision_models/detectors/yolact``` folder.

## Running
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
