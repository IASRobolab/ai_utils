
# Mediapipe setup

## Setup
You don't need to setup anything in partciclar to use MiDaS. You need only to move the weights file in a folder called
```weights``` in your package directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS```. \
The weights can be found on our drive in the ```computer_vision_models/depth``` folder.

## Running
To use MiDaS in your python code you need to import the MidasInference class:
``` python
    from ai_utils.MidasInference import MidasInference
```
Then you need to create the MidasInference object instance passing some parameters, such as:
- **model_weights**: [_string_] mandatory\
The path of the neural network weights. In general for our ROS convention the weights should be placed in a ```YOUR_PKG_DIR_PATH/weights``` folder.
- **display_img**: [_boolean_] default = False\
When ```True``` print the estimated monocular depth
- **grayscale**: [_boolean_] default = False\
When ```True``` print the estimated monocular depth in grayscale (only if **display_img** is True)
- **concatenate**: [_boolean_] default = True\
When ```True``` print the estimated monocular depth along with the input rgb image (only if **display_img** is True)
- **display_fps**: [_boolean_] default = False\
When ```True``` print the inference fps on the terminal

``` python
    midas = MidasInference(model_weights="YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pt", display_img=True)
```
To use the midas monocular depth estimation use the get_idepth_image function to evaluate the input rgb image with the Neural Network.
``` python
    idepth = midas.get_idepth_image(input_img)
```

**inputs:**

- _img_: [numpy array] mandatory\
the image on which to compute the inverse depth.

**outputs:**
- idepth: [np.array] \
An numpy image containing the estimated inverse depth map of the input image. To obtain the real depth some transformations should be done. 