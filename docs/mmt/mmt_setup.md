
# MMT setup

## Setup
To use MMT in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS```. \
The weights can be found on our drive in the ```computer_vision_models/classifiers/mmt``` folder.

## Running
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
