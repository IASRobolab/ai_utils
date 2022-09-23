
# Mediapipe setup

## Setup
You don't need to setup anything to use mediapipe hand pose detector, but if you want to classify some gesture you need
to use an classifier (e.g., an SVM classifier) and in this case you should follow the next step.
To use an SVM hand gesture classifier in your code, you need first to move the weights file in a folder called
```weights``` in your package directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS```. \
The weights can be found on our drive in the ```computer_vision_models/classifiers/mediapipe_hand``` folder.

## Running
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