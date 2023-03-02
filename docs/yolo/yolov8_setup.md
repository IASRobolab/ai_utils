
# Yolo v8 setup

## Setup
To use Yolo v8 in your code, you need first to move the weights file in a folder called ```weights``` in your package
directory, i.e., ```YOUR_PKG_DIR_PATH/weights/YOUR_WEIGHTS.pth```. \
The weights can be downloaded both from the links in the original Yolo v8 repository or on our drive in the
```computer_vision_models/detectors/yolo``` folder.

## Running
To use yolo in  your python code you need to import the Yolov8Inference class:
``` python
    from Yolov8Inference import Yolov8Inference
```
Then you need to create the Yolov8Inference object instance passing some parameters, such as:
- **model_weights**: [_string_] mandatory\
The path of the neural network weights. In general for our ROS convention the weights should be placed in a
 ```YOUR_PKG_DIR_PATH/weights``` folder.
- **display_img**: [_boolean_] default = False\
if True the classification image is displayed on screen.
- **score_threshold**: [_double_] default = 0.5 \
the object with a confidence threshold less than this parameter are discarded and not passed in the output.
``` python
    yolo = Yolov8Inference(model_weights="/YOUR_WEIGHTS_PATH")
```
Finally, use the img_inference() function to evaluate the image with the Neural Network.
``` python
    inference_dict = yolo.img_inference(input_image)
```
**inputs:**

- _input_image_: [numpy array] mandatory\
the image on which to compute the inference

**outputs:**
- _inference_dict_: [dict] \
a dictionary containing the object inferences found on input image divided by class (Key).

## Training on custom dataset for instance segmentation
* Before start labelling, be sure to rename all your image with universal names 
* Create an account in https://app.roboflow.com/
* Inside roboflow Create a new project accordingly to your desired task (e.g. instance segmentation)
* Label all your data with roboflow app. Then you can decide which augmentation you want to use on the data, the train/val/test splits etc.
* Once the labelling is done you can export the structure data as a zip. The structure contains 3 folders (train/val/test) each containing a folder of images and a folder of annotations in the Yolo format (a txt file for each image containing the annotations), and a data.yaml file containing the metadata of the dataset (i.e. the paths of the train/val/test splits and the class names) 
* Run the following block of code to train on your custom dataset: 
``` python
    from ultralytics import YOLO
    model = YOLO(PATH TO THE PRETRAINED MODEL (e.g. './yolov8x-seg.pt')) 
    model.train(data= PATH TO data.yaml OF YOUR CUSTOM DATA, epochs=NEPOCHS, batch=BATCHSIZE)
```

## TensorRT optimization to speed up the inference
``` python
    from ultralytics import YOLO
    model = YOLO(PATH TO THE PRETRAINED MODEL (e.g. './yolov8x-seg.pt')) 
    model.export(format='engine', device=0, half=True)
```
