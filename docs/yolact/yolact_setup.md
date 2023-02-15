
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

**outputs:**
- _inference_dict_: [dict] \
a dictionary containing the object inferences found on input image divided by class (Key).

## Training on custom dataset
* Before start labelling, be sure to rename all your data with universal names 
* Label all your data with labelme (https://www.dlology.com/blog/how-to-create-custom-coco-data-set-for-instance-segmentation/) tool. It creates a json file for each image with its annotations
* From the yolact repo, run labelme2coco.py script to generate the coco annotation (test and train) json file which include all your annotated data
* Modify the train/test json files created and be sure that the the first category_id starts with 1 and not 0. Rename all the category_id in the file by adding 1 to all entries 
* Add the definition of the custom dataset and the network inside the data/config.py of yolact repo. In the section DATASETS of the config file add a block of code 
```
my_custom_dataset = dataset_base.copy({
    'name': 'My Dataset',

    'train_images': 'path_to_training_images',
    'train_info':   'path_to_training_annotation',

    'valid_images': 'path_to_validation_images',
    'valid_info':   'path_to_validation_annotation',

    'class_names': ('my_class_id_1', 'my_class_id_2', 'my_class_id_3', ...)
})
```
We are just overwriting some variables form “dataset_base”, so make sure your custom dataset definition comes after that. In the section YOLACT++ CONFIGS of the config file add a block of code 
```
yolact_plus_resnet50_CUSTOM_DATASET_config = yolact_plus_resnet50_config.copy({
    'name': 'yolact_plus_resnet50_CUSTOM_DATASET',
    # Dataset stuff
    'dataset': my_custom_dataset,
    'num_classes': len(my_custom_dataset.class_names) + 1,
})
```
Again, we are overwriting some variables from “yolact_resnet50_config”, so make sure your custom config comes after that
* Finally, to train the network on your customized dataset run the following script from the yolact repo
``` python
    python train.py --config=yolact_plus_resnet50_CUSTOM_DATASET_config
```
* Refer to this link for additional help: https://www.immersivelimit.com/tutorials/train-yolact-with-a-custom-coco-dataset
