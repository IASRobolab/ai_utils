from ai_utils.YolactInference import YolactInference
import cv2
from pathlib import Path
import os

##### ! change the weights folder below with your paths ! #####
weights_folder = str(Path.home()) + "/ros_workspaces/ros1/navigation_ws/src/vision_controller/weights"
##### ! ############################################### ! #####

# Neural Network initialization 
yolact = YolactInference(
    model_weights = weights_folder + "/yolact_plus_resnet50_54_800000.pth"
)

# ADD TESTS
def test_add_single_class_in_empty_set():
    yolact.add_class("cacca")
    assert yolact.classes_white_list == {"cacca"}

def test_add_single_class_in_set_with_same_class():
    yolact.add_class("cacca")
    assert yolact.classes_white_list == {"cacca"}

def test_add_list():
    yolact.add_class(["prova", "lista"])
    assert yolact.classes_white_list == {"cacca", "prova", "lista"}

def test_add_set():
    yolact.add_class({"try", "set"})
    assert yolact.classes_white_list == {"cacca", "prova", "lista", "try", "set"}

def test_add_list_in_set_with_same_class():
    yolact.add_class("cacca")
    assert yolact.classes_white_list == {"cacca", "prova", "lista", "try", "set"}

def test_set_in_set_with_same_class():
    yolact.add_class("cacca")
    assert yolact.classes_white_list == {"cacca", "prova", "lista", "try", "set"}

# REMOVE TESTS
def test_remove_class_from_set_where_present():
    yolact.remove_class("cacca")
    assert yolact.classes_white_list == {"prova", "lista", "try", "set"}

def test_remove_list_from_set_where_present():
    yolact.remove_class(["prova", "lista"])
    assert yolact.classes_white_list == {"try", "set"}

def test_remove_set_from_set_where_present():
    yolact.remove_class({"try", "set"})
    assert yolact.classes_white_list == set()

def test_remove_class_from_set_where_not_present():
    yolact.remove_class("cacca")
    assert yolact.classes_white_list == set()

def test_remove_list_from_set_where_not_present():
    yolact.remove_class(["prova", "lista"])
    assert yolact.classes_white_list == set()

def test_remove_set_from_set_where_not_present():
    yolact.remove_class({"try", "set"})
    assert yolact.classes_white_list == set()

# CHECK IF SET FUNCTION ARE WORKING
def test_set_is_empty():
    assert not yolact.classes_white_list

def test_class_not_in_white_list():
    assert not ("cacca" in yolact.classes_white_list)

def test_set_is_not_empty():
    yolact.add_class("cacca")
    assert yolact.classes_white_list

def test_class_not_in_white_list():
    assert "cacca" in yolact.classes_white_list


def test_clear_white_list():
    yolact.add_class(["prova", "lista"])
    yolact.add_class({"try", "set"})
    yolact.empty_white_list()
    assert yolact.classes_white_list == set()

# TEST IMAGE PREDICTION FILTERING
image = cv2.imread("tests/test_image.png")

def test_prediction_with_no_filtering():
    pred = yolact.img_inference(image)
    print(pred.keys())
    assert list(pred.keys()) == ['person', 'chair', 'suitcase', 'laptop']

def test_prediction_with_filtering_if_class_not_present():
    yolact.add_class("cacca")
    pred = yolact.img_inference(image)
    assert list(pred.keys()) == list()

def test_prediction_with_filtering_if_class_is_present():
    yolact.add_class("person")
    pred = yolact.img_inference(image)
    assert list(pred.keys()) == ['person']

def test_prediction_with_filtering_if_class_is_removed():
    yolact.remove_class("person")
    pred = yolact.img_inference(image)
    assert list(pred.keys()) == list()

def test_prediction_with_filtering_if_multiple_class_added():
    import collections
    yolact.add_class(["person", "chair"])
    pred = yolact.img_inference(image)
    assert collections.Counter(list(pred.keys())) == collections.Counter(['person', 'chair'])

def test_prediction_with_filtering_if_class_removed_but_white_list_not_empty():
    yolact.remove_class("person")
    pred = yolact.img_inference(image)
    assert list(pred.keys()) == ['chair']

def test_prediction_with_filtering_when_multiple_classes_are_removed_and_white_list_empty():
    yolact.remove_class(["person", "chair", "cacca"])
    print(yolact.classes_white_list)
    pred = yolact.img_inference(image)
    assert list(pred.keys()) == ['person', 'chair', 'suitcase', 'laptop']
