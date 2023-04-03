#!/usr/bin/env python

# system imports
import pickle
import sys
import select
import os
import argparse
import numpy as np
from pathlib import Path

ai_path = str(Path.home()) + '/Documents/yolact'
sys.path.append(ai_path)

mmt_path = str(Path.home()) + '/Documents/MMT'
sys.path.append(mmt_path)

camera_path = str(Path.home()) + '/Documents/camera_utils/src'
sys.path.append(camera_path)

ai_utils_path = str(Path.home()) + '/Documents/ai_utils_new'
sys.path.append(ai_utils_path)


# personal imports
from camera_utils.camera_init import Zed
from ai_utils.detectors.YolactInference import YolactInference
from ai_utils.Reidentificator import Reidentificator
from ai_utils.HandPoseInference import HandPoseInference


parser = argparse.ArgumentParser(
    description='Yolact Inference')
parser.add_argument('--camera_type', default='ZED', type=str,
                    help='RGBD camera')
args = parser.parse_args()


# function used to stop loop functions
def stop_loop(stop_entry: str) -> bool:
    '''
    Used to quit an infinite loop with a char/string entry
    '''
    rlist = select.select([sys.stdin], [], [], 0.001)[0]
    if rlist and sys.stdin.readline().find(stop_entry) != -1:
        return True
    return False


if __name__ == '__main__':

    if args.camera_type=='REALSENSE':
      camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD)
    elif args.camera_type=='ZED':
      camera = Zed(rgb_resolution=Zed.Resolution.HD)
    else:
      sys.exit("Wrong camera type!")

    # load diplay ros params
    display_img_results = True

    # Load detector
    yolact_weights =  "./yolact_plus_resnet50_54_800000.pth"
    yolact = YolactInference(model_weights=yolact_weights, score_threshold=0.35, display_img=display_img_results)

    # Load reidentificator
    mmt_weights = "./old_pytorch_resnet_ibn_REID_feat256_train_msmt17.pth"
    reident = Reidentificator(class_target="person", display_img=display_img_results, model_weights=mmt_weights)

    # load gesture detection model
    hand_pose = HandPoseInference(display_img=display_img_results)
    hand_weights = "./right_hand_model.sav"
    hand_classifier = pickle.load(open(hand_weights, 'rb'))


    stop_char = 'q'  # char used to stop the infinite loops

    # Person calibration loop
    # start_time = time.time()
    while True:
        rgb = camera.get_rgb()
        rgb = np.array(rgb)
        yolact_infer = yolact.img_inference(rgb)
        if reident.calibrate_person(rgb, yolact_infer) or stop_loop(stop_char):
            break



    while not stop_loop(stop_char):
        color_frame = camera.get_rgb()

        # Person detection
        yolact_infer = yolact.img_inference(color_frame)

        # Person reidentification
        reidentified_person = reident.reidentify(color_frame, yolact_infer)

        # if no person re-identified restart detection step
        if reidentified_person is None:
            continue

        reidentified_mask = reidentified_person["masks"]
        reidentified_box = reidentified_person["boxes"]

        hand_img = color_frame.copy()
        hand_img = hand_img[reidentified_box[1]:reidentified_box[3], reidentified_box[0]:reidentified_box[2], :]

        # initialize the prediction class as the last class is the predictor
        gesture_prediction = hand_classifier.n_support_.shape[0] - 1

        hand_results = hand_pose.get_hand_pose(hand_img)
        if hand_results is not None:
            for hand_label in hand_results.keys():
                if hand_label == "Left":
                    continue
                else:
                    gesture_prediction = hand_classifier.predict([hand_results[hand_label]])
                   
        if gesture_prediction == 0:           
            print("OPEN HAND")
        elif gesture_prediction == 1:
            print("CLOSED HAND")
        else:
            print("NOT CLASSIFIED")

