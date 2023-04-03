import math
import os
import pdb
import sys

from pathlib import Path
yolact_path = str(Path.home()) + '/Documents/yolact'
sys.path.append(yolact_path)

camera_path = str(Path.home()) + '/Documents/camera_utils'
sys.path.append(camera_path)

ai_path = str(Path.home()) + '/Documents/ai_utils'
sys.path.append(ai_path)


from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.detectors.YolactInference import YolactInference

import numpy as np
import cv2
import time
import argparse

parser = argparse.ArgumentParser(
    description='Yolact Inference')
parser.add_argument('--camera_type', default='REALSENSE', type=str,
                    help='RGBD camera')
args = parser.parse_args()

if __name__ == '__main__':
    #yolact_weights_new = str(Path.home()) + "/Documents/yolact/weights/yolact_plus_resnet50_boxes_dynamic_69_980.pth"
    yolact_weights_new ='/home/azunino/Documents/robotic_arms_vision/weights/yolact_plus_resnet50_54_800000.pth'
    #yolact_weights_new ='/home/azunino/Documents/yolact/weights/yolact_plus_resnet50_adesubset_classes_58_20000.pth'
#yolact_plus_resnet50_box_penv_plenv_AI4M_79_960.pth"
    #
    #yolact_weights_new = str(Path.home()) + "/Documents/robotic_arms_vision/weights/yolact_plus_resnet50_valve_39_520.pth"
    yolact_new = YolactInference(model_weights=yolact_weights_new, score_threshold=0.5, display_img=True)

    if args.camera_type=='REALSENSE':
      camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)
      #camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD, serial_number='023322062736' )
      #camera = IntelRealsense(rgb_resolution=IntelRealsense.Resolution.HD, serial_number='049122251418' )
    elif args.camera_type=='ZED':
      camera = Zed(rgb_resolution=Zed.Resolution.HD)
    else:
      sys.exit("Wrong camera type!")
    prove = 0 #10000
    somma = 0
    #for i in range(prove):
    while True:
        start_time = time.time()
        prove+=1
        img = camera.get_rgb()
        img = np.array(img)

        infer = yolact_new.img_inference(img)
        
        
        print(1/(time.time()-start_time))
        somma += (time.time()-start_time)

    mean_time = somma/prove
    freq = 1/mean_time

    print("mean time: %.4f \nfrequency: %.4f \n" % (mean_time, freq))




