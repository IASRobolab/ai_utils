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
    
    yolact_weights_new ='/home/azunino/Documents/robotic_arms_vision/weights/yolact_plus_resnet50_54_800000.pth'
    
    yolact_new = YolactInference(model_weights=yolact_weights_new, score_threshold=0.5, display_img=True)

    if args.camera_type=='REALSENSE':
      camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)
      
    elif args.camera_type=='ZED':
      camera = Zed(camera_resolution=Zed.Resolution.HD)
    else:
      sys.exit("Wrong camera type!")
    prove = 0 
    somma = 0
    
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




