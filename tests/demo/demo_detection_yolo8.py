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
from ai_utils.detectors.Yolov8Inference import Yolov8Inference

import numpy as np
import cv2
import time
import argparse


if __name__ == '__main__':
    
    yolo_weights = str(Path.home()) + "/Downloads/yolov8l-seg.pt"
    yolo = Yolov8Inference(model_weights=yolo_weights, score_threshold=0.5, display_img=True, verbose=False)

    camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)

    prove = 0 #10000
    somma = 0
    
    while True:
        start_time = time.time()
        prove+=1
        img = camera.get_rgb()
        img = np.array(img)

        infer = yolo.img_inference(img)
        print(1/(time.time()-start_time))
        somma += (time.time()-start_time)

    mean_time = somma/prove
    freq = 1/mean_time

    print("mean time: %.4f \nfrequency: %.4f \n" % (mean_time, freq))




