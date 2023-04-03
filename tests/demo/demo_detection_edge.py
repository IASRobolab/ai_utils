import math
import os
import pdb
import sys

from pathlib import Path
yolact_path = str(Path.home()) + '/Desktop/yolact_edge'
sys.path.append(yolact_path)

camera_path = str(Path.home()) + '/Documents/camera_utils'
sys.path.append(camera_path)

ai_path = str(Path.home()) + '/Documents/ai_utils'
sys.path.append(ai_path)


from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.detectors.YolactEdgeInference import YolactEdgeInference

import numpy as np
import cv2
import time
import argparse


if __name__ == '__main__':
    
    yolact_weights_new = str(Path.home()) + "/Downloads/yolact_edge_54_800000.pth"
    
    yolact_new = YolactEdgeInference(model_weights=yolact_weights_new, score_threshold=0.4, display_img=True)


    camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)


    while True:
        start_time = time.time()
        
        img = camera.get_rgb()
        img = np.array(img)
        

        

        yolact_infer = yolact_new.img_inference(img)

        
        
        print(1/(time.time()-start_time))
        

    




