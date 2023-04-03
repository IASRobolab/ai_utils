import math
import os
import pdb
import sys

from pathlib import Path
ai_path = str(Path.home()) + '/Documents/Mask2Former'
sys.path.append(ai_path)

camera_path = str(Path.home()) + '/Documents/camera_utils'
sys.path.append(camera_path)

ai_path = str(Path.home()) + '/Documents/ai_utils'
sys.path.append(ai_path)


from camera_utils.cameras.IntelRealsense import IntelRealsense

from camera_utils.camera_init import Zed
from ai_utils.detectors.Mask2FormerInference import Mask2FormerInference

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
    mask2former_weights = "./mask2former.pkl"
    mask2former_new = Mask2FormerInference(model_weights=mask2former_weights, config_file="/home/azunino/Documents/Mask2Former/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml" , display_img=True)
    
    if args.camera_type=='REALSENSE':
      camera = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD)
    elif args.camera_type=='ZED':
      camera = Zed(rgb_resolution=Zed.Resolution.HD)
    else:
      sys.exit("Wrong camera type!")
    somma = 0
    prove = 0
    while True:
        start_time = time.time()
        prove+=1
        img = camera.get_rgb()
        img = np.array(img)
        

        #cv2.namedWindow("immagine", cv2.WINDOW_NORMAL)
        #cv2.imshow("immagine", img)

        infer = mask2former_new.img_inference(img)

        
        somma += (time.time()-start_time)

    mean_time = somma/prove
    freq = 1/mean_time

    print("mean time: %.4f \nfrequency: %.4f \n" % (mean_time, freq))




