import pdb
import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import sys
camera_path = str(Path.home()) + '/Documents/camera_utils'
sys.path.append(camera_path)

ai_path = str(Path.home()) + '/Documents/ai_utils'
sys.path.append(ai_path)

import time
from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.detectors.Yolov8Inference import Yolov8Inference
import pdb

if __name__ == '__main__':

    camera1 = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD, serial_number='023322062736' )
    #camera2 = IntelRealsense(camera_resolution=IntelRealsense.Resolution.HD, serial_number='023322061667') 
    yolo_weights = str(Path.home()) + "/Downloads/yolov8x-seg.pt"
    yolo = Yolov8Inference(model_weights=yolo_weights, score_threshold=0.5, display_img=False, verbose=False)

    while True:
        
        img1 = camera1.get_rgb()
        start_time = time.time()
        infer1 = yolo.img_inference(img1)
        print('TIME SINGLE PROCESS IMAGE' , 1/(time.time()-start_time))
        
        img1 = camera1.get_rgb()
        img2 = camera1.get_rgb()
        start_time = time.time()
        infer1 = yolo.img_inference(img1)
        infer2 = yolo.img_inference(img2)
        print('TIME TWO CONSECUTIVE PROCESSED IMAGES' , 1/(time.time()-start_time))
        
        
        imgs=[]
        imgs.append(camera1.get_rgb())
        imgs.append(camera1.get_rgb())
        #pdb.set_trace()
        start_time = time.time()
        infer = yolo.img_inference(imgs)
        print('TIME TWO BATCHEZED PROCESSED IMAGES' , 1/(time.time()-start_time))
        
        
        
        img1 = camera1.get_rgb()
        img2 = camera1.get_rgb()
        img3 = camera1.get_rgb()
        img4 = camera1.get_rgb()
        start_time = time.time()
        infer1 = yolo.img_inference(img1)
        infer2 = yolo.img_inference(img2)
        infer3 = yolo.img_inference(img3)
        infer4 = yolo.img_inference(img4)
        print('TIME FOUR CONSECUTIVE PROCESSED IMAGES' , 1/(time.time()-start_time))
        
        
        imgs=[]
        imgs.append(camera1.get_rgb())
        imgs.append(camera1.get_rgb())
        imgs.append(camera1.get_rgb())
        imgs.append(camera1.get_rgb())
        #pdb.set_trace()
        start_time = time.time()
        infer = yolo.img_inference(imgs)
        print('TIME FOUR BATCHEZED PROCESSED IMAGES' , 1/(time.time()-start_time))
        
