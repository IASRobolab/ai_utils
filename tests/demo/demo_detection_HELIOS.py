import math
import os
import pdb
import sys

from pathlib import Path
yolact_path = str(Path.home()) + '/Documents/yolact'
sys.path.append(yolact_path)

camera_path = str(Path.home()) + '/Documents/camera_utils'
sys.path.append(camera_path)

ai_path = str(Path.home()) + '/Documents/ai_utils/src'
sys.path.append(ai_path)

from camera_utils.cameras.Helios import Helios

from ai_utils.YolactInference import YolactInference

import numpy as np
import cv2
import time
import argparse
import open3d as o3d

parser = argparse.ArgumentParser(
    description='Yolact Inference')
parser.add_argument('--camera_type', default='ZED', type=str,
                    help='RGBD camera')
args = parser.parse_args()

if __name__ == '__main__':
    yolact_weights_new = str(Path.home()) + "/Documents/robotic_arms_vision/weights/yolact_plus_resnet50_54_800000.pth"#yolact_plus_resnet50_drill_74_750.pth"#yolact_plus_resnet50_54_800000.pth"
    #yolact_plus_resnet50_box_penv_plenv_AI4M_79_960.pth"
    #
    #yolact_weights_new = str(Path.home()) + "/Documents/robotic_arms_vision/weights/yolact_plus_resnet50_valve_39_520.pth"
    yolact_new = YolactInference(model_weights=yolact_weights_new, score_threshold=0.8, display_img=True)

    camera=Helios()
    intrinsics = camera.get_intrinsics()
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(intrinsics['width'], intrinsics['height'], intrinsics['fx'], intrinsics['fy'], intrinsics['px'], intrinsics['py'])
    prove = 0 #10000
    somma = 0
    #for i in range(prove):
    while True:
        start_time = time.time()
        prove+=1
        #img = camera.get_rgb()
        img, depth=camera.get_frames()
        img = np.array(img)
        

        #cv2.namedWindow("immagine", cv2.WINDOW_NORMAL)
        #cv2.imshow("immagine", img)
        img=np.stack((img,)*3, axis=-1)
        yolact_infer = yolact_new.img_inference(img)
        masks = yolact_infer["person"]['masks']
        rgb_new = img.copy()
        depth = depth * masks[0]
        #pdb.set_trace()
        for i in range(3):
          rgb_new[:,:,i] = rgb_new[:,:,i] * masks[0]
        depth = o3d.geometry.Image(depth)
        rgb_new = o3d.geometry.Image(rgb_new)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_new, depth)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
  
        o3d.visualization.draw_geometries([pcd])
        print(1/(time.time()-start_time))
        somma += (time.time()-start_time)

    mean_time = somma/prove
    freq = 1/mean_time

    print("mean time: %.4f \nfrequency: %.4f \n" % (mean_time, freq))




