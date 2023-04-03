
from camera_utils.cameras.IntelRealsense import IntelRealsense
from ai_utils.feature_extractors.MMTExtractor import MMTExtractor
from ai_utils.detectors.Yolov8InferTrack import Yolov8InferTrack

import sys
import select

yolo_weights="/home/frollo/Utils/yolov8_tracking/weights/yolov8l-seg.pt"
reid_weights="/home/frollo/Utils/yolov8_tracking/weights/osnet_x0_25_msmt17.pt"
yolo = Yolov8InferTrack(display_img=True, model_weights=yolo_weights, reid_weights=reid_weights)

camera = IntelRealsense()
mmt = MMTExtractor("/home/frollo/ros_workspaces/ros1/followme_ws/src/followme/followme/weights/old_pytorch_resnet_ibn_REID_feat256_train_msmt17.pth", 'person')

# function used to stop loop functions
def stop_loop(stop_entry):
    '''
    Used to quit an infinite loop with a char/string entry
    '''
    rlist = select.select([sys.stdin], [], [], 0.001)[0]
    if rlist and sys.stdin.readline().find(stop_entry) != -1:
        return True
    return False



while not stop_loop('q'):

    frame = camera.get_rgb()
    infer = yolo.img_inference(frame)
    features = mmt.get_features(frame, infer)

    print(features, end='\n')
    