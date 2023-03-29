#!/usr/bin/env python
'''----------------------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2022
#
# author: Federico Rollo, Andrea Zunino
# mail: rollo.f96@gmail.com
#
# Institute: Leonardo Labs (Leonardo S.p.a - Istituto Italiano di tecnologia)
#
# This file is part of ai_utils. <https://github.com/IASRobolab/ai_utils>
#
# ai_utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ai_utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License. If not, see http://www.gnu.org/licenses/
---------------------------------------------------------------------------------------------------------------------------------'''
import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import logging
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from yolov8.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
# from yolov8.ultralytics.yolo.utils import DEFAULT_CONFIG, LOGGER, SETTINGS, callbacks, colorstr, ops
from yolov8.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from yolov8.ultralytics.yolo.utils.files import increment_path
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors
from yolov8.ultralytics.yolo.data.augment import LetterBox

from trackers.multi_tracker_zoo import create_tracker
import trackers
from ai_utils.detectors.DetectorInterface import DetectorInterface

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', default=True, help='whether to plot masks in native resolution')
    opt, unknown = parser.parse_known_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = Path(trackers.__file__[:-11] + opt.tracking_method + '/configs/' + opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt
    



class Yolov8InferTrack(DetectorInterface):

    def __init__(self,  model_weights, reid_weights, display_img=False, score_threshold=0.5, max_det=15, classes_white_list=set(), argv=None):
    
        DetectorInterface.__init__(self, display_img=display_img, score_threshold=score_threshold, classes_white_list=classes_white_list)
        
        self.args = parse_args(argv)
        # Load model
        self.device = select_device(self.args.device)
        self.args.conf_thres = score_threshold
        self.args.max_det = max_det
        self.is_seg = '-seg' in str(model_weights)
        self.args.yolo_weights = Path(model_weights)
        self.args.reid_weights = Path(reid_weights)
        self.model = AutoBackend(model_weights, device=self.device, dnn=self.args.dnn, fp16=self.args.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))  # warmup
        # Create as many strong sort instances as there are video sources
        self.tracker = create_tracker(self.args.tracking_method, self.args.tracking_config, self.args.reid_weights, self.device, self.args.half)


    @torch.no_grad()
    def output_formatting_and_display(self, dets_out, im0s, im):
        
        det = dets_out[0]
        outputs = None
        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=self.args.line_thickness, example=str(self.names))
   
        if det is not None and len(det):
                if self.is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if self.args.retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        self.masks = process_mask_native(self.proto[0], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        self.masks = process_mask(self.proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    
                # pass detections to strongsort
                with self.dt[3]:
                    outputs = self.tracker.update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    
                    if self.is_seg:
                        # Mask plotting
                        annotator.masks(
                            self.masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if self.args.retina_masks else im[0]
                        )
                     
                    for j, (output) in enumerate(outputs):
                        
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        if self.display_img:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if self.args.hide_labels else (f'{id} {self.names[c]}' if self.args.hide_conf else \
                                (f'{id} {conf:.2f}' if self.args.hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                              
            # Stream results
            
        if self.display_img:
                im0 = annotator.result()
                cv2.namedWindow("YOLO TRACKING", cv2.WINDOW_NORMAL)
                cv2.imshow('YOLO TRACKING', im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()
                    
        ## Block of code for alligning the tracked objects with the detected masks. For each tracked objects we return the corresponding
        ## mask (the one which has the nearest distance between the bbox tracked and detected) detected in the current frame, if present.
        if self.is_seg:
          detection_bboxs = det[:, :4].detach().cpu().numpy()
          masks_track = []
          outputs_track = []
          if outputs is not None and len(outputs) > 0:
            for out in outputs:
              tracking_bbox = out[:4]
              for idx, detection_bbox in enumerate(detection_bboxs):
                dist=np.linalg.norm(detection_bbox - tracking_bbox, axis=0)
                if dist < 10: ## euclidean distance threshold chosen for bounding boxes overlapping between predictions and current detections  
                   masks_track.append(self.masks[idx].detach().cpu().numpy())
                   if not len(outputs_track):
                     outputs_track = np.expand_dims(out, axis=0)
                   else:
                     outputs_track=np.vstack((outputs_track, out))
                   break
          return outputs_track, masks_track
        ##
        else:
          return outputs, None
        
        
    @torch.no_grad()
    def img_inference(self, rgb):
        
        self.dt = (Profile(), Profile(), Profile(), Profile())
        im0s=rgb
        im=np.stack(LetterBox(self.imgsz, self.pt, stride=self.stride)(image=im0s))
        if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)  # contiguous
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.args.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            preds = self.model(im, augment=self.args.augment, visualize=False)

        # Apply NMS
        with self.dt[2]:
            if self.is_seg:
                self.masks = []
                p = non_max_suppression(preds[0], self.args.conf_thres, self.args.iou_thres, self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det, nm=32)
                self.proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, self.args.conf_thres, self.args.iou_thres, self.args.classes, self.args.agnostic_nms, max_det=self.args.max_det)
        inference, masks = self.output_formatting_and_display(p, im0s, im)
     
        inference_out = {}
        if len(inference) > 0: 
            for idx, cls in enumerate(inference[:,5]):
                cls = self.names[int(cls)]
                # expression which evaluates if self.classes_white_list is empty OR the current class is in the white list
                if not self.classes_white_list or cls in self.classes_white_list:
                        if cls not in inference_out.keys():
                            inference_out[cls] = {}
                            inference_out[cls]['scores'] = []
                            inference_out[cls]['boxes'] = []
                            inference_out[cls]['id'] = []
                            inference_out[cls]['masks'] = []
                        inference_out[cls]['scores'].append(inference[idx][6])
                        inference_out[cls]['boxes'].append(inference[idx][0:4])
                        inference_out[cls]['id'].append(inference[idx][4])
                        if self.is_seg: 
                          inference_out[cls]['masks'].append(masks[idx])
                          
        
       
        return inference_out
