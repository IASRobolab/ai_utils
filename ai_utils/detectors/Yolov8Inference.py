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
from ai_utils.detectors.DetectorInterface import DetectorInterface
from ultralytics.yolo.engine.model import YOLO

class Yolov8Inference(DetectorInterface):

    def __init__(self, model_weights, display_img=False, score_threshold=0.5, classes_white_list=set(), verbose=False) -> None:
        DetectorInterface.__init__(self, display_img=display_img, score_threshold=score_threshold, classes_white_list=classes_white_list)

        self.model = YOLO(model_weights)
        self.verbose = verbose


    def img_inference(self, rgb):


        results = self.model.predict(rgb, show=self.display_img, verbose=self.verbose, retina_masks=True)
        results = list(results)[0]

        boxes = results.boxes.xywh.tolist()    
        masks = results.masks.masks.tolist()
        classes = results.boxes.cls.tolist() 
        scores = results.boxes.conf.tolist()

        inference_out = {}
        if results is not None:
            for idx, cls in enumerate(classes):
                cls = self.model.names[cls]
                if scores[idx] > self.score_threshold:
                    # expression which evaluates if self.classes_white_list is empty OR the current class is in the white list
                    if not self.classes_white_list or cls in self.classes_white_list:
                        if cls not in inference_out.keys():
                            inference_out[cls] = {}
                            inference_out[cls]['scores'] = []
                            inference_out[cls]['boxes'] = []
                            inference_out[cls]['masks'] = []
                        inference_out[cls]['scores'].append(scores[idx])
                        inference_out[cls]['boxes'].append(boxes[idx])
                        inference_out[cls]['masks'].append(masks[idx])

        return inference_out
