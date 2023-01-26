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
import glob

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.config import CfgNode as CN

import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

import mask2former.maskformer_model
from mask2former import add_maskformer2_config

from ai_utils.detectors.DetectorInterface import DetectorInterface


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Mask2FormerInference(DetectorInterface):

    def __init__(self, model_weights, config_file, score_threshold=0.5, display_img=False, classes_white_list=set()):

        DetectorInterface.__init__(self, display_img=display_img, score_threshold=score_threshold, classes_white_list=classes_white_list)

        mp.set_start_method("spawn", force=True)
        argv = ["--config-file", config_file, "--confidence-threshold", str(self.score_threshold),  "--opts", "MODEL.WEIGHTS", model_weights]
        parser = get_parser()
        args, _ = parser.parse_known_args(argv)
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE

        self.predictor = DefaultPredictor(cfg)

        self.available_classes = self.metadata.stuff_classes


    def img_inference(self, img):

        predictions = self.predictor(img)
        panoptic_seg, segments_info = predictions["panoptic_seg"]

        # TODO: control if predicitions could be none
        # format the predicted output in a standard Rollo format
        inference_out = {}
        for sinfo in segments_info:
            current_class = self.available_classes[sinfo['category_id']]
            current_class = current_class.split(",")[0]
            current_id = sinfo['id']
            if not self.classes_white_list or current_class in self.classes_white_list:
                if current_class not in inference_out.keys():
                    inference_out[current_class] = {}
                    inference_out[current_class]['masks'] = []
                inference_out[current_class]['masks'].append(
                    (panoptic_seg.detach().cpu().numpy() == current_id).astype(int)
                )

        # returns only prediction  if you don't want to display the formatted image
        if self.display_img:
            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            img = img[:, :, ::-1]
            visualizer = Visualizer(img, self.metadata, instance_mode=self.instance_mode)

            vis_output, vect_semantics = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(self.cpu_device),
                                                                                  segments_info)
            img = vis_output.get_image()[:, :, ::-1]

            cv2.namedWindow("Mask2Former", cv2.WINDOW_NORMAL)
            cv2.imshow("Mask2Former", img)

            if cv2.waitKey(1) == ord('q'):
                print("Closed Mask2Former Image Viewer.")
                exit(0)

        return inference_out
