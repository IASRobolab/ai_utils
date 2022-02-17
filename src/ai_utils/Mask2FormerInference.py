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


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, display_img, classes=None):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
            display_img (bool): True if you want to print segmentation on image, False otherwise
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        available_classes = self.metadata.stuff_classes
        if classes is None:
            classes = []

        predictions = self.predictor(image)
        panoptic_seg, segments_info = predictions["panoptic_seg"]

        # TODO: control if predicitions could be none
        # format the predicted output in a standard Rollo format
        inference_out = {}
        for sinfo in segments_info:
            current_class = available_classes[sinfo['category_id']]
            current_id = sinfo['id']
            if not classes or current_class in classes:
                if current_class not in inference_out.keys():
                    inference_out[current_class] = {}
                    inference_out[current_class]['masks'] = []
                inference_out[current_class]['masks'].append(
                    (panoptic_seg[0].detach().cpu().numpy() == current_id).astype(int)
                )

        # returns only prediction  if you don't want to display the formatted image
        if  display_img:
            # Convert image from OpenCV BGR format to Matplotlib RGB format.
            image = image[:, :, ::-1]
            visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

            vis_output, vect_semantics = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to(self.cpu_device),
                                                                                  segments_info)
            image = vis_output.get_image()[:, :, ::-1]

        return image, inference_out


class Mask2FormerInference:

    def __init__(self, model_weights, config_file,  display_img=False):
        self.display_img = display_img

        mp.set_start_method("spawn", force=True)
        argv = ["--config-file", config_file, "--opts", "MODEL.WEIGHTS", model_weights]
        parser = get_parser()
        args, _ = parser.parse_known_args(argv)
        # args.opts = "MODEL.WEIGHTS"
        setup_logger(name="fvcore")
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)

        self.demo = VisualizationDemo(cfg)

    def img_inference(self, img, classes=None):
        # predictions, visualized_output = demo.run_on_image(image)

        visualized_output, inference_out  = self.demo.run_on_image(img, self.display_img, classes)

        return visualized_output, inference_out