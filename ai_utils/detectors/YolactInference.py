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
from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
from data import cfg, set_cfg
import torch
import torch.backends.cudnn as cudnn
import argparse
from collections import defaultdict
import cv2
from ai_utils.detectors.DetectorInterface import DetectorInterface
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.set_defaults(mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False)

    args, unknown = parser.parse_known_args()

    return args


color_cache = defaultdict(lambda: {})


class YolactInference(DetectorInterface):

    def __init__(self, model_weights, display_img=False, return_img=False, score_threshold=0.5, top_k=15, classes_white_list=set(), argv=None):
        '''
        Yolact detector used to classify, detect and segment objects on an image
        :param display_img: boolean used to return the results plotted on image
        :param score_threshold: threshold used to filter the inference output which are more confident than the
        threshold
        :param model_weights: the name of the model parameters used for the detection (they should be palced in
        the directory ~/weights)
        :param top_k: maximum value of returned object on a single image
        :param argv: additional parameters extracted from the parser
        '''
        DetectorInterface.__init__(self, display_img=display_img, score_threshold=score_threshold, classes_white_list=classes_white_list)
        
        self.args = parse_args(argv)
        self.top_k = top_k
        self.return_img = return_img
        model_path = SavePath.from_str(model_weights)
        self.args.config = model_path.model_name + '_config'
        set_cfg(self.args.config)

        with torch.no_grad():

            if self.args.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            print('Loading YOLACT model...', end='')
            self.net = Yolact()
            try:
                self.net.load_weights(model_weights)
            except FileNotFoundError:
                print('\n\033[91mThe weight in ' + model_weights + " does not exists. You must download them "
                                                                   "in that directory.\033[0m")
                exit(1)
            self.net.eval()
            print(' Done.')

            if self.args.cuda:
                self.net = self.net.cuda()

            self.net.detect.use_fast_nms = self.args.fast_nms
            self.net.detect.use_cross_class_nms = self.args.cross_class_nms
            cfg.mask_proto_debug = self.args.mask_proto_debug


    def output_formatting_and_display(self, dets_out, img, class_color=True, fps_str=''):
        '''
        This function displays the image with the inference results if self.display_img=True and
        format the network inference
        :param dets_out: the inference output of the network
        :param img: the original image
        :param class_color: boolean used to format the same class objects with the same color
        :param fps_str: mica lo so che vor dì
        :return:the reformatted inference outputs as a list [classes, scores, boxes, masks_out]
        '''

        mask_alpha = 0.5

        img_gpu = img / 255.0
        h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb=self.args.display_lincomb,
                            crop_masks=self.args.crop,
                            score_threshold=self.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):

            idx = t[1].argsort(0, descending=True)[:self.top_k]

            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
            masks = t[3][idx]
            
            ###REMOVE OBJECTS NOT IN CLASSES WHITE LIST
            idx_rm=[]
            if self.classes_white_list:
              
              for idx, cls in enumerate(classes):
                cls_name = cfg.dataset.class_names[cls]
                if cls_name not in self.classes_white_list:
                  idx_rm.append(idx)
                  
              classes=np.delete(classes, idx_rm)
              scores=np.delete(scores, idx_rm)
              boxes=np.delete(boxes, idx_rm,0)
              masks=masks.detach().clone().cpu().numpy()
              masks=torch.from_numpy(np.delete(masks, idx_rm,0)).cuda()
            ###
            
            masks_out = masks.detach().clone().cpu().numpy()
            
            # if no classes have been found return None Inference
            if classes.shape[0] == 0:
                inference = None
            else:
                inference = [classes, scores, boxes, masks_out]
            
        if self.display_img or self.return_img:

            num_dets_to_consider = min(self.top_k, classes.shape[0])
            for j in range(num_dets_to_consider):
                if scores[j] < self.score_threshold:
                    num_dets_to_consider = j
                    break

            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            def get_color(j, on_gpu=None):
                global color_cache
                color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

                if on_gpu is not None and color_idx in color_cache[on_gpu]:
                    return color_cache[on_gpu][color_idx]
                else:
                    color = COLORS[color_idx]

                    # The image might come in as RGB or BRG, depending
                    color = (color[2], color[1], color[0])
                    if on_gpu is not None:
                        color = torch.Tensor(color).to(on_gpu).float() / 255.
                        color_cache[on_gpu][color_idx] = color
                    return color

            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            if self.args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
                # After this, mask is of size [num_dets, h, w, 1]
                masks = masks[:num_dets_to_consider, :, :, None]

                # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                colors = torch.cat(
                    [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in
                     range(num_dets_to_consider)],
                    dim=0)
                masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

                # This is 1 everywhere except for 1-mask_alpha where the mask is
                inv_alph_masks = masks * (-mask_alpha) + 1

                # I did the math for this on pen and paper. This whole block should be equivalent to:
                #    for j in range(num_dets_to_consider):
                #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
                masks_color_summand = masks_color[0]
                if num_dets_to_consider > 1:
                    inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
                    masks_color_cumul = masks_color[1:] * inv_alph_cumul
                    masks_color_summand += masks_color_cumul.sum(dim=0)

                img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

            if self.args.display_fps:
                # Draw the box for the fps on the GPU
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

                img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

            # Then draw the stuff that needs to be done on the cpu
            # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
            img_numpy = (img_gpu * 255).byte().cpu().numpy()

            if self.args.display_fps:
                # Draw the text on the CPU
                text_pt = (4, text_h + 2)
                text_color = [255, 255, 255]

                cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

            if self.args.display_text or self.args.display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    color = get_color(j)
                    score = scores[j]

                    if self.args.display_bboxes:
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                    if self.args.display_text:
                        _class = cfg.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (_class, score) if self.args.display_scores else _class

                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.6
                        font_thickness = 1

                        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                        text_pt = (x1, y1 - 3)
                        text_color = [255, 255, 255]

                        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color,
                                    font_thickness,
                                    cv2.LINE_AA)

            if self.display_img:
              cv2.namedWindow("Yolact", cv2.WINDOW_NORMAL)
              cv2.imshow("Yolact", img_numpy)
              if cv2.waitKey(1) == ord('q'):
                  print("Closed Yolact Image Viewer.")
                  exit(0)

        if self.return_img:
            return inference, img_numpy
        return inference

    def img_inference(self, rgb):
        '''
        Used to make the inference on an image depending on the weight with which the object has been initialized
        :param rgb: the image on which make the inference
        :return: the image formatted with the results (if self.display is True) and a dictionary containing the
        inferences e.g., {'class_0': {'scores': [s1, s2], 'boxes': [b1, b2], 'masks': [m1, m2]},
                          'class_1': {'scores': [s1], 'boxes': [b1], 'masks': [m2]}, OTHER_CLASSES..}
        '''

        frame = torch.from_numpy(rgb).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        
        with torch.no_grad():
            if self.return_img:
                inference, yolact_img = self.output_formatting_and_display(preds, frame)
            else: 
                inference = self.output_formatting_and_display(preds, frame)

        inference_out = {}
        
        if inference is not None:
            for idx, cls in enumerate(inference[0]):
                cls = cfg.dataset.class_names[cls]
                if inference[1][idx] > self.score_threshold:
                    # expretion which evaluates if self.classes_white_list is empty OR the current class is in the white list
                    if not self.classes_white_list or cls in self.classes_white_list:
                        if cls not in inference_out.keys():
                            inference_out[cls] = {}
                            inference_out[cls]['scores'] = []
                            inference_out[cls]['boxes'] = []
                            inference_out[cls]['masks'] = []
                            inference_out[cls]['id'] = [] 
                        inference_out[cls]['scores'].append(inference[1][idx])
                        inference_out[cls]['boxes'].append(inference[2][idx])
                        inference_out[cls]['masks'].append(inference[3][idx])
                        inference_out[cls]['id'].append(-1) # the id is set to -1 here, because we do not perform tracking. We make the dictionary uniform between the detectors

        if self.return_img:
            return inference_out, yolact_img
        return inference_out
