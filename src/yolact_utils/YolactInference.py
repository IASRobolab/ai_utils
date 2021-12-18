#!/usr/bin/env python

import sys
from pathlib import Path
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

home_path = str(Path.home())

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
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

    global args
    args = parser.parse_args(argv)


color_cache = defaultdict(lambda: {})


class YolactInference:

    def __init__(self, display=False, score_threshold=0.5,
                 trained_model='yolact_plus_resnet50_54_800000.pth', argv=None):
        parse_args(argv)
        self.display = display  # boolean to chose if display image results or not
        self.score_threshold = score_threshold  # threshold used to filter the detectio results
        trained_model = home_path + '/weights/' + trained_model
        model_path = SavePath.from_str(trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

        with torch.no_grad():

            if args.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')

            print('Loading YOLACT model...', end='')
            self.net = Yolact()
            self.net.load_weights(trained_model)
            self.net.eval()
            print(' Done.')

            if args.cuda:
                self.net = self.net.cuda()

            self.net.detect.use_fast_nms = args.fast_nms
            self.net.detect.use_cross_class_nms = args.cross_class_nms
            cfg.mask_proto_debug = args.mask_proto_debug

    def prep_display(self, dets_out, img, class_color=True, fps_str=''):
        img_orig = img.byte().cpu().numpy()
        img_numpy = None

        mask_alpha = 0.5

        img_gpu = img / 255.0
        h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                            crop_masks=args.crop,
                            score_threshold=self.score_threshold)
            cfg.rescore_bbox = save

        with timer.env('Copy'):

            idx = t[1].argsort(0, descending=True)[:args.top_k]

            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
            masks = t[3][idx]
            masks_out = masks.detach().clone().cpu().numpy()

        if self.display:

            num_dets_to_consider = min(args.top_k, classes.shape[0])
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
            if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
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

            if args.display_fps:
                # Draw the box for the fps on the GPU
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

                img_gpu[0:text_h + 8, 0:text_w + 8] *= 0.6  # 1 - Box alpha

            # Then draw the stuff that needs to be done on the cpu
            # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
            img_numpy = (img_gpu * 255).byte().cpu().numpy()

            if args.display_fps:
                # Draw the text on the CPU
                text_pt = (4, text_h + 2)
                text_color = [255, 255, 255]

                cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

            if num_dets_to_consider == 0:
                return img_numpy, None

            if args.display_text or args.display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    color = get_color(j)
                    score = scores[j]

                    if args.display_bboxes:
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                    if args.display_text:
                        _class = cfg.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

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

        if img_numpy is None:
            return img_orig, [classes, scores, boxes, masks_out]
        else:
            return img_numpy, [classes, scores, boxes, masks_out]

    def img_inference(self, rgb, classes=[]):
        frame = torch.from_numpy(rgb).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        
        img_numpy, inference = self.prep_display(preds, frame)

        inference_out = {}
        if inference is not None:

            for idx, cls in enumerate(inference[0]):
                # print(idx, cls)
                if cls not in inference_out.keys() and (cls in classes or not classes):
                    inference_out[cls] = {}
                    inference_out[cls]['scores'] = []
                    inference_out[cls]['boxes'] = []
                    inference_out[cls]['masks'] = []
                inference_out[cls]['scores'].append(inference[1][idx])
                inference_out[cls]['boxes'].append(inference[2][idx])
                inference_out[cls]['masks'].append(inference[3][idx])
        
        return img_numpy, inference_out
