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
from ai_utils.feature_extractors.FeatureExtractorInterface import FeatureExtractorInterface

from mmt import models
from mmt.utils.serialization import load_checkpoint, copy_state_dict
import torch
from torch.nn import functional as F
import numpy as np


class FastBaseTransform(torch.nn.Module):
    """
    Transform that does all operations on the GPU for super speed.
    This doesn't support a lot of config settings and should only be used for production.
    Maintain this as necessary.
    """
    MEANS = (103.94, 116.78, 123.68)
    STD = (57.38, 57.12, 58.40)

    def __init__(self):
        super().__init__()
        self.mean = torch.Tensor(self.MEANS).float().cuda()[None, :, None, None]
        self.std = torch.Tensor(self.STD).float().cuda()[None, :, None, None]

    def forward(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
        # img assumed to be a pytorch BGR image with channel order [n, h, w, c]
        img_size = (256, 128)
        img = img.permute(0, 3, 1, 2).contiguous()
        img = F.interpolate(img, img_size, mode='bilinear', align_corners=False)
        img = (img - self.mean) / self.std
        img = img[:, (2, 1, 0), :, :].contiguous()
        # Return value is in channel order [n, c, h, w] and RGB
        return img


class MMTExtractor(FeatureExtractorInterface):

    def __init__(self, model_weights, target_class) -> None:

        super().__init__(target_class)

        # TODO add num_feature, add num_classes

        self.transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
        print('Loading REID model...', end='')
        self.model_REID = models.create(
            'resnet_ibn50a', pretrained=False,
            num_features=256, dropout=0,
            num_classes=0, norm=True
        )
        self.model_REID.cuda()
        self.model_REID = torch.nn.DataParallel(self.model_REID)
        try:
            checkpoint = load_checkpoint(model_weights)
        except ValueError:
            print(
                '\n\033[91mWeights not found in ' + model_weights + ". You must download them "
                "in that directory.\033[0m"
            )
            exit(1)
        copy_state_dict(checkpoint['state_dict'], self.model_REID)
        self.model_REID.eval()
        print('Done.')


    def get_features(self, image: np.ndarray, detector_inference: dict):

        if not self.target_class in detector_inference.keys():
            return None

        boxes = detector_inference[self.target_class]['boxes']
        masks = detector_inference[self.target_class]['masks']

        images = []

        for id in range(len(boxes)):
            rgb_new = image.copy()
            for i in range(3):
                rgb_new[:, :, i] = rgb_new[:, :, i] * masks[id]
            image_transformed = self.transform(
                torch.from_numpy(rgb_new[boxes[id][1]:boxes[id][3], boxes[id][0]:boxes[id][2], :]).unsqueeze(
                    0).cuda().float())
            images.append(image_transformed[0])

        images = [images.cuda().float() for images in images]
        images = torch.stack(images, 0)
        # PASS THE IMAGES INSIDE THE EXTERNAL NETWORK
        features = self.model_REID(images).data.cpu()
        #self.model_REID(img_transformed.cuda()).data.cpu()[0].numpy()
    
        return features
