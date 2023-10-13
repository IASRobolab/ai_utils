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
from ai_utils.feature_extractors.FeatureExtractorOutput import FeatureExtractorOutput
from ai_utils.detectors.DetectorOutput import DetectedObject, DetectorOutput

from mmt import models
from mmt.utils.serialization import load_checkpoint, copy_state_dict
from mmt.loss.triplet import SoftTripletLoss
from mmt.utils.serialization import save_checkpoint
# from mmt.loss import CrossEntropyLabelSmooth
# from mmt.utils.lr_scheduler import WarmupMultiStepLR
import torch
from torch.nn import functional as F
import numpy as np

import os

class ContinualTrain(object):

    def __init__(self, model, margin=0.0):
        super(ContinualTrain, self).__init__()
        self.model = model
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, batch_imgs, targets, optimizer):
        s_features, _ = self.model(batch_imgs)
        # backward main #
        loss_tr = self._forward(s_features, targets)
        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()
        # print('Epoch:' , epoch)
        print('Loss_tr:' , loss_tr)
        

    def _forward(self, s_features, targets):
        loss_tr = self.criterion_triple(s_features, s_features, targets)
        return loss_tr


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

    def __init__(self, model_weights, target_classes, train = False, batch_size = 8, save_imgs = False) -> None:

        super().__init__(target_classes, save_imgs)

        self.continual_learning = train

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
        if train:
            self.batch_size = batch_size
            self.batch_multiplier = 0
            self.model_REID.train()
            self.trainer = ContinualTrain(self.model_REID)
            params = []
            for key, value in self.model_REID.named_parameters():
                if not value.requires_grad:
                    continue
                params += [{"params": [value], "lr": 0.00035, "weight_decay": 5e-4}] # 0.00035
            self.optimizer = torch.optim.Adam(params)
            # self.lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        else:
            self.model_REID.eval()
        print('MMT Feature Extractor Loaded.')


    def get_features(self, image: np.ndarray, detector_inference: DetectorOutput):

        target_class_inference = detector_inference.get_detected_objects_by_class(self.target_classes)
        
        if not target_class_inference:
            return None

        images = []
        obj_list = []
        object : DetectedObject


        if self.save_imgs:
            self.original_images.clear()
        self.cropped_images.clear()


        for cls in target_class_inference.keys():
            for object in target_class_inference[cls]:

                obj_list.append(object)
                rgb_new = image.copy()
                for i in range(3):
                    rgb_new[:, :, i] = rgb_new[:, :, i] * object.mask

                image_transformed = self.transform(
                    torch.from_numpy(rgb_new[object.bbox[1]:object.bbox[3], object.bbox[0]:object.bbox[2], :]).unsqueeze(0).cuda().float()
                )
                trasf_img = image_transformed[0].cuda().float()
                images.append(trasf_img)

                if self.save_imgs:
                    self.original_images[object.idx] = rgb_new[object.bbox[1]:object.bbox[3], object.bbox[0]:object.bbox[2], :]
                self.cropped_images[object.idx] = trasf_img

        images = torch.stack(images, 0)

        # PASS THE IMAGES INSIDE THE EXTERNAL NETWORK

        features = self.model_REID(images).data.cpu().numpy()

        feature_out = FeatureExtractorOutput(features, obj_list)
        
        return feature_out
    

    def get_pure_feature(self, images, batch = 8):
        idx = 0
        features = []
        length = len(images)

        while (idx*batch < length):
            batch_idx = idx*batch
            imgs = torch.stack(images[batch_idx:batch_idx+batch], 0)
            features += self.model_REID(imgs).data.cpu().numpy().tolist()
            idx += 1

        return features

    # substitute network weights
    def set_network_weights(self, state_dict):
        copy_state_dict(state_dict, self.model_REID)

    # get current network weights
    def get_network_weights(self):
        return self.model_REID.state_dict()
    

    # Train the network in continual learning manner
    def cl_train(self, imgs: list):
        # lr_scheduler.step()

        targets = torch.zeros(self.batch_size).type(dtype=torch.int64)

        target_len = int(self.batch_size/2)
        targets[0:target_len] = 1
        targets = targets.cuda()

        images_len = len(imgs)

        # format images compliant to batch (discard the one in eccess)
        check_batch_multiple = (images_len/2)%self.batch_size
        if check_batch_multiple != 0:
            batch_compliant_length = images_len/2 - check_batch_multiple
            imgs = imgs[:int(batch_compliant_length)] + imgs[int(images_len/2):int(images_len/2+batch_compliant_length)]

        images_len = len(imgs)

        self.batch_multiplier = int(images_len/self.batch_size)

        for iter in range(self.batch_multiplier):
            target_start = iter * int(self.batch_size/2)
            target_end = target_start + int(self.batch_size/2)

            distractor_start = target_start + int(images_len/2)
            distractor_end = distractor_start + int(self.batch_size/2)

            target_imgs = imgs[target_start:target_end]
            distractor_imgs = imgs[distractor_start:distractor_end]

            batch_imgs = torch.stack(target_imgs + distractor_imgs, 0)
            self.trainer.train(batch_imgs, targets, self.optimizer)


    def save_weights(self, path):

        weights_name = 'checkpoint_' + str(self.batch_size) + '_' + str(self.batch_multiplier) + '.pth.tar'

        weights = self.model_REID.state_dict()
        save_checkpoint(
            {
            'state_dict': weights,
            }, 
            False, 
            fpath=os.path.join(path, weights_name)
        )
