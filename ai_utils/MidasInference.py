'''----------------------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2022
#
# author: Federico Rollo
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

import os
import torch
import cv2
import time
import numpy as np

from midas.model_loader import load_model


class MidasInference:

    def __init__(self, model_weights, display_img=False, grayscale=False, concatenate=True, display_fps=False):

        self.display_img = display_img
        self.display_fps = display_fps
        self.grayscale = grayscale
        self.concatenate = concatenate

        model_type = os.path.splitext(model_weights.split('/')[-1])[0]

        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: %s" % self.device)

        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, model_weights, model_type, False, None, False)
    

    def get_depth_image(self, frame): 

        with torch.no_grad():
            time_start = time.time()
            if frame is not None:
                original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
                image = self.transform({"image": original_image_rgb/255})["image"]

                sample = torch.from_numpy(image).to(self.device).unsqueeze(0)

                prediction = self.model.forward(sample)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size= original_image_rgb.shape[1::-1][::-1],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )

        if self.display_img:
            original_image_bgr = np.flip(original_image_rgb, 2)
            out_img = self.create_side_by_side(original_image_bgr, prediction)

            cv2.namedWindow('MiDaS Depth Estimation', cv2.WINDOW_NORMAL)
            cv2.imshow('MiDaS Depth Estimation', out_img/255)

            if cv2.waitKey(1) == ord('q'):
                print("\nClosed MiDaS Image Viewer.")
                exit(0)
        
        if self.display_fps:
            elapsed_time = time.time()-time_start
            if elapsed_time > 0:
                fps = 1 / elapsed_time 
            print(f"\rFPS: {round(fps,2)}", end="")

        return prediction


    def create_side_by_side(self, image, depth):
        """
        Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
        for better visibility.

        Args:
            image: the RGB image
            depth: the depth map
            grayscale: use a grayscale colormap?

        Returns:
            the image and depth map place side by side
        """
        depth_min = depth.min()
        depth_max = depth.max()
        normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
        normalized_depth *= 3

        right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
        if not self.grayscale:
            right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

        if image is None or not self.concatenate:
            return right_side
        else:
            return np.concatenate((image, right_side), axis=1)
