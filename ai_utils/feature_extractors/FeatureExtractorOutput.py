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
from ai_utils.detectors.DetectorOutput import DetectedObject
import pdb

class FeatureObject:

    def __init__(self, cls, idx, feature, bbox, score, mask) -> None:
        self.cls = cls
        self.idx = idx
        self.feature = feature
        self.bbox = bbox
        self.score = score
        self.mask = mask
    
    @classmethod
    def init_by_detector(self, feature, detected_object: DetectedObject) -> None:
        return self(detected_object.cls, detected_object.idx, feature, detected_object.bbox, detected_object.score, detected_object.mask)
    

class FeatureExtractorOutput:

    def __init__(self, features: list, detected_objects: list) -> None:

        self.features_objects = {}

        for idx in range(len(features)):
            feat_obj = FeatureObject.init_by_detector(features[idx], detected_objects[idx])
            if not detected_objects[idx].cls in self.features_objects.keys():
                self.features_objects[detected_objects[idx].cls] = []

            self.features_objects[detected_objects[idx].cls].append(feat_obj)


    def get_feature_objects(self) -> dict:
        return self.features_objects


    def get_object_by_id(self, id: int) -> FeatureObject:

        object: FeatureObject
        # pdb.set_trace()
        for object in list(self.features_objects.values())[0]:
            if id == object.idx:
                return object

        return None