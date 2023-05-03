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
class FeatureObject:

    def __init__(self, id, feature, bbox, mask) -> None:
        # TODO put detectorOutputInterface? or better use another initializer to initialize with another init
        self.id = id
        self.feature = feature
        self.bbox = bbox
        self.mask = mask
    

class FeatureExtractorOutput:

    def __init__(self, features: list, bboxes: list, masks: list, ids = None) -> None:
        if not ids:
            ids = [-1]*len(features)

        self.features_objects = []
        for idx in range(len(features)):
            feat_obj = FeatureObject(ids[idx], features[idx], bboxes[idx], masks[idx])

        self.features_objects.append(feat_obj)


    def get_feature_objects(self):
        return self.features_objects


    def get_object_by_id(self, id):

        object: FeatureObject
        for object in self.features_objects:
            if id == object.id:
                return object

        return None # TODO should we raise an exception instead?