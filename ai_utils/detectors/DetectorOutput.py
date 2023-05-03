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
class DetectedObject:

    def __init__(self, idx, cls, bbox, mask, score) -> None:
        # TODO add bbox type?
        self.idx = idx
        self.cls = cls
        self.bbox = bbox
        self.mask = mask
        self.score = score


class DetectorOutput:

    def __init__(self, classes, bboxes, scores, masks=None , ids=None) -> None:
        if not ids:
            ids = [-1]*len(classes)

        if not masks:
            masks = [None]*len(classes)

        self.detected_objects = {}

        for idx in range(len(ids)):
            detected_obj = DetectedObject(ids[idx], classes[idx], bboxes[idx], masks[idx], scores[idx])

            if not detected_obj.cls in  self.detected_objects.keys():
                self.detected_objects[detected_obj.cls] = []
            
            self.detected_objects[detected_obj.cls].append(detected_obj)


    def get_detected_objects(self):
        return self.detected_objects


    def get_detected_objects_by_class(self, cls):
        return self.detected_objects[cls]

    
    def get_detected_object_by_id(self, id, cls=None):

        if cls:
            classes = [cls]
        else:
            classes =  self.detected_objects.keys()

        object : DetectedObject
        for cls in classes:
            for object in self.detected_objects[cls]:
                if object.idx == id:
                    return object

        # if no object have been found
        return None


    