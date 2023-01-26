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
class DetectorInterface:

    classes_white_list: set
    display_img: bool
    score_threshold: float

    def __init__(self, display_img=False, score_threshold=0.5, classes_white_list=set()) -> None:
        self.display_img = display_img
        self.score_threshold = score_threshold

        if classes_white_list is None:
            classes_white_list = set()
        elif not isinstance(classes_white_list, set):
            classes_white_list = set(classes_white_list)

        self.classes_white_list = classes_white_list


    def img_inference(self, image):
        raise NotImplementedError


    def add_class(self, cls) -> bool:
        if isinstance(cls, str):
            self.classes_white_list.add(cls)
        elif isinstance(cls, list) or isinstance(cls, set):
            self.classes_white_list.update(cls)
        else:
            print("\033[91mERROR: add_class function accept only string, lists or sets.\033[0m")
            return False
        return True


    def remove_class(self, cls) -> bool:
        if isinstance(cls, str):
            self.classes_white_list.discard(cls)
        elif isinstance(cls, list) or isinstance(cls, set):
            for value in cls:
                self.classes_white_list.discard(value)
        else:
            print("\033[91mERROR: remove_class function accept only string, lists or sets.\033[0m")
            return False
        return True


    def empty_white_list(self) -> bool:
        self.classes_white_list.clear()
        return True