
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