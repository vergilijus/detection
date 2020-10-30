import numpy as np


class Result:
    """
    Represent detection result of a single image
    """

    def __init__(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray):
        self.boxes = boxes.astype(float)
        self.classes = classes.astype(int)
        self.scores = scores.astype(float)

    def dict(self) -> dict:
        return {
            'boxes': self.boxes.tolist(),
            'classes': self.classes.tolist(),
            'scores': self.scores.tolist()
        }

    def threshold(self, min_score):
        index = self.scores[self.scores >= min_score]
        return Result(self.boxes[index],
                      self.classes[index],
                      self.scores[index])

    def threshold_(self, min_score):
        index = self.scores[self.scores >= min_score]
        self.boxes = self.boxes[index]
        self.classes = self.classes[index]
        self.scores = self.scores[index]
        return self

    def filter(self, score, classes):
        raise NotImplementedError

    def nms(self, iou: float):
        raise NotImplementedError


class Detector:

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def detect(self, img: np.array, threshold: float) -> Result:
        raise NotImplementedError
