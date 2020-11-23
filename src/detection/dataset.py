from pathlib import Path
from typing import Iterable, Iterator

from detection.detector import Detection
from detection.detector.detector import Detections
from detection.label_map import LabelMap

FORMATS = ['.png', '.jpg', '.jpeg']


class Dataset(Iterable):
    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        raise NotImplementedError


class YOLO(Dataset):

    def __init__(self, gt_detections: Detections, label_map: LabelMap) -> None:
        super().__init__()
        self.items = gt_detections
        self.label_map = label_map

    def load(self, path):
        pass

    def save(self, path):
        pass

    def __next__(self):
        pass
