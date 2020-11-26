from typing import Dict, Iterable, List

Labels = List[str]


class LabelMap:
    def __init__(self, label_map: Dict[int, str]) -> None:
        self.label_map: Dict[int, str] = label_map

    @staticmethod
    def from_list(labels: List[str]):
        return LabelMap({k: v for k, v in enumerate(labels)})

    def labels(self) -> Labels:
        return [v for k, v in sorted(self.label_map.items())]

    def to_labels(self, classes: Iterable[int]) -> List[str]:
        return [self.label_map[c] for c in classes]


class COCO(LabelMap):

    def __init__(self) -> None:
        super().__init__({i + 1: value for i, value in enumerate(
            ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
             'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
             'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
             'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
             'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
             'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
             'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
             'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
             'toothbrush'])})
