from typing import List

import numpy as np


class Box:

    def __init__(self, xmin, ymin, xmax, ymax) -> None:
        self.box = np.array([xmin, ymin, xmax, ymax])

    def points(self):
        x1, y1, x2, y2 = self.box
        return (x1, y1), (x2, y2)

    def p1(self):
        return tuple(self.box[:2])

    def p2(self):
        return tuple(self.box[2:])

    def norm(self, h, w):
        self.box = norm_boxes(self.box, (h, w))
        return self

    def denorm(self, h, w):
        self.box = denorm_boxes(self.box, (h, w))
        return self

    def int(self):
        self.box = self.box.astype(int)
        return self


Boxes = List[Box]


def norm_boxes(boxes: np.ndarray, shape) -> np.ndarray:
    """
    Normalize boxes coordinates from given shape to [0, 1]
    :param boxes: numpy array of shape (n_boxes, 4) box format [xmin, ymin, xmax, ymax]
    :param shape: original image shape in format (height, width) or (height, width, channel)
    :return: normalized boxes
    """
    h, w = shape[:2]
    return boxes / np.array([w, h, w, h])


def denorm_boxes(boxes: np.ndarray, shape) -> np.ndarray:
    """
    Denormalize boxes to required shape
    :param boxes: numpy array of shape (n_boxes, 4) box format [xmin, ymin, xmax, ymax]
    :param shape: image shape in format (height, width) or (height, width, channel)
    :return: denormalized boxes
    """
    h, w = shape[:2]
    return boxes * np.array([w, h, w, h])


def scale_boxes(boxes: np.ndarray, src_shape, dst_shape) -> np.ndarray:
    """
    Scale boxes to required shape
    :param boxes: numpy array of shape (n_boxes, 4) box format [xmin, ymin, xmax, ymax]
    :param src_shape: source image shape in format (height, width) or (height, width, channel)
    :param dst_shape: destination image shape in format (height, width) or (height, width, channel)
    :return: scaled boxes
    """
    h1, w1 = src_shape[:2]
    h2, w2 = dst_shape[:2]
    wk = w2 / w1
    hk = h2 / h1
    return boxes * np.array([wk, hk, wk, hk])


def boxes_to_yolo(boxes: np.ndarray):
    """
    Convert boxes from format [xmin, ymin, xmax, ymax] to [cx, xy, width, height]
    :param boxes: boxes in format [xmin, ymin, xmax, ymax]
    :return: boxes in format [cx, cy, width, height]
    """
    yolo_boxes = np.zeros_like(boxes)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    yolo_boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    yolo_boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    yolo_boxes[:, 0] = boxes[:, 0] + w / 2
    yolo_boxes[:, 1] = boxes[:, 1] + h / 2
    return yolo_boxes


def bbox(boxes: np.ndarray) -> np.ndarray:
    """
    Return bounding box of bounding boxes
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    return np.array([np.min(x1), np.min(y1), np.max(x2), np.max(y2)])


def bbox_of_boxes(boxes: List[Box]) -> Box:
    return Box(*bbox(np.array([b.box for b in boxes])))
