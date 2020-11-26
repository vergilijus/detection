import numpy as np


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
