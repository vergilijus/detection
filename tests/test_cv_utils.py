from detection.cv_utils import denorm_boxes, scale_boxes, boxes_to_yolo
import numpy as np


def test_denorm_boxes():
    norm_boxes = np.array([[0, 0, 1, 1], [0.25, 0.25, 0.5, 0.5], [0.25, 0.5, 0.75, 1]])
    required_boxes = np.array([[0, 0, 100, 80], [25, 20, 50, 40], [25, 40, 75, 80]])
    boxes = denorm_boxes(norm_boxes, (80, 100))
    assert np.all(boxes == required_boxes)


def test_scale_boxes():
    norm_boxes = np.array([[0, 0, 1, 1], [0.25, 0.25, 0.5, 0.5], [0.25, 0.5, 0.75, 1]])
    boxes = np.array([[0, 0, 100, 80], [25, 20, 50, 40], [25, 40, 75, 80]])
    assert np.all(norm_boxes == scale_boxes(norm_boxes, (1, 1), (1, 1)))
    assert np.all(norm_boxes == scale_boxes(boxes, (80, 100), (1, 1)))
    assert np.all(boxes == scale_boxes(norm_boxes, (1, 1), (80, 100)))


def test_box_to_yolo():
    box1 = [0, 0, 50, 100]
    box2 = [10, 20, 20, 40]
    boxes = np.array([box1, box2])
    yolo_box1 = [25, 50, 50, 100]
    yolo_box2 = [15, 30, 10, 20]
    yolo_boxes = np.array([yolo_box1, yolo_box2])
    assert np.all(yolo_boxes == boxes_to_yolo(boxes))
