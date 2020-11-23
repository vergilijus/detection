import json

import numpy as np

from detection.detector import Detection


def test_result_dict():
    boxes = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64)
    classes = np.array([0, 1], dtype=np.float)
    scores = np.array([0.5, 0.7])
    res = Detection(boxes, classes, scores)
    assert res.dict() == {
        'boxes': [[0, 0, 0, 0], [0, 0, 0, 0]],
        'classes': [0, 1],
        'scores': [0.5, 0.7]
    }
    json.dumps(res.dict())  # check json serializable
