import numpy as np

from detection.detector.detector import Result


def test_threshold():
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]])
    classes = np.array([0, 1, 2, 3])
    scores = np.array([0.6, 0.2, 0.7, 0.1])
    result1 = Result(boxes, classes, scores)
    result2 = result1.threshold(0.5)

    assert np.all(result1.boxes == boxes)
    assert np.all(result1.classes == classes)
    assert np.all(result1.scores == scores)

    assert result2.boxes.tolist() == [[0, 0, 1, 1], [2, 2, 3, 3]]
    assert result2.classes.tolist() == [0, 2]
    assert result2.scores.tolist() == [0.6, 0.7]


def test_threshold_():
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]])
    classes = np.array([0, 1, 2, 3])
    scores = np.array([0.6, 0.2, 0.7, 0.1])
    result = Result(boxes, classes, scores)
    result.threshold_(0.5)

    assert result.boxes.tolist() == [[0, 0, 1, 1], [2, 2, 3, 3]]
    assert result.classes.tolist() == [0, 2]
    assert result.scores.tolist() == [0.6, 0.7]


def test_filter():
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]])
    classes = np.array([0, 1, 2, 3])
    scores = np.array([0.6, 0.2, 0.7, 0.1])
    result = Result(boxes, classes, scores)
    filtered1 = result.filter([1, 3])
    filtered2 = result.filter(np.array([1, 3]))

    assert filtered1.boxes.tolist() == [[1, 1, 2, 2], [3, 3, 4, 4]]
    assert filtered1.classes.tolist() == [1, 3]
    assert filtered1.scores.tolist() == [0.2, 0.1]
    assert filtered2.boxes.tolist() == [[1, 1, 2, 2], [3, 3, 4, 4]]
    assert filtered2.classes.tolist() == [1, 3]
    assert filtered2.scores.tolist() == [0.2, 0.1]
