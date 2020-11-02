from detection.capture import get_capture
from detection.detector import TFDetector
import cv2 as cv

from detection.drawing import draw_results
from detection.label_map import COCO

CACHE_DIR = 'models'


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--capture', type=str, required=True)
    args = parser.parse_args()

    detector = TFDetector(CACHE_DIR, args.model)
    capture = get_capture(args.capture)

    def detect_and_show(frame):
        result = detector.detect(frame, threshold=0.3).filter([1])
        labels = COCO().labels(result.classes)
        draw_results(frame, result.boxes, result.classes, result.scores, labels)
        cv.imshow('frame', frame)
        cv.waitKey(1)

    capture.on_frame(detect_and_show)
    capture.start()


if __name__ == '__main__':
    main()
