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

    cv.namedWindow('detection', cv.WINDOW_GUI_NORMAL)

    def detect_and_show(frame):
        result = detector.detect(frame, threshold=0.5)
        labels = COCO().labels(result.classes)
        draw_results(frame, result.boxes,
                     classes=result.classes,
                     scores=result.scores,
                     labels=labels)
        cv.imshow('detection', frame)
        cv.waitKey(1)

    capture.on_frame(detect_and_show)
    capture.start()


if __name__ == '__main__':
    main()
