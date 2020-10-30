from detection.capture import get_capture
from detection.detector.detector_tf import TFDetector
import cv2 as cv

from detection.drawing import plot_boxes


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--capture', type=str, required=True)
    args = parser.parse_args()

    detector = TFDetector(args.model)
    capture = get_capture(args.capture)

    def detect_and_show(frame):
        result = detector.detect(frame)
        plot_boxes(frame, result.boxes, result.classes, result.scores)
        cv.imshow('frame', frame)
        cv.waitKey(1)

    capture.on_frame(detect_and_show)
    capture.start()


if __name__ == '__main__':
    main()
