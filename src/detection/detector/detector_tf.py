import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from detection.detector.detector import Detector, Result
from detection.cv_utils import denorm_boxes

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class TFDetector(Detector):
    """
    Detector compatible with TensorFlow 2 Detection Model Zoo
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    """

    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = tf.saved_model.load(model)

    def detect(self, img: np.ndarray, threshold) -> Result:
        img = img[:, :, ::-1]  # BGR -> RGB
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.model(input_tensor)

        # Convert to numpy arrays, take index [0] to remove the batch dimension
        # Leave the first num_detections
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']

        # Convert boxes to format [xmin, ymin, xmax, ymax]
        boxes = boxes[:, [1, 0, 3, 2]]
        boxes = denorm_boxes(boxes, img.shape)

        return Result(boxes, classes, scores).threshold_(threshold)
