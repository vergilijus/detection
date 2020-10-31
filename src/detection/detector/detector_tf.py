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

    NAME = 'tf'
    MODELS = []

    def __init__(self, cache_dir, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = self.load_model(cache_dir, self.NAME, model)

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

    @staticmethod
    def load_model(cache_dir, cache_subdir, model_name):
        import ssl
        # noinspection PyUnresolvedReferences,PyProtectedMember
        ssl._create_default_https_context = ssl._create_unverified_context
        model_date = '20200711'
        base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
        model_dir = tf.keras.utils.get_file(fname=model_name,
                                            origin=f'{base_url}{model_date}/{model_name}.tar.gz',
                                            untar=True,
                                            cache_subdir=cache_subdir,
                                            cache_dir=cache_dir)
        saved_model_dir = os.path.join(model_dir, 'saved_model')
        return tf.saved_model.load(saved_model_dir)

