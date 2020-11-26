import os
from typing import Iterable, Iterator, List

from detection.detector.detector import Detections
from detection.label_map import LabelMap
from detection.utils import get_base_name

FORMATS = ['.png', '.jpg', '.jpeg']


class Dataset(Iterable):
    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        raise NotImplementedError


class YOLO(Dataset):

    def __init__(self, files: List[str], gt_detections: Detections, label_map: LabelMap) -> None:
        super().__init__()
        self.files = files
        self.items: Detections = gt_detections
        self.label_map = label_map

    def load(self, path):
        pass

    @staticmethod
    def _save_obj_names(path, label_map: LabelMap):
        full_path = os.path.join(path, 'obj.names')
        with open(full_path, 'w') as f:
            f.writelines([f'{label}\n' for label in label_map.labels()])

    def _save_obj_data(self, path):
        full_path = os.path.join(path, 'obj.data')
        with open(full_path, 'w') as f:
            content = \
                f"""classes = {len(self.label_map.labels())}
train = data/train.txt
names = data/obj.names
backup = backup/
            """
            f.write(content)

    def _save_train_txt(self, path):
        full_path = os.path.join(path, 'train.txt')
        lines = [f'data/obj_train_data/{f}\n' for f in self.files]
        with open(full_path, 'w') as f:
            f.writelines(lines)

    def _save_labels(self, path):
        for img_file, item in zip(self.files, self.items):
            base_name = get_base_name(img_file)
            label_file = f'{base_name}.txt'
            data_dir = os.path.join(path, 'obj_train_data')
            full_path = os.path.join(data_dir, label_file)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            with open(full_path, 'w') as f:
                lines = [f'{c} {box[0]} {box[1]} {box[2]} {box[3]}\n' for c, box in zip(item.classes, item.boxes)]
                f.writelines(lines)

    def save(self, path, zip=False):
        self._save_labels(path)
        self._save_train_txt(path)
        self._save_obj_data(path)
        self._save_obj_names(path, self.label_map)

    def __next__(self):
        pass
