import logging
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Callable, Iterator, Tuple, Union

import cv2 as cv
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def is_image(source) -> bool:
    name, ext = os.path.splitext(source)
    return type(source) is str \
           and os.path.exists(source) \
           and os.path.isfile(source) \
           and ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']


def is_dir(source) -> bool:
    return type(source) is str \
           and os.path.exists(source) \
           and os.path.isdir(source)


def is_rtsp(source) -> bool:
    return type(source) is str \
           and len(source) > 6 \
           and source[:7] == 'rtsp://'


def get_capture(source: Union[str, int]):
    if is_image(source):
        return ImageCapture(source)
    elif is_rtsp(source):
        return RTSPCapture(source)
    else:
        return VideoCapture(source)


class Capture:

    def __init__(self, source) -> None:
        logger.info(f'Starting capture: {source}')
        self.source = source
        self._on_frame = lambda _: None
        self._on_no_frame = lambda: None

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        return self.read()

    def start(self):
        for _, frame in tqdm(self):
            status, frame = self.read()
            if status:
                self._on_frame(frame)
            else:
                self._on_no_frame()

    def read(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError

    def is_opened(self) -> bool:
        raise NotImplementedError

    def isOpened(self) -> bool:
        """ Alias for opencv VideoCapture interoperability"""
        return self.is_opened()

    def release(self):
        raise NotImplementedError

    def on_frame(self, f: Callable[[np.ndarray], None]):
        self._on_frame = f

    def on_no_frame(self, f: Callable[[np.ndarray], None]):
        self._on_no_frame = f


class VideoCapture(Capture):
    def __init__(self, source) -> None:
        super().__init__(source)
        if source in [str(i) for i in range(9)]:
            source = int(source)
        self.cap = cv.VideoCapture(source)

    def read(self) -> (bool, np.ndarray):
        return self.cap.read()

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def set_camera_res(self, w, h):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, int(w))
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, int(h))
        return self.get_frame_size()

    def release(self):
        self.cap.release()

    def get_frame_size(self):
        return int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)), \
               int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))

    def get(self, value):
        return self.cap.get(value)


class RTSPCapture(VideoCapture):

    def __init__(self, source, restart_after=3) -> None:
        super().__init__(source)
        self.source = source
        self.restart_after = restart_after
        self.frames_queue = Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.is_opened = True
        self.no_frame_count = 0
        self.executor.submit(self._worker)

    def _worker(self):
        while self.is_opened and self.cap.isOpened():
            status, frame = self.cap.read()
            self.frames_queue.put((status, frame))
            if self.frames_queue.qsize() == 2:
                self.frames_queue.get()

    def _read(self) -> (bool, np.ndarray):
        try:
            return self.frames_queue.get(block=True, timeout=5)
        except Empty:
            return False, None

    def read(self) -> (bool, np.ndarray):
        if self.no_frame_count > self.restart_after:
            self.restart()

        status, frame = self._read()
        if not status:
            self.no_frame_count += 1
        else:
            self.no_frame_count = 0

        return status, frame

    def restart(self):
        self.release()
        self.__init__(self.source)

    def isOpened(self) -> bool:
        return self.is_opened

    def set_camera_res(self, w, h):
        raise NotImplementedError

    def release(self):
        self.is_opened = False
        self.executor.shutdown()
        super().release()


class ImageCapture(Capture):

    def __init__(self, source) -> None:
        super().__init__(source)
        self.img = cv.imread(source)
        self.is_opened = True

    def read(self):
        return True, self.img.copy()

    def is_opened(self) -> bool:
        return self.is_opened

    def release(self):
        self.is_opened = False


class DirectoryCapture(Capture):

    def __init__(self, source, extensions=('.png', '.jpeg', '.jpg')) -> None:
        super().__init__(source)
        if not os.path.isdir(source):
            print('Path should be a directory')
            return
        self.dir = source
        self.names = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]
        self.path_to_imgs = [os.path.join(source, f) for f in self.names]
        self.id = 0

    def read(self):
        img = cv.imread(self.path_to_imgs[self.id])
        self.id += 1
        return True, img

    def is_opened(self) -> bool:
        return self.id < len(self.path_to_imgs)

    def release(self):
        pass

    def get_size(self):
        pass

    def get_file_name(self):
        return self.names[self.id]
