from time import time


def get_base_name(path):
    return path.rsplit('.', 1)[0]


def download(url, file):
    from urllib.request import urlretrieve
    from tqdm import tqdm

    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=file, reporthook=t.update_to, data=None)
        t.total = t.n


class FPSCounter:
    def __init__(self) -> None:
        self.start_time = None
        self.total_time = 0
        self.total_frames = 0

    def update(self):
        if not self.start_time:
            self.start_time = time()
            return
        self.total_time = time() - self.start_time
        self.total_frames += 1

    def get_fps(self) -> float:
        if self.total_time == 0:
            return 0
        else:
            return round(self.total_frames / self.total_time, 2)
