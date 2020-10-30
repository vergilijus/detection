from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='detection',
    version='0.0.1',
    description='Utils for object detection testing, evaluation and benchmarking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pypa/sampleproject',
    author='Ruslan Dulimov',
    keywords='object detection',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7, <4',
    install_requires=[
        'opencv-python',
        'numpy',
        'tqdm',
        'webcolors'
    ],
    extras_require={
        'torch': ['torch', 'torchvision'],
        'tf': ['tensorflow']
    },

    entry_points={
        'console_scripts': [
            'detection=detection.main:main',
        ],
    },
)
