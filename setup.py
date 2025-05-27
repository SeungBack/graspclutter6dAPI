from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os


setup(
    name='graspclutter6dAPI',
    version='0.0.1',
    description='GraspClutter6D API',
    author='Seunghyeok Back',
    author_email='shback@kimm.re.kr',
    url='https://sites.google.com/view/graspclutter6d',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'transforms3d==0.3.1',
        'open3d>=0.8.0.0',
        'trimesh',
        'tqdm',
        'Pillow',
        'opencv-python',
        'pillow',
        'matplotlib',
        'pywavefront',
        'trimesh',
        'scikit-image',
        'autolab_core',
        'autolab-perception',
        'cvxopt',
        'dill',
        'h5py',
        'scikit-learn',
        'grasp_nms',
        'numpy==1.23.4',
    ]
)