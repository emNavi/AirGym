from setuptools import find_packages
from distutils.core import setup

setup(
    name='airgym',
    version='0.0.0',
    author='emNavi Tech',
    license="BSD 3-Clause",
    packages=find_packages(),
    author_email='',
    description='IsaacGym Drone RL Project',
    install_requires=['matplotlib',
                      'numpy',
                      'torch',
                      'pytorch3d',
                      'usd-core',
                      'rospkg',]
)