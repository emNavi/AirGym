from setuptools import find_packages
from distutils.core import setup

setup(
    name='airgym',
    version='0.0.2',
    author='emNavi Tech',
    license="BSD 3-Clause",
    packages=find_packages(),
    author_email='',
    description='IsaacGym Drone RL Project',
    install_requires=["numpy",
                    "scipy",
                    "pyyaml",
                    "pillow",
                    "imageio",
                    "ninja",
                    'matplotlib',
                    'torch==2.0.0',
                    'rospkg',
                    'gym==0.23.1',
                    'rlpx4controller',
                    'usd-core',
                    'pytorch3d',
                    'tensorboardX'
                    ]
)