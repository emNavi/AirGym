from setuptools import find_packages
from distutils.core import setup

# setup(
#     name='airgym',
#     version='0.0.0',
#     author='',
#     license="",
#     packages=find_packages(),
#     author_email='',
#     description='IsaacGym Drone RL Project',
#     install_requires=['isaacgym',
#                       'matplotlib',
#                       'numpy',
#                       'torch',
#                       'pytorch3d',
#                       'usd-core',]
# )

setup(
    name='airgym',
    version='0.0.0',
    author='',
    license="",
    packages=find_packages(),
    author_email='',
    description='IsaacGym Drone RL Project',
    install_requires=['matplotlib',
                      'numpy',
                      'torch',]
)