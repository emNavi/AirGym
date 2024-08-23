# AirGym
> This project is inspired by Aerial Gym Simulator but greatly improves it. 

This project provides a realistic dynamics and RL framework for Sim2Real tasks of quadcopter. Drones can be trained in AirGym and then transferred to reality.

## Requirements
1. Ubuntu 20.04 or 22.04
1. [Conda](https://www.anaconda.com/download) or [Miniconda ](https://docs.conda.io/en/latest/miniconda.html)
1. [NVIDIA Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) ([Pytorch]((https://pytorch.org/)) needs to upgrade for 40 series of GPUs. Please follow the installation guidance.)

> Note this repository has been tested on Ubuntu 20.04/22.04 with PyTorch 2.0.0 + CUDA11.8.

## Installation
### 1. Install IsaacGym Preview 4 with torch2.0.0+cuda11.8
1. Download package from the [official page](https://developer.nvidia.com/isaac-gym) and unzip.
1. Edit `install_requires` in `python/setup.py`:
    ```
    install_requires=[
                "numpy",
                "scipy",
                "pyyaml",
                "pillow",
                "imageio",
                "ninja",
            ],
    ```
1. Edit `dependencies` in `python/rlgpu_conda_env.yml`:
    ```python
    dependencies:
    - python=3.8
    - numpy=1.20
    - pyyaml
    - scipy
    - tensorboard
    ```
1. Create a new conda environment named `rlgpu` and install `isaacgym`:
    ```bash
    cd isaacgym
    ./create_conda_env_rlgpu.sh
    ```
1. Install PyTorch2.0.0 and CUDA11.8:
    ```bash
    conda activate rlgpu
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

### 2. Install [rlPx4Controller](https://github.com/FP-Flight/rlPx4Controller)
1. Install Eigen (The recommend version is 3.3.7):
    ```bash
    sudo apt install libeigen3-dev
    ```
1. Install pybind11:
    ```bash
    pip install pybind11
    ```
1. Install rlPx4Controller
    ```bash
    git clone git@github.com:FP-Flight/rlPx4Controller.git
    cd rlPx4Controller
    pip install -e .
    ```
### 3. Install AirGym
```bash
git clone git@github.com:FP-Flight/AirGym.git
cd AirGym/
pip install -e .

```
### 4. Test the installation
1. Run the example script:
    ```bash
    cd airgym/scripts
    python example.py --controller_test
    ```

## Training
We train the model by rl-games==1.6.1. 
```bash
python airgym/rl_games/runner.py --headless
```
Algorithm related parameters can be edited in `.yaml` files. Environment and simulator related parameters are located in ENV_config files like `X152bPx4_config.py`.