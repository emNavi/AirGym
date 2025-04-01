#!/bin/bash

set -e  # 发生错误时终止脚本

# 获取 sudo 权限（会提示用户输入密码）
sudo -v

# 让 sudo 会话在脚本运行期间保持活动状态
while true; do
    sudo -n true
    sleep 60
    kill -0 "$$" || exit
done 2>/dev/null &

echo -e "\n\e[1;34m==============================\e[0m"
echo -e "\e[1;34m  Environment Setup Script  \e[0m"
echo -e "\e[1;34m==============================\e[0m\n"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATUS_FILE="$ROOT_DIR/install_status.log"

# 记录当前步骤
function mark_step_done() {
    echo "$1" >> "$STATUS_FILE"
}

# 检查步骤是否已完成
function is_step_done() {
    grep -q "$1" "$STATUS_FILE" 2>/dev/null
}

# 初始化状态文件
if [ ! -f "$STATUS_FILE" ]; then
    touch "$STATUS_FILE"
fi

# 1️⃣ 检查 Ubuntu 版本
STEP="Check Ubuntu version"
if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 1/7] Checking Ubuntu version...\e[0m"
    OS_VERSION=$(lsb_release -rs)
    if [ "$OS_VERSION" != "20.04" ]; then
        echo -e "\e[1;31mWarning: This script is only tested on Ubuntu 20.04.\e[0m"
    fi
    echo -e "\e[1;32m✔ Ubuntu version: $OS_VERSION\e[0m\n"
    mark_step_done "$STEP"
fi

# 2️⃣ 检查 Conda 是否安装
STEP="Check Conda installation"
if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 2/7] Checking Conda installation...\e[0m"
    if ! command -v conda &> /dev/null; then
        echo -e "\e[1;31mError: Conda is not installed. Please install Anaconda or Miniconda first.\e[0m"
        exit 1
    fi
    echo -e "\e[1;32m✔ Conda is installed.\e[0m\n"
    mark_step_done "$STEP"
fi

# 3️⃣ 创建 Conda 环境
STEP="Create Conda environment"
ENV_NAME=airgym
if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 3/7] Creating Conda environment...\e[0m"
    CONDA_DIR="$(conda info --base)"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"

    # 检查并移除已有的环境
    if conda env list | grep -q "$ENV_NAME"; then
        conda deactivate || true
        conda remove -y -n "${ENV_NAME}" --all
    fi

    conda env create -f ./airgym_conda_env.yml
    conda activate "${ENV_NAME}"

    echo -e "\e[1;32m✔ Conda environment is activated.\e[0m\n"
    mark_step_done "$STEP"
fi

# 4️⃣ 安装 PyTorch
STEP="Install PyTorch"
if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 4/7] Installing PyTorch...\e[0m"
    # conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install networkx==2.8.4
    pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch3d
    echo -e "\e[1;32m✔ PyTorch installed successfully.\e[0m"
    mark_step_done "$STEP"
fi

# 5️⃣ 安装 rlPx4Controller
STEP="Setup rlPx4Controller"
if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 5/7] Setting up rlPx4Controller...\e[0m"
    sudo apt install -y libeigen3-dev
    pip install pybind11

    RLPX4_CONTROLLER_F_DIR="$HOME"
    cd "$RLPX4_CONTROLLER_F_DIR"
    if [ ! -d "rlPx4Controller" ]; then
        git clone git@github.com:emNavi/rlPx4Controller.git
    fi
    cd rlPx4Controller
    pip install -e .

    echo -e "\e[1;32m✔ rlPx4Controller installed successfully.\e[0m"
    mark_step_done "$STEP"
fi

# 6️⃣ 安装 AirGym
STEP="Setup AirGym"
if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 6/7] Installing AirGym...\e[0m"
    cd "$ROOT_DIR"
    pip install usd-core rospkg matplotlib opencv-python tensorboardX
    pip install -e .
    echo -e "\e[1;32m✔ AirGym installed successfully.\e[0m"
    mark_step_done "$STEP"
fi

# 7️⃣ 下载并安装 Isaac Gym
STEP="Setup IsaacGym"
DOWNLOAD_URL="https://developer.nvidia.com/isaac-gym-preview-4"
ISAAC_GYM_F_DIR="$HOME"
ISAAC_GYM_DIR="$ISAAC_GYM_F_DIR/isaacgym"
TARBALL_NAME="isaacgym_preview4.tar.gz"

if ! is_step_done "$STEP"; then
    echo -e "\e[1;33m[Step 7/7] Setting up Isaac Gym...\e[0m"
    mkdir -p "$ISAAC_GYM_F_DIR"

    # 仅在未下载时执行下载
    if [ ! -f "$ISAAC_GYM_F_DIR/$TARBALL_NAME" ]; then
        echo "Downloading Isaac Gym Preview 4..."
        wget --progress=bar:force -O "$ISAAC_GYM_F_DIR/$TARBALL_NAME" "$DOWNLOAD_URL"
    fi

    # 仅在未解压时执行解压
    if [ ! -d "$ISAAC_GYM_DIR" ]; then
        echo -e "\e[1;32m✔ Download completed. Extracting...\e[0m"
        tar -xzf "$ISAAC_GYM_F_DIR/$TARBALL_NAME" -C "$ISAAC_GYM_F_DIR"
    fi

    pip install -e "$ISAAC_GYM_DIR/python"
    echo -e "\e[1;32m✔ Isaac Gym installed successfully.\e[0m"
    mark_step_done "$STEP"
fi

# 完成
echo -e "\e[1;34m==============================\e[0m"
echo -e "\e[1;34m  Installation Completed!  \e[0m"
echo -e "\e[1;34m==============================\e[0m\n"

# 删除同级目录下的 install_status.log
rm -f "$(dirname "$0")/install_status.log"