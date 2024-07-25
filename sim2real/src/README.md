# Control
## requirement
```bash
sudo apt install ros-noetic-mavros-extras
```

## 环境配置

```bash
pip install rl_games==1.6.1
```

### Eigen 库找不到
```
find_package(Eigen3 REQUIRED) # try to find manually installed eigen (Usually in /usr/local with provided FindEigen3.cmake)
message("Eigen lib find")

message(${EIGEN3_INCLUDE_DIRS})
# 头文件目录为 EIGEN3_INCLUDE_DIRS ，不要用错

```


