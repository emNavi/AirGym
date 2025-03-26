import os
import re
from xml.etree import ElementTree as ET

def modify_urdf_file(file_path):
    # 解析URDF文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 遍历所有的<cylinder>标签
    for cylinder in root.findall(".//cylinder"):
        # 获取radius属性
        radius = cylinder.get("radius")
        if radius:
            new_radius = float(radius) * 0.6
            # 更新radius属性
            cylinder.set("radius", str(new_radius))

    # 保存修改后的URDF文件
    tree.write(file_path, encoding="utf-8", xml_declaration=True)

def modify_urdf_files_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".urdf"):
            file_path = os.path.join(folder_path, filename)
            modify_urdf_file(file_path)
            print(f"Modified {filename}")

# 指定文件夹路径
folder_path = "thin"

# 调用函数修改文件夹中的URDF文件
modify_urdf_files_in_folder(folder_path)