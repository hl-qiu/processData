import numpy as np
from colmap_loader import *

# # 读取 npy 文件
# point_cloud = np.load("E:\恒纪元\\3D-Gaussian-Splatting\interp_aleks-teapot\\aleks-teapot\points.npy")
# print(point_cloud)

# TODO 从points3D.bin获取点云，并存储成npy格式
xyz, rgb, _ = read_points3D_binary(
    "E:\恒纪元\\3D-Gaussian-Splatting\data\wedding\wedding_2\colmap\sparse\\0\points3D.bin")
np.save("E:\恒纪元\\3D-Gaussian-Splatting\data\wedding\wedding_2\colmap\sparse\\0\points.npy", xyz)
