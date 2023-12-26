# 该脚本是为了从blender数据集的tranforms_train.json构造colmap的相机参数和图片参数数据集，以便使用指定相机视角的colmap进行重建。
# 参考：https://www.cnblogs.com/li-minghao/p/11865794.html
# 运行方法：python blender_camera2colmap.py

import numpy as np
import json
import os
import imageio
import math

# TODO: change image size
H = 800
W = 800

blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# 注意：最后输出的图片名字要按自然字典序排列，例：0, 1, 100, 101, 102, 2, 3...因为colmap内部是这么排序的
fnames = list(sorted(os.listdir('images')))
fname2pose = {}

with open('transforms_train.json', 'r') as f:
    meta = json.load(f)

fx = 0.5 * W / np.tan(0.5 * meta['camera_angle_x'])  # original focal length
if 'camera_angle_y' in meta:
    fy = 0.5 * H / np.tan(0.5 * meta['camera_angle_y'])  # original focal length
else:
    fy = fx
if 'cx' in meta:
    cx, cy = meta['cx'], meta['cy']
else:
    cx = 0.5 * W
    cy = 0.5 * H
with open('created/sparse/cameras.txt', 'w') as f:
    f.write(f'1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}')
    idx = 1
    for frame in meta['frames']:
        fname = frame['file_path'].split('/')[-1]
        if not (fname.endswith('.png') or fname.endswith('.jpg')):
            fname += '.png'
        # blend到opencv的转换：y轴和z轴方向翻转
        pose = np.array(frame['transform_matrix']) @ blender2opencv
        fname2pose.update({fname: pose})

with open('created/sparse/images.txt', 'w') as f:
    for fname in fnames:
        pose = fname2pose[fname]
        # 参考https://blog.csdn.net/weixin_44120025/article/details/124604229：colmap中相机坐标系和世界坐标系是相反的
        # blender中：world = R * camera + T; colmap中：camera = R * world + T
        # 因此转换公式为
        # R’ = R^-1
        # t’ = -R^-1 * t
        R = np.linalg.inv(pose[:3, :3])
        T = -np.matmul(R, pose[:3, 3])
        q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4 * q0)

        f.write(f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n')
        idx += 1

with open('created/sparse/points3D.txt', 'w') as f:
   f.write('')

