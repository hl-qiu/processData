import json
import os
import re
import numpy as np
from math import cos, sin

def readPoseTxt(filename):
    pos_raw = {}
    with open(filename, "r", encoding='utf-8') as file:
        lines = file.readlines()
        i = 7  # 开始查找的行号
        image_names = []
        positions = []
        directions = []
        while i < len(lines):
            tmp = []
            if ".jpg" in lines[i]:
                image_name = re.findall(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}.{0,2}\.jpg)',
                                        lines[i])
                position_match = [float(kv.split('=')[1]) for kv in lines[i + 6].split('：')[1].split('，')]
                direction_match = [float(kv.split('=')[1]) for kv in lines[i + 7].split('：')[1].split('，')]

                image_names.append(image_name[0])
                positions.append(position_match)
                directions.append(direction_match)

                pos_raw[image_name[0]] = {
                    'position': np.array(position_match),
                    'direction': np.array(direction_match)
                }
            # 更新行号
            i += 20
        return pos_raw

def euler_to_rotation_matrix(omega, phi, kappa):
    R_x = np.array([[1, 0, 0],
                    [0, cos(omega), -sin(omega)],
                    [0, sin(omega), cos(omega)]])

    R_y = np.array([[cos(phi), 0, sin(phi)],
                    [0, 1, 0],
                    [-sin(phi), 0, cos(phi)]])

    R_z = np.array([[cos(kappa), -sin(kappa), 0],
                    [sin(kappa), cos(kappa), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def calculate_camera_pose(position, direction):
    omega = direction[0]
    phi = direction[1]
    kappa = direction[2]
    R = euler_to_rotation_matrix(omega, phi, kappa)
    t = np.array(position).reshape(3, 1)

    pose = np.hstack((R, t))
    pose = np.vstack((pose, [0,0,0,1]))
    return pose


if __name__ == '__main__':
    # # 读取pos.txt中的数据
    # pos_raw = readPoseTxt('pos.txt')
    # # 将欧拉角转为c2w矩阵
    # c2ws = {}
    # for image_name, value in pos_raw.items():
    #     c2w = calculate_camera_pose(value['position'], value['direction'])
    #     c2ws[image_name] = {
    #         'c2w': np.array(c2w)
    #     }
    # print(c2ws)
    #
    # # 读取图片文件夹，删除不需要的图片位姿
    # names = []
    # for filename in os.listdir('E:\恒纪元\\3D-Gaussian-Splatting\data\jianmo\jianmo\images'):
    #     names.append(filename)
    #
    # new_c2ws = c2ws.copy()
    # for image_name, c2w in c2ws.items():
    #     if image_name not in names:
    #         del new_c2ws[image_name]
    #
    # print(new_c2ws)
    #
    #
    # def convert_focal_length(focal_length_mm, sensor_width_mm, sensor_height_mm, ppa_x, ppa_y):
    #     # 计算水平和垂直像素尺寸
    #     pixel_size_w = ppa_x * sensor_width_mm
    #     pixel_size_h = ppa_y * sensor_height_mm
    #
    #     # 计算水平和垂直方向上的焦距（像素）
    #     focal_length_w = focal_length_mm / pixel_size_w
    #     focal_length_h = focal_length_mm / pixel_size_h
    #
    #     return focal_length_w, focal_length_h
    #
    #
    # # 内部参数
    # focal_length_mm = 3.069  # 焦距（毫米）
    # sensor_width_mm = 6.144  # 传感器宽度（毫米）
    # sensor_height_mm = 6.144  # 传感器高度（毫米）
    # ppa_x = 3e-06  # X方向上每平方毫米的像素数目
    # ppa_y = 3e-06  # Y方向上每平方毫米的像素数目
    #
    # # 调用函数进行转换
    # focal_length_w, focal_length_h = convert_focal_length(focal_length_mm, sensor_width_mm, sensor_height_mm, ppa_x,
    #                                                       ppa_y)
    #
    # print("水平焦距（像素）：", focal_length_w)
    # print("垂直焦距（像素）：", focal_length_h)
    #
    # data_list = {}
    # data_list["w"] = 1024
    # data_list["h"] = 1024
    # data_list["fl_x"] = focal_length_w
    # data_list["fl_y"] = focal_length_h
    # data_list["camera_model"] = "OPENCV"
    # data_list["frames"] = []
    # for idx, (image_name, v) in enumerate(new_c2ws.items()):
    #     data_list_tmp = {}
    #     data_list_tmp["file_path"] = os.path.join("images", image_name)
    #     data_list_tmp["transform_matrix"] = v['c2w'].tolist()
    #     # 将读取的JSON数据添加到列表中
    #     data_list["frames"].append(data_list_tmp)
    #
    # print(data_list)
    #
    # # 将列表中的JSON数据写入输出文件中
    # with open('transforms.json', 'w') as output_json_file:
    #     json.dump(data_list, output_json_file, indent=4)

    import laspy
    from plyfile import PlyData, PlyElement
    #
    # # 打开.LAS文件
    # in_las = laspy.file.File("point_cloud.las", mode="r")
    #
    # # 创建.PLY文件对象
    # vertices = [in_las.x, in_las.y, in_las.z]
    # vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    # if any(in_las.red):
    #     vertex_dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    #     colors = [in_las.red, in_las.green, in_las.blue]
    #     vertices += colors
    # vertex_data = np.array(list(zip(*vertices)), dtype=vertex_dtype)
    # vertex_element = PlyElement.describe(vertex_data, 'vertex')
    # plydata = PlyData([vertex_element])
    #
    # # 保存.PLY文件
    # plydata.write("point_cloud.ply")

    plydata = PlyData.read("point_cloud.ply")
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T


