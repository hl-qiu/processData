import math
import os
import json
import sys

import numpy as np


def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ], [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ], [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2
        ]
    ])


if __name__ == "__main__":
    # TODO colmap目录位置
    TEXT_FOLDER = "./colmap"
    cameras = {}
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        camera_angle_x = math.pi / 2
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            print(
                f"camera {camera_id}:\n\tres={camera['w'], camera['h']}\n\tcenter={camera['cx'], camera['cy']}\n\tfocal={camera['fl_x'], camera['fl_y']}\n\tfov={camera['fovx'], camera['fovy']}\n\tk={camera['k1'], camera['k2']} p={camera['p1'], camera['p2']} ")
            cameras[camera_id] = camera
    if len(cameras) == 0:
        print("No cameras found!")
        sys.exit(1)
    camera = cameras[camera_id]
    colmap2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    colmap_data = []
    with open(os.path.join(TEXT_FOLDER, "images.txt"), 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        elements = line.split()
        idx = int(elements[0])
        q0, q1, q2, q3, tx, ty, tz = map(float, elements[1:8])
        image_name = elements[9]

        # Reconstruct the rotation matrix from quaternion
        R = qvec2rotmat([q0, q1, q2, q3])

        # Transform from colmap to Blender coordinate system
        R_blender = (R @ colmap2blender[:3, :3]).T
        t_blender = -R_blender.T @ np.array([tx, ty, tz])

        # Create the transform matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_blender
        transform_matrix[:3, 3] = t_blender

        # Construct the frame information
        frame_info = {
            'file_path': image_name,
            'transform_matrix': transform_matrix
        }
        colmap_data.append(frame_info)

    # Create the inverse Blender transform file
    blender_json_data = {
        'camera_angle_x': 2 * math.atan(0.5 * camera["w"] / camera["fl_x"]),
        'camera_angle_y': 2 * math.atan(0.5 * camera["h"] / camera["fl_y"]),
        'frames': colmap_data
    }
    # TODO
    blender_json_path = os.path.join(TEXT_FOLDER, 'transforms_train.json')
    with open(blender_json_path, 'w') as f:
        json.dump(blender_json_data, f, indent=4)

    print(f'Transformed colmap data saved to {blender_json_path}')
