import os
import shutil

# 将图片和对应的位姿json文件保存到新的文件夹
def copy_images_and_jsons(source_folder, destination_folder):
    # 创建目标文件夹
    os.makedirs(destination_folder, exist_ok=True)
    # 获取源文件夹中所有jpg文件的路径列表
    image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]
    # 创建目标目录
    os.mkdir(destination_folder + '\\images')
    os.mkdir(destination_folder + '\\jsons')
    for image_file in image_files:
        # 构建jpg文件的完整路径
        image_path = os.path.join(source_folder, image_file)
        # 构建相同名称的json文件完整路径
        json_path = os.path.join(source_folder, image_file.replace('.jpg', '.json'))
        # 检查json文件是否存在
        if os.path.exists(json_path):
            # 将jpg文件和json文件复制到目标文件夹
            shutil.copy(image_path, os.path.join(destination_folder, 'images\\' + image_file))
            shutil.copy(json_path, os.path.join(destination_folder, 'jsons\\' + image_file))

    print("所有jpg文件及其同名的json文件已复制到目标文件夹。")


if __name__ == '__main__':
    # 1、提取图片和json文件
    source_folder = 'E:\恒纪元\\3D-Gaussian-Splatting\iPhone_lidar\\2023_10_08_15_51_49'  # 源文件夹的路径
    destination_folder = 'E:\恒纪元\\3D-Gaussian-Splatting\iPhone_lidar\\2023_10_08_15_51_49_new'  # 目标文件夹的路径
    copy_images_and_jsons(source_folder, destination_folder)
    # 2、将json位姿转换为transform.json文件格式
    # 读取内参
    # 读取图片名称
    # 根据图片id构建对应的json文件名称
    # 读取json文件，读取其中位姿信息

    # 写回到transforms.json文件
