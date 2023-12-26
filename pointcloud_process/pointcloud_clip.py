from plyfile import PlyData, PlyElement
import numpy as np
import laspy


# TODO las 转 ply
def las2ply(path, outpath):
    # las 转 ply
    # 打开.LAS文件
    in_las = laspy.file.File(path, mode="r")
    # 创建.PLY文件对象
    vertices = [in_las.x, in_las.y, in_las.z]
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if any(in_las.red):
        vertex_dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        colors = [in_las.red, in_las.green, in_las.blue]
        vertices += colors
    vertex_data = np.array(list(zip(*vertices)), dtype=vertex_dtype)
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    plydata = PlyData([vertex_element])
    
    # 保存.PLY文件
    plydata.write(outpath)
    
# TODO 点云裁剪预处理（修改ply属性名）
# 将ply中的每个属性都加上'scalar_'前缀，除了['x','y','z','nx','ny','nz']
def preProcessing(path, outpath):
    # 读取PLY文件
    plydata = PlyData.read(path)
    points = plydata['vertex'].data
    new_names = []
    for name in points.dtype.names:
        if name not in ['x','y','z','nx','ny','nz']:
            name = 'scalar_' + name
        new_names.append(name)
        print("name = ", name)
    
    # 定义新的属性名列表
    # new_names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1',
    #              'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    new_dtype = np.dtype([(new_name, points.dtype[i]) for i, new_name in enumerate(new_names)])
    # 创建新的points数组，应用新的dtype
    new_points = np.array([tuple(row) for row in points], dtype=new_dtype)
    # 创建新的PlyElement
    new_element = PlyElement.describe(new_points, 'vertex')
    # 创建新的PlyData对象
    new_plydata = PlyData([new_element], text=plydata.text)
    # 将修改后的内容写回到PLY文件
    new_plydata.write(outpath)
    
    
# TODO 点云裁剪后处理（修改ply属性名）   
# 删去'scalar_'前缀，除了['x','y','z','nx','ny','nz']
def postProcessing(path, outpath):
    # 读取PLY文件
    plydata = PlyData.read(path)
    points = plydata['vertex'].data
    new_names = []  # 定义新的属性名列表
    for name in points.dtype.names:
        if name.startswith('scalar_'):
            name = name.split('_', 1)[1]
        new_names.append(name)
        print("name = ", name)
    
    new_dtype = np.dtype([(new_name, points.dtype[i]) for i, new_name in enumerate(new_names)])
    # 创建新的points数组，应用新的dtype
    new_points = np.array([tuple(row) for row in points], dtype=new_dtype)
    # 创建新的PlyElement
    new_element = PlyElement.describe(new_points, 'vertex')
    # 创建新的PlyData对象
    new_plydata = PlyData([new_element], text=plydata.text)
    # 将修改后的内容写回到PLY文件
    new_plydata.write(outpath)
if __name__ == '__main__':
    pass

    # preProcessing("E:\\恒纪元\\3D-Gaussian-Splatting\\data\\garden_sh0\\point_cloud\\iteration_30000\\point_cloud.ply", 
    #               "E:\\恒纪元\\3D-Gaussian-Splatting\\data\\garden_sh0\\point_cloud\\iteration_30000\\point_cloud_scalar.ply")
    # postProcessing("E:\\恒纪元\\3D-Gaussian-Splatting\\data\\garden_sh0\\point_cloud\\iteration_30000\\point_cloud_scalar_new.ply", 
    #                "E:\\恒纪元\\3D-Gaussian-Splatting\\data\\garden_sh0\\point_cloud\\iteration_30000\\point_cloud_final.ply")
    
    
    # 读取PLY文件
    plydata = PlyData.read("E:\\恒纪元\\3D-Gaussian-Splatting\\data\\bicycle\\point_cloud.ply")
    points = plydata['vertex'].data
    print(points)

    
