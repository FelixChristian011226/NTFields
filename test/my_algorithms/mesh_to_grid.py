import open3d as o3d
import numpy as np

def mesh_to_voxel_grid(obj_file_path, voxel_size=1/127, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """
    将.obj文件转换为voxel grid。

    参数:
        obj_file_path (str): .obj文件的路径。
        voxel_size (float): 体素大小。

    返回:
        voxel_grid (o3d.geometry.VoxelGrid): 生成的体素网格。
    """
    # 读取三角网格
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    
    # 获取顶点并进行缩放
    vertices = np.asarray(mesh.vertices)
    
    # # 缩放比例
    # scale_x = 1/9.76
    # scale_y = 1/6.44
    # scale_z = 1/2.33

    # 按比例缩放顶点
    scaled_vertices = vertices * np.array([scale_x, scale_y, scale_z])
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())

    # 更新网格顶点
    mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)

    # 可视化缩放后的网格
    # o3d.visualization.draw_geometries([mesh])

    # 生成体素网格
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    # 可视化体素网格
    # o3d.visualization.draw_geometries([voxel_grid])

    # ------------ 计算每个方向的体素方格数量 ------------
    # min_bound = voxel_grid.get_min_bound()
    # max_bound = voxel_grid.get_max_bound()

    # 计算每个方向上的体素数量
    # num_voxels_x = int(np.ceil((max_bound[0] - min_bound[0]) / voxel_size))
    # num_voxels_y = int(np.ceil((max_bound[1] - min_bound[1]) / voxel_size))
    # num_voxels_z = int(np.ceil((max_bound[2] - min_bound[2]) / voxel_size))

    # print(f"X方向上的体素方格个数: {num_voxels_x}")
    # print(f"Y方向上的体素方格个数: {num_voxels_y}")
    # print(f"Z方向上的体素方格个数: {num_voxels_z}")

    return voxel_grid

# 例子:
if __name__ == "__main__":
    obj_file_path = "./datasets/gibson/0/mesh_z_up.obj"
    voxel_grid = mesh_to_voxel_grid(obj_file_path)
    print(voxel_grid)
    # 可视化体素网格
    o3d.visualization.draw_geometries([voxel_grid])