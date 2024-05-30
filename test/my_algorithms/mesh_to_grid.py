import open3d as o3d
import numpy as np

print('input')
mesh = o3d.io.read_triangle_mesh("./datasets/gibson/0/mesh_z_up.obj")
# fit to unit cube
vertices = np.asarray(mesh.vertices)

# 3. 定义缩放比例
# scale_x = 1/48  # x 方向缩放比例
# scale_y = 1/32  # y 方向缩放比例
# scale_z = 1/12  # z 方向缩放比例

scale_x = 1/9.76  # x 方向缩放比例
scale_y = 1/6.44  # y 方向缩放比例
scale_z = 1/2.33  # z 方向缩放比例

# 4. 按比例缩放顶点
scaled_vertices = vertices * np.array([scale_x, scale_y, scale_z])
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())


# 5. 更新网格顶点
mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)

o3d.visualization.draw_geometries([mesh])

print('voxelization')
voxel_size = 1/127
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size)
o3d.visualization.draw_geometries([voxel_grid])

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