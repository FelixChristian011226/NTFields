import open3d as o3d
import numpy as np

# 加载体素化点云数据
path = "datasets/gibson/0/voxelized_point_cloud_128res_20000points.npz"
occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,)*3)
grid = np.array(input, dtype=np.float32)

# 初始化点云对象并生成点
points = []
for x in range(grid.shape[0]):
    for y in range(grid.shape[1]):
        for z in range(grid.shape[2]):
            if grid[x, y, z] == 1:
                points.append([x, y, z])

points = np.array(points)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 将点云转换为体素网格
voxel_size = 1.0  # 每个体素的大小
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
    point_cloud, voxel_size, min_bound=(0, 0, 0), max_bound=(128, 128, 128)
)

# 设置体素颜色
# colors = np.array([[1, 0, 0]] * len(voxel_grid.get_voxels()))  # 红色
# for voxel in voxel_grid.get_voxels():
#     voxel_grid.colors[voxel.grid_index] = o3d.utility.Vector3dVector(colors)

# 可视化体素网格
o3d.visualization.draw_geometries([voxel_grid])