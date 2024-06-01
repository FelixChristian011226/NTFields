import numpy as np
import open3d as o3d
import sys
sys.path.append('./test')
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot
from my_algorithms.mesh_to_grid import mesh_to_voxel_grid
from timeit import default_timer as timer

voxel_num = 256

voxel_grid = mesh_to_voxel_grid("./datasets/gibson/0/mesh_z_up.obj", voxel_size=1/(voxel_num-1), scale_x=1/9.76, scale_y=1/6.44, scale_z=1/2.33)
# o3d.visualization.draw_geometries([voxel_grid])


X_dimensions = np.array([(0, voxel_num), (0, voxel_num), (0, voxel_num)])  # dimensions of Search Space
# obstacles
# Obstacles = np.array(
#     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
#      (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])

# 获取体素网格中的体素
voxel_coords = np.asarray(voxel_grid.get_voxels())

# 初始化障碍物列表
Obstacles = []

for voxel in voxel_coords:
    x, y, z = voxel.grid_index
    x1, y1, z1 = x + 1, y + 1, z + 1  # 体素的对角点
    Obstacles.append((x, y, z, x1, y1, z1))

# 转换为NumPy数组
Obstacles = np.array(Obstacles)
print("Obstacles num:", Obstacles.size/6)


x_init = (10, 10, 10)  # starting location
x_goal = (10, voxel_num-10, voxel_num-10)  # goal location

q = 1  # length of tree edges
r = 5  # length of smallest edge to check for intersection with obstacles
max_samples = 10240  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRT(X, q, x_init, x_goal, max_samples, r, prc)
start = timer()
path = rrt.rrt_search()
end = timer()
print("path",path)

print("time: ",end-start)
# plot
plot = Plot("rrt_3d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
