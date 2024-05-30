import numpy as np
import open3d as o3d
import sys
sys.path.append('./test')
from rrt_algorithms.rrt.rrt import RRT
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot

# 加载体素化点云数据
path = "datasets/gibson/0/voxelized_point_cloud_128res_20000points.npz"
occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,)*3)
grid = np.array(input, dtype=np.float32)

alpha = 0.15  # 比例阈值
Obstacles = []
grid_size = grid.shape[0]

copy_grid = np.copy(grid)

for axis in range(3):  # 分别固定x,y,z轴
    for coord in range(grid_size):
        for i in range(grid_size):
            for j in range(grid_size):
                done = False  # 检查是否遍历了整个平面
                # 分别赋值max和min的坐标
                min_dim1, max_dim1, min_dim2, max_dim2 = grid_size, 0, grid_size, 0

                for dim1 in range(grid_size):
                    for dim2 in range(grid_size):
                        if axis == 0:
                            has_obstacle = copy_grid[coord,dim1,dim2] == 1
                        elif axis == 1:
                            has_obstacle = copy_grid[dim1,coord,dim2] == 1
                        else:
                            has_obstacle = copy_grid[dim1,dim2,coord] == 1

                        if has_obstacle:
                            min_dim1 = min(min_dim1, dim1)
                            max_dim1 = max(max_dim1, dim1)
                            min_dim2 = min(min_dim2, dim2)
                            max_dim2 = max(max_dim2, dim2)

                        if dim1 == grid_size - 1 and dim2 == grid_size - 1:
                            done = True

                if done:  # 如果遍历了整个平面
                    n = np.count_nonzero(copy_grid[coord,min_dim1:max_dim1+1,min_dim2:max_dim2+1])
                    m = (max_dim1 - min_dim1 + 1)*(max_dim2 - min_dim2 + 1)
                    if n > m * alpha and min_dim1 > 0 and max_dim1 < grid_size - 1 and min_dim2 > 0 and max_dim2 < grid_size -1:
                        if axis == 0:
                            Obstacles.append((coord, min_dim1, min_dim2, coord+1, max_dim1+1, max_dim2+1))
                            copy_grid[coord,min_dim1:max_dim1+1,min_dim2:max_dim2+1] = 0
                        elif axis == 1:
                            Obstacles.append((min_dim1, coord, min_dim2, max_dim1+1, coord+1, max_dim2+1))
                            copy_grid[min_dim1:max_dim1+1,coord,min_dim2:max_dim2+1] = 0
                        else:
                            Obstacles.append((min_dim1, min_dim2, coord, max_dim1+1, max_dim2+1, coord+1))
                            copy_grid[min_dim1:max_dim1+1,min_dim2:max_dim2+1,coord] = 0
Obstacles = np.array(Obstacles)
print("Obstacle number:",Obstacles.size/6)



X_dimensions = np.array([(0, 128), (0, 128), (0, 128)])  # dimensions of Search Space
# obstacles
# Obstacles = np.array(
#     [(20, 20, 20, 40, 40, 40), (20, 20, 60, 40, 40, 80), (20, 60, 20, 40, 80, 40), (60, 60, 20, 80, 80, 40),
#      (60, 20, 20, 80, 40, 40), (60, 20, 60, 80, 40, 80), (20, 60, 60, 40, 80, 80), (60, 60, 60, 80, 80, 80)])



x_init = (10, 10, 10)  # starting location
x_goal = (110, 110, 110)  # goal location

q = 8  # length of tree edges
r = 5  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRT(X, q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()
print("path.length:",path)

# plot
plot = Plot("rrt_3d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
