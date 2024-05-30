import sys
sys.path.append('.')
from models import model_3d as md
import torch
import os 
import numpy as np
import math
import open3d as o3d
from timeit import default_timer as timer
from scipy.spatial import KDTree
import random

# 加载模型和数据
modelPath = './Experiments/Gib'
dataPath = './datasets/gibson/'
womodel = md.Model(modelPath, dataPath, 3, [0,0], device='cuda')
womodel.load('./Experiments/Gib/Model_Epoch_01100_0.pt')

path = "datasets/gibson/0/voxelized_point_cloud_128res_20000points.npz"
occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,) * 3)
grid = np.array(input, dtype=np.float32)
grid = torch.from_numpy(grid).to('cuda:0').float().unsqueeze(0)
f_0, f_1 = womodel.network.env_encoder(grid)

robot_num = 5
start_goals = [
    [-6, -4, -6, 6, 7, -2.5],
    [-6, 4, -6, 8, 2, -2.5],
    [-8, -2, -6, 4, 7, -2.5],
    [4, 0, -6, 5, 6, -2.5],
    [-4, 2, -6, 9, -3, -2.5]
]
radius = [0.02, 0.015, 0.025, 0.04, 0.03]
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
point_clouds = []

# RRT参数
step_size = 0.02
max_iter = 20000

def is_colliding(point, grid, threshold=0.5):
    """Check if a point is in collision with obstacles."""
    x, y, z = (point * 64).astype(int)
    x, y, z = np.clip([x, y, z], 0, 127)  # 确保索引在有效范围内
    return grid[0, x, y, z] > threshold

def steer(from_node, to_node, step_size):
    """Steer from 'from_node' towards 'to_node' by 'step_size'."""
    direction = to_node - from_node
    length = np.linalg.norm(direction)
    direction = direction / length
    new_node = from_node + step_size * direction
    new_node = np.clip(new_node, -1, 1)  # 确保新节点在[-1, 1]范围内
    return new_node

def rrt_planner(start, goal, grid, step_size, max_iter):
    start = np.array(start[:3]) / 20.0
    goal = np.array(goal[:3]) / 20.0
    tree = KDTree([start])
    nodes = [start]
    parents = {0: None}
    for _ in range(max_iter):
        if _ == max_iter - 1:
            print("Max iterations reached!")
        rand_point = np.random.rand(3) * 2 - 1
        nearest_index = tree.query(rand_point, 1)[1]
        nearest_node = nodes[nearest_index]
        new_node = steer(nearest_node, rand_point, step_size)
        if not is_colliding(new_node, grid):
            nodes.append(new_node)
            parents[len(nodes) - 1] = nearest_index
            tree = KDTree(nodes)
            if np.linalg.norm(new_node - goal) < step_size:
                nodes.append(goal)
                parents[len(nodes) - 1] = len(nodes) - 2
                break

    path = [goal]
    current_index = len(nodes) - 1
    while parents[current_index] is not None:
        path.append(nodes[current_index])
        current_index = parents[current_index]
    path.append(start)
    path.reverse()
    return path

# 运行RRT算法为每个机器人生成路径
for i in range(robot_num):
    path = rrt_planner(start_goals[i][:3], start_goals[i][3:], grid, step_size, max_iter)
    xyz = np.array(path) * 20
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector([colors[i]] * len(xyz))
    point_clouds.append(pcd)

# 读取场景网格对象并进行缩放和法向量计算
mesh = o3d.io.read_triangle_mesh("datasets/gibson/0/mesh_z_up_scaled.off")
mesh.scale(20, center=(0,0,0))
mesh.compute_vertex_normals()

# 可视化
o3d.visualization.draw_geometries([mesh] + point_clouds)