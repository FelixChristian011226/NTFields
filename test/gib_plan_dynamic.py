import sys
sys.path.append('.')
from models import model_3d as md
import torch
import os 
import numpy as np
import math
from torch import Tensor
from torch.autograd import Variable, grad

from timeit import default_timer as timer
import math

import igl
import open3d as o3d

our_path = []
our_time = []
our_dis = []
collision = 0

modelPath = './Experiments/Gib'
dataPath = './datasets/gibson/'

womodel = md.Model(modelPath, dataPath, 3, [0, 0], device='cuda')
womodel.load('./Experiments/Gib/Model_Epoch_01100_0.pt')

path = "datasets/gibson/0/voxelized_point_cloud_128res_20000points.npz"
occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,) * 3)
grid = np.array(input, dtype=np.float32)
print(np.shape(grid))

grid = torch.from_numpy(grid).to('cuda:0').float()
grid = grid.unsqueeze(0)

f_0, f_1 = womodel.network.env_encoder(grid)

robot_num = 5

start_goals = [
    [-6, -4, -6, 6, 7, -2.5],
    [2, 4, -6, 8, 2, -2.5],
    [-8, -2, -6, 4, 7, -2.5],
    [4, 0, -6, 5, 6, -2.5],
    [-4, 2, -6, 9, -3, -2.5]
]

radius = [
    0.02,
    0.015,
    0.025,
    0.04,
    0.03
]

speed = [
    0.03,
    0.03,
    0.03,
    0.03,
    0.03
]

colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1]
]

point_clouds = []
index = 0
collision_detected = True
safe_dis = 0.00
safe_cst = 0.02
step = 0.5

XP = torch.zeros((robot_num, 6))
prev_XP = torch.zeros((robot_num, 6))
for i in range(robot_num):
    XP[i] = torch.tensor(start_goals[i])
    XP[i] = XP[i] / 20.0
    prev_XP[i] = XP[i].clone()
not_done = [True] * robot_num
point = [[] for _ in range(robot_num)]

start = timer()

iter = 0
while any(not_done):
    print("iter", iter)
    prev_XP = XP.clone()
    for i in range(robot_num):
        if not not_done[i]:
            continue

        XP_i = XP[i].view(1, 6).to('cuda')

        dis = torch.norm(XP_i[:, 3:6] - XP_i[:, 0:3])
        if dis < speed[i]:
            print("robot", i, "done")
            not_done[i] = False
            continue

        gradient = womodel.Gradient(XP_i.clone(), f_0, f_1)

        XP_i[:, :3] = XP_i[:, :3] + step * speed[i] * gradient[:, :3]
        point[i].append(XP_i[:, 0:3])

        XP[i] = XP_i.view(-1).to('cpu')

    collision_detected = True
    while collision_detected:
        collision_detected = False
        for i in range(robot_num):
            if not not_done[i]:
                continue

            for j in range(i + 1, robot_num):
                if not not_done[j]:
                    continue

                XP1 = XP[i].view(1, 6).to('cuda')
                XP2 = XP[j].view(1, 6).to('cuda')
                prev_XP1 = prev_XP[i].view(1, 6).to('cuda')
                prev_XP2 = prev_XP[j].view(1, 6).to('cuda')
                prev_v1 = torch.norm(XP1[:, 0:3] - prev_XP1[:, 0:3])
                prev_v2 = torch.norm(XP2[:, 0:3] - prev_XP2[:, 0:3])

                rbt_dis = float(torch.norm(XP1[:, 0:3] - XP2[:, 0:3]))
                safe_dis = radius[i] + radius[j] + safe_cst
                if rbt_dis < safe_dis:
                    collision_detected = True
                    print("collision between robot", i, "and robot", j)
                    collision += 1

                    adjust = (safe_dis - rbt_dis + 0.01) * 0.5 * (XP1[:, 0:3] - XP2[:, 0:3]) / rbt_dis

                    XP1[:, 0:3] = XP1[:, 0:3] + adjust
                    XP2[:, 0:3] = XP2[:, 0:3] - adjust

                    v1 = torch.norm(XP1[:, 0:3] - prev_XP1[:, 0:3])
                    v2 = torch.norm(XP2[:, 0:3] - prev_XP2[:, 0:3])

                    XP1[:, 0:3] = (XP1[:, 0:3] - prev_XP1[:, 0:3]) * prev_v1 / v1 + prev_XP1[:, 0:3]
                    XP2[:, 0:3] = (XP2[:, 0:3] - prev_XP2[:, 0:3]) * prev_v2 / v2 + prev_XP2[:, 0:3]

                    point[i].pop()
                    point[j].pop()

                    point[i].append(XP1[:, 0:3])
                    point[j].append(XP2[:, 0:3])
                    XP[i] = XP1.view(-1).to('cpu')
                    XP[j] = XP2.view(-1).to('cpu')

    iter = iter + 1
    if iter > 500:
        break

end = timer()

# Prepare point clouds for visualization
path_indices = [0] * robot_num
current_paths = [torch.cat(point[i][:1]).to('cpu').data.numpy() * 20 for i in range(robot_num)]
pcds = [o3d.geometry.PointCloud() for _ in range(robot_num)]
for i in range(robot_num):
    pcds[i].points = o3d.utility.Vector3dVector(current_paths[i])
    pcds[i].colors = o3d.utility.Vector3dVector([colors[i]] * len(current_paths[i]))

mesh = o3d.io.read_triangle_mesh("datasets/gibson/0/mesh_z_up_scaled.off")
mesh.scale(20, center=(0, 0, 0))
mesh.compute_vertex_normals()

def update_paths(vis):
    global path_indices, current_paths, pcds

    all_done = True
    for i in range(robot_num):
        if path_indices[i] < len(point[i]):
            all_done = False
            path_indices[i] += 1
            current_paths[i] = torch.cat(point[i][:path_indices[i]]).to('cpu').data.numpy() * 20
            pcds[i].points = o3d.utility.Vector3dVector(current_paths[i])
            pcds[i].colors = o3d.utility.Vector3dVector([colors[i]] * len(current_paths[i]))  # 重新设置颜色
    
    vis.clear_geometries()
    vis.add_geometry(mesh)
    for pcd in pcds:
        vis.add_geometry(pcd)
    
    vis.poll_events()
    vis.update_renderer()
    
    if all_done:
        return False  # Stop the animation
    return True  # Continue the animation


o3d.visualization.draw_geometries_with_animation_callback([mesh] + pcds, update_paths)
