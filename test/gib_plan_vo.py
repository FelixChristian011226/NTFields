import sys
sys.path.append('.')
from models import model_3d as md
import torch
import os 
import numpy as np
import math
import torch
from torch import Tensor
from torch.autograd import Variable, grad

from timeit import default_timer as timer
import math

import igl
import open3d as o3d

    #except:
    #    continue
our_path=[]
our_time=[]
our_dis=[]
collision=0

modelPath = './Experiments/Gib'#arona,bolton,cabin,A_test
#modelPath = './Experiments/Gib_res_changelr_scale'

dataPath = './datasets/gibson/'#Arona,Cabin,Bolton#filePath = './Experiments/Gibson'

# 实例化一个模型对象并加载预训练模型
womodel    = md.Model(modelPath, dataPath, 3,[0,0], device='cuda')
womodel.load('./Experiments/Gib/Model_Epoch_01100_0.pt')

# 加载体素化点云数据
path = "datasets/gibson/0/voxelized_point_cloud_128res_20000points.npz"
occupancies = np.unpackbits(np.load(path)['compressed_occupancies'])
input = np.reshape(occupancies, (128,)*3)
grid = np.array(input, dtype=np.float32)
print(np.shape(grid))

# 将数据转换为 PyTorch 张量，并移到 GPU 上
grid = torch.from_numpy(grid).to('cuda:0').float()
grid = grid.unsqueeze(0)

# 获取环境特征
f_0, f_1 = womodel.network.env_encoder(grid)

# 机器人个数
robot_num = 5

# 定义多个起始点和终点
start_goals = [
    [-6,-4,-6,6,7,-2.5],
    [2,4,-6,8,2,-2.5],
    [-8,-2,-6,4,7,-2.5],
    [4,0,-6,5,6,-2.5],
    [-4,2,-6,9,-3,-2.5]
]

# 定义路径颜色
colors = [
    [1, 0, 0],  # 红色
    [0, 1, 0],  # 绿色
    [0, 0, 1],  # 蓝色
    [1, 1, 0],  # 黄色
    [1, 0, 1]   # 粉色
]

# 创建点云对象列表
point_clouds = []
index = 0
safe_dis = 0.06

# 初始化，存入起点终点，not_done数组记录机器人是否已经规划完毕，point[]记录路径点
XP = torch.zeros((robot_num, 6))  # 初始化一个形状为 (robot_num, 6) 的零张量
for i in range(robot_num):
    XP[i] = torch.tensor(start_goals[i])  # 将 start_goals 中的每个元素赋值给 XP 的每行
    XP[i] = XP[i]/20.0  # 对坐标进行归一化
not_done = [True] * robot_num
point = [[] for _ in range(robot_num)]

start = timer()

iter = 0
while any(not_done):
    print("iter",iter)

    # 遍历每个机器人，获得速度
    for i in range(robot_num):
        if(not_done[i]==False):
            continue

        # 将 XP[i] 变形为 (1, 6) 的形状
        XP_i = XP[i].view(1, 6).to('cuda')

        dis = torch.norm(XP_i[:, 3:6] - XP_i[:, 0:3])
        if dis < 0.06:
            print("robot", i, "done")
            not_done[i] = False
            continue

        # 梯度下降规划单步路径
        gradient = womodel.Gradient(XP_i.clone(), f_0, f_1)
        # XP_i = XP_i + 0.03 * gradient
        XP_i[:, :3] += 0.03 * gradient[:, :3]
        point[i].append(XP_i[:, 0:3])

        # 更新 XP[i]
        XP[i] = XP_i.view(-1).to('cpu')  # 将 XP_i 变回一维张量并移回 CPU

    # 第二次遍历，计算机器人间的碰撞，碰撞则进行调整
    for i in range(robot_num):
        if(not_done[i]==False):
            continue

        for j in range(i+1, robot_num):
            if(not_done[j]==False):
                continue

            XP1 = XP[i].view(1, 6).to('cuda')
            XP2 = XP[j].view(1, 6).to('cuda')

            rbt_dis = torch.norm(XP1[:, 0:3] - XP2[:, 0:3])
            if rbt_dis < safe_dis:
                print("collision between robot", i, "and robot", j)
                collision += 1

                # 计算调整量
                # adjust = 0.03 * (XP1[:, 0:3] - XP2[:, 0:3]) / dis
                adjust = (safe_dis-rbt_dis) *0.5 * (XP1[:, 0:3] - XP2[:, 0:3])

                XP1[:, 0:3] += adjust
                XP2[:, 0:3] -= adjust

                # 回退到上一步
                # point[i].pop()
                # point[j].pop()
                del point[i][-1]
                del point[j][-1]
                print("poped!")

                # 将调整后的路径点添加到列表中
                point[i].append(XP1[:, 0:3])
                point[j].append(XP2[:, 0:3])
                XP[i] = XP1.view(-1).to('cpu')
                XP[j] = XP2.view(-1).to('cpu')


    iter=iter+1
    if(iter>500):
        break

end = timer()

for i in range(robot_num):
    # 将路径点列表转换为 NumPy 数组，并按比例放大
    xyz = torch.cat(point[i]).to('cpu').data.numpy()
    xyz = 20 * xyz

    # 创建点云对象并设置颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector([colors[i]] * len(xyz))

    # 将点云对象添加到列表中
    point_clouds.append(pcd)

# 读取场景网格对象并进行缩放和法向量计算
mesh = o3d.io.read_triangle_mesh("datasets/gibson/0/mesh_z_up_scaled.off")
mesh.scale(20, center=(0,0,0))
mesh.compute_vertex_normals()

# 将所有点云对象和网格对象传递给 draw_geometries 函数进行可视化
o3d.visualization.draw_geometries([mesh] + point_clouds)
