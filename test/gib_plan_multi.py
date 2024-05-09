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
meshes = []  # 存储网格对象
index = 0

for start_goal in start_goals:
    for i in range(5):
        # 定义起点和终点坐标
        XP = np.array([start_goal])
        XP = Variable(Tensor(XP)).to('cuda')
        XP=XP/20.0  # 对坐标进行归一化

        #print(XP)
        # 计算起点和终点间的距离
        dis=torch.norm(XP[:,3:6]-XP[:,0:3])

        start = timer()

        # 初始化路径点列表
        point0=[]
        point1=[]
        point0.append(XP[:,0:3])
        point1.append(XP[:,3:6])

        #print(id)

        # 迭代进行梯度下降路径规划，直到距离小于阈值或迭代次数超过500次
        iter=0
        while dis>0.06:
            # 计算梯度
            gradient = womodel.Gradient(XP.clone(), f_0, f_1)

            # 更新路径点坐标
            XP = XP + 0.03 * gradient

            # 计算新的起点和终点间的距离
            dis=torch.norm(XP[:,3:6]-XP[:,0:3])

            # 将路径点坐标添加到列表中
            point0.append(XP[:,0:3])
            point1.append(XP[:,3:6])

            iter=iter+1
            if(iter>500):
                break

        end = timer()
        print("time",end-start)

    # 反转终点数列，并和起点数列拼接
    point1.reverse()
    point=point0+point1

    # 反转终点数列，并和起点数列拼接
    point1.reverse()
    point=point0+point1

    # 将路径点列表转换为 NumPy 数组，并按比例放大
    xyz= torch.cat(point).to('cpu').data.numpy()
    xyz=20*xyz

    # 创建点云对象并设置颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector([colors[index]] * len(xyz))
    index = index+1
    # pcd.paint_uniform_color(colors[i])

    # 将点云对象添加到列表中
    point_clouds.append(pcd)  

# 读取场景网格对象并进行缩放和法向量计算
mesh = o3d.io.read_triangle_mesh("datasets/gibson/0/mesh_z_up_scaled.off")
mesh.scale(20, center=(0,0,0))
mesh.compute_vertex_normals()

# 将所有点云对象和网格对象传递给 draw_geometries 函数进行可视化
o3d.visualization.draw_geometries([mesh] + point_clouds)
