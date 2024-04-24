import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable, grad


# 创建数据集类 `_numpy2dataset`，用于将numpy数组转换为PyTorch数据集
class _numpy2dataset(torch.utils.data.Dataset):
    def __init__(self, points, speed, grid):
        # Creating identical pairs 将输入的点集和速度转换为PyTorch的`Tensor`类型，并使用`Variable`封装
        points = Variable(Tensor(points))
        speed  = Variable(Tensor(speed))
        # 将点集和速度连接起来形成数据
        self.data=torch.cat((points,speed),dim=1)
        self.grid  = Variable(Tensor(grid))

    # 将数据和网格发送到指定的设备上
    def send_device(self,device):
        self.data    = self.data.to(device)
        self.grid    = self.grid.to(device)

    # 获取指定索引的数据和索引值
    def __getitem__(self, index):
        data = self.data[index]
        return data, index

    # 获取数据集的长度
    def __len__(self):
        return self.data.shape[0]

def Database(PATH):
    
    try:
        points = np.load('{}/sampled_points.npy'.format(PATH))
        speed = np.load('{}/speed.npy'.format(PATH))
        occupancies = np.unpackbits(np.load('{}/voxelized_point_cloud_128res_20000points.npz'.format(PATH))['compressed_occupancies'])
        input = np.reshape(occupancies, (128,)*3)
        grid = np.array(input, dtype=np.float32)
        #print(tau.min())
    except ValueError:
        print('Please specify a correct source path, or create a dataset')
    rows=points.shape[0]
    
    print(points.shape,speed.shape)
    print(np.shape(grid))
    #print(XP.shape,YP.shape)
    database = _numpy2dataset(points,speed,grid)
    #database = _numpy2dataset(XP,YP)
    return database





