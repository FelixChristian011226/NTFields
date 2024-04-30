import matplotlib
import numpy as np
import math
import random
import time

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch import Tensor
from torch.nn import Conv3d
from torch.autograd import Variable, grad
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from models import database as db

import matplotlib
import matplotlib.pylab as plt

import pickle5 as pickle 

from timeit import default_timer as timer

class NN(torch.nn.Module):
    
    def __init__(self, device, dim):#10
        super(NN, self).__init__()
        self.dim = dim

        h_size = 256 #512,256       # 定义隐藏层神经元数量，用于编码器和解码器
        fh_size = 128               # 定义环境特征的隐藏层神经元数量
        

        #3D CNN encoder
        # 输入1通道，输出16通道，卷积核大小3x3x3，填充1，填充模式为zeros，输出尺寸: 256 -> maxpooling -> 128
        self.conv_in = Conv3d(1, 16, 3, padding=1, padding_mode='zeros')  # out: 256 ->m.p. 128
        # 输入16通道，输出32通道，卷积核大小3x3x3，填充1，填充模式为zeros，输出尺寸: 128
        self.conv_0 = Conv3d(16, 32, 3, padding=1, padding_mode='zeros')  # out: 128
        
        self.actvn = torch.nn.ReLU()                    # 激活函数ReLU

        self.maxpool = torch.nn.MaxPool3d(2)            # 3D最大池化层，核大小为2x2x2

        self.conv_in_bn = torch.nn.BatchNorm3d(16)      # 对输入进行3D批量归一化，有助于加速网络训练过程。
        self.device = device

        feature_size = (1 +  16 ) * 7 #+ 3              # 特征大小计算，（输入通道数+输出通道数）* 池化后尺寸

        displacment = 0.0222#0.0222                     # 位移量
        displacments = []                               # 存储位移
        
        displacments.append([0, 0, 0])                  # 添加原始位置
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment              # 在x方向上增加位移
                displacments.append(input)

        #displacments: [[0, 0, 0], [-0.0222, 0, 0], [0.0222, 0, 0], [0, -0.0222, 0], [0, 0.0222, 0], [0, 0, -0.0222], [0, 0, 0.0222]]
        
        # 将位移转换为张量，并移到指定设备
        self.displacments = torch.Tensor(displacments).to(self.device)#cuda

        #decoder

        self.scale = 10     #尺度参数，用于解码器。在解码器中，我们将特征缩放10倍，以便更好地训练网络。

        self.act = torch.nn.ELU()   # 激活函数ELU。ELU是一种非线性激活函数，它在负值区域有一个非零梯度，有助于加速网络训练过程。

        self.nl1=5      # 编码器中的线性层数量
        self.nl2=7      # 解码器中的线性层数量

        '''
        ==编码器==
        encoder:  共nl1+1层。1层Linear(dim,h_size), nl1-1层Linear(h_size, h_size), 1层Linear(h_size, h_size)。
        encoder1: 共nl1层。1层Linear(dim,h_size), nl1-1层Linear(h_size, h_size)。
        '''
        self.encoder = torch.nn.ModuleList()                # 编码器的线性层。ModuleList是一个包含模块的列表，可以像列表一样进行迭代。
        self.encoder1 = torch.nn.ModuleList()
        
        self.encoder.append(Linear(dim,h_size))             # 添加第一个线性层
        self.encoder1.append(Linear(dim,h_size))
        
        for i in range(self.nl1-1):
            self.encoder.append(Linear(h_size, h_size))     # 添加隐藏层线性层
            self.encoder1.append(Linear(h_size, h_size)) 
        
        self.encoder.append(Linear(h_size, h_size))         # 添加最后一层线性层

        '''
        ==解码器==
        generator:  共nl2+2层。1层Linear(2*h_size + 2*fh_size, 2*h_size), nl2-1层Linear(2*h_size, 2*h_size), 1层Linear(2*h_size,h_size), 1层Linear(h_size,1)。
        generator1: 共nl2层。1层Linear(2*h_size + 2*fh_size, 2*h_size), nl2-1层Linear(2*h_size, 2*h_size)。
        '''
        self.generator = torch.nn.ModuleList()
        self.generator1 = torch.nn.ModuleList()

        self.generator.append(Linear(2*h_size + 2*fh_size, 2*h_size)) 
        self.generator1.append(Linear(2*h_size + 2*fh_size, 2*h_size)) 

        for i in range(self.nl2-1):
            self.generator.append(Linear(2*h_size, 2*h_size)) 
            self.generator1.append(Linear(2*h_size, 2*h_size)) 
        
        self.generator.append(Linear(2*h_size,h_size))
        self.generator.append(Linear(h_size,1))

        '''
        ==环境特征提取器==
        fc_env0: Linear(17*7, 128)。
        fc_env1: Linear(128, 128)。
        '''
        self.fc_env0 = Linear(feature_size, fh_size)
        self.fc_env1 = Linear(fh_size, fh_size)
    
    def init_weights(self, m):
        
        if type(m) == torch.nn.Linear:  # 如果是线性层
            stdv = (1. / math.sqrt(m.weight.size(1))/1.)*2  # 计算标准差。均匀分布初始化。防止模型陷入梯度消失或爆炸。
            #stdv = np.sqrt(6 / 64.) / self.T
            m.weight.data.uniform_(-stdv, stdv)
            m.bias.data.uniform_(-stdv, stdv)

    '''
    x: (1, 128, 128, 128)
    '''
    def env_encoder(self, x):
        x = x.unsqueeze(1)  # 在第1维度上增加一个维度。将输入张量转换为适合3D卷积层的格式，因为通常情况下，3D卷积层的输入要求是四维张量，batch size×channels×height×width。
        f_0 = x             # 输入特征，形状为(1, 1, 128, 128, 128)

        net = self.actvn(self.conv_in(x))   # 使用ReLU激活函数对卷积结果进行非线性变换
        net = self.conv_in_bn(net)          # 对卷积结果进行3D批量归一化
        f_1 = net
        return f_0, f_1 # f_0:(1, 1, 128, 128, 128), f_1:(1, 16, 128, 128, 128)
    
    def env_features(self, coords, f_0, f_1):
        # 计算环境特征
        # 大概这个coords是一个形状为(batch_size, 6)的张量，其中每个样本的前三个值是起点坐标，后三个值是终点坐标
        '''
        coords = torch.tensor([[ 0.5277, -1.7155,  0.4340,  0.3320, -0.2521, -0.2794],
                               [-0.3683, -0.4754, -0.0120, -1.0494,  0.4487,  0.4717],
                               [-0.7196, -0.4240, -0.0673,  0.4960, -0.4940, -0.4898]])
        '''
        
        coords = coords.clone().detach().requires_grad_(False)  # 复制并且不追踪梯度

        p0=coords[:,:3] # 张量中每个样本的从零数前三个值，提取起点坐标
        '''
        p0 = tensor([[ 0.5277, -1.7155,  0.4340],
                     [-0.3683, -0.4754, -0.0120],
                     [-0.7196, -0.4240, -0.0673]])
        '''
        p1=coords[:,3:] # 张量中每个样本的第三个之后的值，提取终点坐标
        '''
        p1 = tensor([[ 0.3320, -0.2521, -0.2794],
                     [-1.0494,  0.4487,  0.4717],
                     [ 0.4960, -0.4940, -0.4898]])
        '''

        size=p0.shape[0]    # 获取批量大小

        p = torch.vstack((p0,p1))   # 垂直方向拼接起点和终点坐标 p的形状为(2*batch_size, 3)
        '''
        p = tensor([[ 0.5277, -1.7155,  0.4340],
                    [-0.3683, -0.4754, -0.0120],
                    [-0.7196, -0.4240, -0.0673],
                    [ 0.3320, -0.2521, -0.2794],
                    [-1.0494,  0.4487,  0.4717],
                    [ 0.4960, -0.4940, -0.4898]])
        '''
        
        p = torch.index_select(p, 1, torch.LongTensor([2,1,0]).to(self.device)) # 重新排序坐标，将(x, y, z)调整为(z, y, x)
        '''
        p = tensor([[ 0.4340, -1.7155,  0.5277],
                    [-0.0120, -0.4754, -0.3683],
                    [-0.0673, -0.4240, -0.7196],
                    [-0.2794, -0.2521,  0.3320],
                    [ 0.4717,  0.4487, -1.0494],
                    [-0.4898, -0.4940,  0.4960]])
        '''

        p=2*p   # 坐标缩放
        '''
        p = tensor([[ 0.8680, -3.4310,  1.0554],
                    [-0.0240, -0.9508, -0.7366],
                    [-0.1346, -1.3440, -1.4392],
                    [-0.5588, -0.5042,  0.6640],
                    [ 0.9434,  0.8974, -2.0988],
                    [-0.9796, -0.9880,  0.9920]])
        '''
        
        p = p.unsqueeze(0)  # 增加维度，p的形状为(1, 2*batch_size, 3)

        p = p.unsqueeze(1).unsqueeze(1) # 增加维度，p的形状为(1, 1, 1, 2*batch_size, 3)
        '''
        p = tensor([[[[[ 0.8680, -3.4310,  1.0554],
                       [-0.0240, -0.9508, -0.7366],
                       [-0.1346, -1.3440, -1.4392],
                       [-0.5588, -0.5042,  0.6640],
                       [ 0.9434,  0.8974, -2.0988],
                       [-0.9796, -0.9880,  0.9920]]]]])
        '''

        p = torch.cat([p + d for d in self.displacments], dim=2)    # 将位移应用到坐标，p的形状为(1, 1, 7, 2*batch_size, 3)
        '''
        displacments: [[0, 0, 0], [-0.0222, 0, 0], [0.0222, 0, 0], [0, -0.0222, 0], [0, 0.0222, 0], [0, 0, -0.0222], [0, 0, 0.0222]]
        p = tensor([[[  [[ 0.8680, -3.4310,  1.0554],
                         [-0.0240, -0.9508, -0.7366],
                         [-0.1346, -1.3440, -1.4392],
                         [-0.5588, -0.5042,  0.6640],
                         [ 0.9434,  0.8974, -2.0988],
                         [-0.9796, -0.9880,  0.9920]],
                        [[ 0.8458, -3.4310,  1.0554],
                         [-0.0462, -0.9508, -0.7366],
                         [-0.1124, -1.3440, -1.4392],
                         [-0.5810, -0.5042,  0.6640],
                         [ 0.9656,  0.8974, -2.0988],
                         [-0.9574, -0.9880,  0.9920]],
                        [[ 0.8902, -3.4310,  1.0554],
                         [-0.0018, -0.9508, -0.7366],
                         [-0.1568, -1.3440, -1.4392],
                         [-0.5364, -0.5042,  0.6640],
                         [ 0.9212,  0.8974, -2.0988],
                         [-1.0018, -0.9880,  0.9920]],
                        [[ 0.8680, -3.4532,  1.0554],
                         [-0.0240, -0.9730, -0.7366],
                         [-0.1346, -1.3662, -1.4392],
                         [-0.5588, -0.5264,  0.6640],
                         [ 0.9434,  0.8752, -2.0988],
                         [-0.9796, -1.0102,  0.9920]],
                        [[ 0.8680, -3.4088,  1.0554],
                         [-0.0240, -0.9298, -0.7366],
                         [-0.1346, -1.2918, -1.4392],
                         [-0.5588, -0.4818,  0.6640],
                         [ 0.9434,  0.9418, -2.0988],
                         [-0.9796, -0.9658,  0.9920]],
                        [[ 0.8680, -3.4310,  1.0332],
                         [-0.0240, -0.9508, -0.7588],
                         [-0.1346, -1.3440, -1.4620],
                         [-0.5588, -0.5042,  0.6418],
                         [ 0.9434,  0.8974, -2.1210],
                         [-0.9796, -0.9880,  1.0142]],
                        [[ 0.8680, -3.4310,  1.0776],
                         [-0.0240, -0.9508, -0.7144],
                         [-0.1346, -1.3440, -1.4166],
                         [-0.5588, -0.5042,  0.6872],
                         [ 0.9434,  0.8974, -2.0754],
                         [-0.9796, -0.9880,  0.9698]]  ]]])
        
        '''

        #print(p.shape)
        # 使用双线性插值从输入特征图中提取特征
        '''
        grid_sample(input, grid, mode, padding_mode)
        input: 形状为(N, C, Din, Hin, Win)的输入特征图
        grid: 形状为(N, Dout, Hout, Wout, 3)的网格
        mode: 插值模式，可选值为'nearest'和'bilinear'
        padding_mode: 填充模式，可选值为'zeros'和'border'
        return: 输出特征图，形状为(N, C, Dout, Hout, Wout)
        '''
        feature_0 = F.grid_sample(f_0, p, mode='bilinear', padding_mode='border')
        feature_1 = F.grid_sample(f_1, p, mode='bilinear', padding_mode='border')
        # feature_0的形状为(N=1, C=1, Dout=1, Hout=7, Wout=2*batch_size) 
        # feature_1的形状为(N=1, C=16, Dout=1, Hout=7, Wout=2*batch_size)
        
        # 合并特征
        features = torch.cat((feature_0, feature_1), dim=1)
        # (1, 17, 1, 7, 2*batch_size)
        
        # 重新调整特征形状
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))
        # (1, 17*7, 2*batch_size)

        #print(features.size())
        # 去除不必要的维度
        features = torch.squeeze(features.transpose(1, -1))      
        # (2*batch_size, 17*7)               

        # 对特征应用激活函数 
        features = self.act(self.fc_env0(features))
        features = self.act(self.fc_env1(features))
        # (2*batch_size, 128), (2*batch_size, 128)

        # 将特征切分为起点和终点特征
        features0=features[:size,:]
        features1=features[size:,:]
        # (batch_size, 128), (batch_size, 128)
        
        return features0, features1

    def out(self, coords, features0, features1):
        # 输出函数，计算最终的输出
        # coords: (batch_size, 2*dim)
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        size = coords.shape[0]
        # 提取起点和终点坐标
        x0 = coords[:,:self.dim]
        x1 = coords[:,self.dim:]
        # 垂直方向拼接
        x = torch.vstack((x0,x1))
        # x: (2*batch_size, dim)
        
        # encoder[0]: Linear(dim,h_size=256)
        x  = self.act(self.encoder[0](x))
        # encoder[1~nl1-1]: Linear(h_size, h_size)
        # encoder1[1~nl1-1]: Linear(h_size, h_size)
        for ii in range(1,self.nl1):
            x_tmp = x
            x  = self.act(self.encoder[ii](x))
            x  = self.act(self.encoder1[ii](x) + x_tmp) 
        
        # encoder[-1]: Linear(h_size, h_size)
        x = self.encoder[-1](x)
        # x: (2*batch_size, h_size=256)

        # 将特征切分为起点和终点特征
        x0 = x[:size,...]
        x1 = x[size:,...]
        # (batch_size, h_size=256)
        
        x_0 = torch.max(x0,x1)
        x_1 = torch.min(x0,x1)
        # (batch_size, h_size=256)


        features_0 = torch.max(features0,features1)
        features_1 = torch.min(features0,features1)
        # (batch_size, fh_size=128)

        # 将特征和坐标拼接。
        x = torch.cat((x_0, x_1, features_0, features_1),1)
        # x: (batch_size, 2*h_size + 2*fh_size)
        
        # generator[0]: Linear(2*h_size + 2*fh_size, 2*h_size)
        x = self.act(self.generator[0](x)) 

        # generator[1~nl2-1]: Linear(2*h_size, 2*h_size)
        # generator1[1~nl2-1]: Linear(2*h_size, 2*h_size)
        for ii in range(1, self.nl2):
            x_tmp = x
            x = self.act(self.generator[ii](x)) 
            x = self.act(self.generator1[ii](x) + x_tmp) 
        
        # generator[-2]: Linear(2*h_size, h_size)
        y = self.generator[-2](x)
        x = self.act(y)     # ELU

        # generator[-1]: Linear(h_size, 1)
        y = self.generator[-1](x)
        x = torch.sigmoid(0.1*y) 
        
        return x, coords
        # x: (batch_size, 1)
        # coords: (batch_size, 2*dim)

    def forward(self, coords, grid):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        f_0, f_1 = self.env_encoder(grid)
        feature0, feature1= self.env_features(coords, f_0, f_1)
        output, coords = self.out(coords, feature0, feature1)
        
        return output, coords


class Model():
    def __init__(self, ModelPath, DataPath, dim, pos,device='cpu'):

        self.Params = {}
        self.Params['ModelPath'] = ModelPath
        self.Params['DataPath'] = DataPath
        self.dim = dim
        self.pos = pos

        # Pass the JSON information
        self.Params['Device'] = device

        self.Params['Network'] = {}

        self.Params['Training'] = {}
        self.Params['Training']['Batch Size'] = 10000
        self.Params['Training']['Number of Epochs'] = 1
        self.Params['Training']['Resampling Bounds'] = [0.2, 0.95]
        self.Params['Training']['Print Every * Epoch'] = 1
        self.Params['Training']['Save Every * Epoch'] = 10
        self.Params['Training']['Learning Rate'] = 2e-4#5e-5

        # Parameters to alter during training
        self.total_train_loss = []
    
    def gradient(self, y, x, create_graph=True):                                                               
                                                                                  
        grad_y = torch.ones_like(y) # 创建一个与y大小相同的全为1的张量grad_y，用于指定梯度的起始值

        '''
        autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
        outputs: 求导的因变量（需要求导的函数）
        inputs: 求导的自变量
        grad_outputs: 如果 outputs为标量,则grad_outputs=None;如果outputs是向量,则此参数必须写
        设output为y=f(x)=(y1,y2,...,ym)∈R^m,input为x=(x1,x2,...,xn)∈R^n,grad_outputs为g=(g1,g2,...,gm)∈R^m(都是向量)
        运算过程如下：
        Jacobi矩阵J=[∂y1/∂x1,∂y1/∂x2,...,∂y1/∂xn;
                    ∂y2/∂x1,∂y2/∂x2,...,∂y2/∂xn;
                    ... ... ... ... ... ... ...
                    ∂ym/∂x1,∂ym/∂x2,...,∂ym/∂xn]
        g=[g1,g2,...,gm]
        则output=J*g=[g1*∂y1/∂x1+g2*∂y2/∂x1+...+gm*∂ym/∂x1,
                     g1*∂y1/∂x2+g2*∂y2/∂x2+...+gm*∂ym/∂x2,
                     ... ... ... ... ... ... ...
                     g1*∂y1/∂xn+g2*∂y2/∂xn+...+gm*∂ym/∂xn]
        
        '''
        grad_x = torch.autograd.grad(y, x, grad_y, only_inputs=True, retain_graph=True, create_graph=create_graph)[0]
        # y: (batch_size, 1)
        # x: (batch_size, 2*dim)
        # grad_x: (batch_size, 2*dim)
        
        return grad_x  

    def Loss(self, points, features0, features1, Yobs):
        
        start=time.time()
        # out的输出为x:(batch_size, 1), coords:(batch_size, 2*dim)
        tau, Xp = self.network.out(points, features0, features1)
        dtau = self.gradient(tau, Xp)
        end=time.time()
        
        # 距离
        D = Xp[:,self.dim:]-Xp[:,:self.dim] #qg-qs
        # (batch_size, dim)
        
        # 矩阵点积，这里是计算平方，即距离的平方
        T0 = torch.einsum('ij,ij->i', D, D)
        # (batch_size)

        DT0=dtau[:,:self.dim]   #起点qs导数
        DT1=dtau[:,self.dim:]   #终点qg导数
        # (batch_size, dim)

        # T01=∇τ^2(τ在起点qs求导)
        # T02=-2τ∇τ·D(τ在起点qs求导)=-2∇τ(qg-qs)
        T01    = T0*torch.einsum('ij,ij->i', DT0, DT0)
        T02    = -2*tau[:,0]*torch.einsum('ij,ij->i', DT0, D)

        # T11=∇τ^2(τ在终点qg求导)
        # T12=2τ∇τ·D(τ在终点qg求导)=2∇τ(qg-qs)
        T11    = T0*torch.einsum('ij,ij->i', DT1, DT1)
        T12    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D)
        
        # T3=τ^2
        T3    = tau[:,0]**2 
        
        # S0=∇τ^2+2τ∇τ(qg-qs)+τ^2(τ在起点qs求导)
        # S1=∇τ^2-2τ∇τ(qg-qs)+τ^2(τ在终点qg求导)
        S0 = (T01-T02+T3)
        S1 = (T11-T12+T3)

        # 预测速度
        sq_Ypred0 = 1/torch.sqrt(torch.sqrt(S0)/T3)
        sq_Ypred1 = 1/torch.sqrt(torch.sqrt(S1)/T3)

        # 真实速度
        sq_Yobs0=torch.sqrt(Yobs[:,0])
        sq_Yobs1=torch.sqrt(Yobs[:,1])

        # 计算损失
        diff = abs(1-sq_Ypred0/sq_Yobs0)+abs(1-sq_Ypred1/sq_Yobs1)+\
            abs(1-sq_Yobs0/sq_Ypred0)+abs(1-sq_Yobs1/sq_Ypred1)

        # 计算均值
        loss_n = torch.sum(diff)/Yobs.shape[0]

        # 为什么返回两个一样的..
        loss = loss_n

        return loss, loss_n, diff

    def train(self):

        self.network = NN(self.Params['Device'],self.dim)
        self.network.apply(self.network.init_weights)
        #self.network.float()
        # 将模型移动到指定的设备上进行计算。
        self.network.to(self.Params['Device'])

        # 使用AdamW优化器
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.Params['Training']['Learning Rate']
            ,weight_decay=0.1)
        
        self.dataset = db.Database(self.Params['DataPath'])
                
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size']),
            num_workers = 0,
            shuffle=True)
        

        weights = Tensor(torch.ones(len(self.dataset))).to(
            torch.device(self.Params['Device']))
        
        # 计算数据集中每个样本的权重，用于加权采样。
        dists=torch.norm(self.dataset.data[:,0:3]-self.dataset.data[:,3:6],dim=1)
        weights = dists.max()-dists

        # 对计算出的权重进行截断，确保权重在一定范围内。
        weights = torch.clamp(
                weights/weights.max(), self.Params['Training']['Resampling Bounds'][0], self.Params['Training']['Resampling Bounds'][1])
        
        train_sampler_wei = WeightedRandomSampler(
                weights, len(weights), replacement=True)
            
        train_loader_wei = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.Params['Training']['Batch Size']),
            sampler=train_sampler_wei
        )

        # 速度
        speed = self.dataset.data[:,2*self.dim:]

        # 网格
        grid = self.dataset.grid
        grid = grid.to(self.Params['Device'])
        grid = grid.unsqueeze(0)
        print(speed.min())
        #'''
        
        weights = Tensor(torch.ones(len(self.dataset))).to(
                        torch.device(self.Params['Device']))
        
        prev_diff = 1.0
        current_diff = 1.0
        #step = 1.0
        tt =time.time()

        # 使用pickle深拷贝当前模型和优化器的状态。
        current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
        current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
        #p=(torch.rand((5,6))-0.5).cuda()
        prev_state_queue = []
        prev_optimizer_queue = []

        self.l0 = 500

        self.l1 = 500

        for epoch in range(1, self.Params['Training']['Number of Epochs']+1):
            total_train_loss = 0

            total_diff=0

            self.lamb = min(1.0,max(0,(epoch-self.l0)/self.l1))
            
            # 将当前模型状态和优化器状态加入到队列中
            prev_state_queue.append(current_state)
            prev_optimizer_queue.append(current_optimizer)
            # 最多记录5个
            if(len(prev_state_queue)>5):
                prev_state_queue.pop(0)
                prev_optimizer_queue.pop(0)
            
            # 将当前模型状态和优化器状态加入到历史状态队列中
            current_state = pickle.loads(pickle.dumps(self.network.state_dict()))
            current_optimizer = pickle.loads(pickle.dumps(self.optimizer.state_dict()))
            
            # 更新学习率
            self.optimizer.param_groups[0]['lr']  = max(5e-4*(1-epoch/self.l0),1e-5)
            
            prev_diff = current_diff
            iter=0
            while True:
                total_train_loss = 0
                total_diff = 0

                for i, data in enumerate(train_loader_wei, 0):#train_loader_wei,dataloader
                    
                    data=data[0].to(self.Params['Device'])
                    #data, indexbatch = data
                    points = data[:,:2*self.dim]#.float()#.cuda()
                    speed = data[:,2*self.dim:]#.float()#.cuda()

                    feature0=torch.zeros((points.shape[0],128)).to(self.Params['Device'])
                    feature1=torch.zeros((points.shape[0],128)).to(self.Params['Device'])
                    
                    # 如果 lamb 大于0，则计算环境特征，并对其进行加权
                    if self.lamb > 0:
                        f_0, f_1 = self.network.env_encoder(grid)
                        feature0, feature1 = self.network.env_features(points, f_0, f_1)
                        feature0 = feature0*self.lamb
                        feature1 = feature1*self.lamb

                    # 计算损失，并进行反向传播
                    loss_value, loss_n, wv = self.Loss(points, feature0, feature1, speed)
                    loss_value.backward()

                    # 更新参数并清零梯度
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # 累加训练损失和损失差异
                    total_train_loss += loss_value.clone().detach()
                    total_diff += loss_n.clone().detach()
                    
                    
                    del points, speed, loss_value, loss_n, wv#,indexbatch
                
                # 计算平均训练损失和平均损失差异
                total_train_loss /= len(dataloader)#dataloader train_loader
                total_diff /= len(dataloader)#dataloader train_loader

                # 更新当前损失和损失比例
                current_diff = total_diff
                diff_ratio = current_diff/prev_diff
            
                # 如果损失比例不在指定范围内，则进行模型参数的回滚
                if (diff_ratio < 1.2 and diff_ratio > 0):#1.5
                    break
                else:
                    iter+=1
                    with torch.no_grad():
                        random_number = random.randint(0, min(4,epoch-1))
                        self.network.load_state_dict(prev_state_queue[random_number], strict=True)
                        self.optimizer.load_state_dict(prev_optimizer_queue[random_number])
                    
                    print("RepeatEpoch = {} -- Loss = {:.4e}".format(
                        epoch, total_diff))
                
            # 将训练损失记录到列表中
            self.total_train_loss.append(total_train_loss)
  
            # 打印每个epoch的训练损失
            if epoch % self.Params['Training']['Print Every * Epoch'] == 0:
                with torch.no_grad():
                    print("Epoch = {} -- Loss = {:.4e}".format(
                        epoch, total_diff.item()))

            # 保存模型
            if (epoch % self.Params['Training']['Save Every * Epoch'] == 0) or (epoch == self.Params['Training']['Number of Epochs']) or (epoch == 1):
                self.plot(epoch,total_diff.item(),grid)
                with torch.no_grad():
                    self.save(epoch=epoch, val_loss=total_diff)

    def save(self, epoch='', val_loss=''):
        '''
            Saving a instance of the model
        '''
        torch.save({'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': self.total_train_loss,
                    'val_loss': self.total_train_loss}, '{}/Model_Epoch_{}_ValLoss_{:.6e}.pt'.format(self.Params['ModelPath'], str(epoch).zfill(5), val_loss))

    def load(self, filepath):
        
        checkpoint = torch.load(
            filepath, map_location=torch.device(self.Params['Device']))

        self.network = NN(self.Params['Device'],self.dim)

        self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.network.to(torch.device(self.Params['Device']))
        self.network.float()
        self.network.eval()

        
    def load_pretrained_state_dict(self, state_dict):
        own_state=self.state_dict

    '''
    计算时间
    T(qs,qg)=||qs-qg||/τ(qs,qg)
    '''
    def TravelTimes(self, Xp, feature0, feature1):
        Xp = Xp.to(torch.device(self.Params['Device']))
        
        tau, coords = self.network.out(Xp, feature0, feature1)
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim] #qg-qs
        
        T0 = torch.einsum('ij,ij->i', D, D)

        TT = torch.sqrt(T0)/tau[:, 0]

        del Xp, tau, T0
        return TT
    
    '''
    计算τ
    '''
    def Tau(self, Xp, feature0, feature1):
        Xp = Xp.to(torch.device(self.Params['Device']))
     
        tau, coords = self.network.out(Xp, feature0, feature1)
        
        return tau

    '''
    计算速度
    '''
    def Speed(self, Xp, feature0, feature1):
        Xp = Xp.to(torch.device(self.Params['Device']))

        tau, Xp = self.network.out(Xp, feature0, feature1)
        dtau = self.gradient(tau, Xp)        
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim] #qg-qs
        T0 = torch.einsum('ij,ij->i', D, D) #||qg-qs||^2

        DT1 = dtau[:,self.dim:] #终点导数

        T1    = T0*torch.einsum('ij,ij->i', DT1, DT1)   #∇τ^2(τ在终点qg求导)
        T2    = 2*tau[:,0]*torch.einsum('ij,ij->i', DT1, D) #2τ∇τ·D(τ在终点qg求导)=2∇τ(qg-qs)=-2∇τ(qs-qg)

        T3    = tau[:,0]**2
        
        S = (T1-T2+T3)  #∇τ^2-2τ∇τ(qs-qg)+τ^2(τ在终点qg求导)

        Ypred = T3 / torch.sqrt(S)
        
        del Xp, tau, dtau, T0, T1, T2, T3
        return Ypred
    
    def Gradient(self, Xp, f_0, f_1):
        Xp = Xp.to(torch.device(self.Params['Device']))
       
        #Xp.requires_grad_()
        feature0, feature1 = self.network.env_features(Xp, f_0, f_1)
        
        tau, Xp = self.network.out(Xp, feature0, feature1)
        dtau = self.gradient(tau, Xp)
        
        
        D = Xp[:,self.dim:]-Xp[:,:self.dim]
        T0 = torch.sqrt(torch.einsum('ij,ij->i', D, D))
        T3 = tau[:, 0]**2

        V0 = D
        V1 = dtau[:,self.dim:]
        
        Y1 = 1/(T0*tau[:, 0])*V0
        Y2 = T0/T3*V1


        Ypred1 = -(Y1-Y2)
        Spred1 = torch.norm(Ypred1)
        Ypred1 = 1/Spred1**2*Ypred1

        V0=-D
        V1=dtau[:,:self.dim]
        
        Y1 = 1/(T0*tau[:, 0])*V0
        Y2 = T0/T3*V1

        Ypred0 = -(Y1-Y2)
        Spred0 = torch.norm(Ypred0)

        Ypred0 = 1/Spred0**2*Ypred0
        
        return torch.cat((Ypred0, Ypred1),dim=1)
     
    def plot(self,epoch,total_train_loss, grid):
        limit = 0.5
        xmin     = [-limit,-limit]
        xmax     = [limit,limit]
        spacing=limit/40.0
        X,Y      = np.meshgrid(np.arange(xmin[0],xmax[0],spacing),np.arange(xmin[1],xmax[1],spacing))

        Xsrc = [0]*self.dim
        
        Xsrc[0] = self.pos[0]
        Xsrc[1] = self.pos[1]

        XP       = np.zeros((len(X.flatten()),2*self.dim))
        XP[:,:self.dim] = Xsrc
        XP[:,self.dim+0]  = X.flatten()
        XP[:,self.dim+1]  = Y.flatten()
        XP = Variable(Tensor(XP)).to(self.Params['Device'])

        feature0=torch.zeros((XP.shape[0],128)).to(self.Params['Device'])
        feature1=torch.zeros((XP.shape[0],128)).to(self.Params['Device'])
        
        if self.lamb > 0:
            f_0, f_1 = self.network.env_encoder(grid)
            feature0, feature1 = self.network.env_features(XP, f_0, f_1)
            feature0 = feature0*self.lamb
            feature1 = feature1*self.lamb
            
        
        tt = self.TravelTimes(XP,feature0, feature1)
        ss = self.Speed(XP,feature0, feature1)#*5
        tau = self.Tau(XP,feature0, feature1)
        
        TT = tt.to('cpu').data.numpy().reshape(X.shape)
        V  = ss.to('cpu').data.numpy().reshape(X.shape)
        TAU = tau.to('cpu').data.numpy().reshape(X.shape)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,V,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.Params['ModelPath']+"/plots"+str(epoch)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        quad1 = ax.pcolormesh(X,Y,TAU,vmin=0,vmax=1)
        ax.contour(X,Y,TT,np.arange(0,3,0.05), cmap='bone', linewidths=0.5)#0.25
        plt.colorbar(quad1,ax=ax, pad=0.1, label='Predicted Velocity')
        plt.savefig(self.Params['ModelPath']+"/tauplots"+str(epoch)+"_"+str(round(total_train_loss,4))+"_0.jpg",bbox_inches='tight')

        plt.close(fig)
