import os 
import numpy as np
from timeit import default_timer as timer
import igl
import traceback
import math
import torch
import pytorch_kinematics as pk

import bvh_distance_queries
import math

def arm_rand_sample_bound_points(numsamples, dim, 
                             vertices, faces,velocity_max, offset, margin,
                             out_path_ ,path_name_,end_effect_):
    numsamples = int(numsamples)

    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    chain = pk.build_serial_chain_from_urdf(
        open(out_path_+'/'+path_name_+".urdf").read(), end_effect_)
    chain = chain.to(dtype=dtype, device=d)

    scale = 0.9 * math.pi/0.5
    #base = torch.from_numpy(np.array([[0, -0.5*np.pi, 0.0, -0.5*np.pi,0.0,0.0]])).cuda()
    #vertices = vertices*2.5
    
    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(faces, dtype=torch.long, device='cuda')
    
    triangles = vertices[faces].unsqueeze(dim=0)

    print(vertices.shape)
    print(faces.shape)
    print(triangles.shape)

    X_list = []
    Y_list = []
    OutsideSize = numsamples + 10
    WholeSize = 0
    while OutsideSize > 0:
        P  = torch.rand((2*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((2*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((2*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(3.)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]
        #print(x0.shape[0])

        if(x0.shape[0]<=1):
            continue

        

        th_batch = scale*x0#+base
        whole_p = []
        batch_size = 80000
        for batch_id in range(math.floor(x0.shape[0]/batch_size)+1):
            if batch_id*batch_size==x0.shape[0]:
                break
            #print(batch_id)
            tg_batch = chain.forward_kinematics(
                th_batch[batch_id*batch_size:
                        min((batch_id+1)*batch_size,x0.shape[0]),:]
                        , end_only = False)

            p_list=[]
            iter = 0
            pointsize = 0
            for tg in tg_batch:
                print(iter,tg)
                if iter>1:
                    v = np.load(out_path_+'/meshes/collision/'+tg+'.npy')
                    nv = np.ones((v.shape[0],4))
                    pointsize = pointsize+v.shape[0]

                    nv[:,:3]=v[:,:3]
                    m = tg_batch[tg].get_matrix()
                    #print(m.shape)
                    t=torch.from_numpy(nv).float().cuda()
                    p=torch.matmul(m[:],t.T)
                    #p=p.cpu().numpy()
                    p = torch.permute(p, (0, 2, 1)).contiguous()
                    #p=np.transpose(p,(0,2,1))
                    p_list.append(p)
                    del m,p,t,nv, v
                iter = iter+1
            #print(pointsize)
            #p = np.concatenate(p_list,axis=1)
            p = torch.cat(p_list, dim=1)
            p = torch.reshape(p,(p.shape[0]*p.shape[1],p.shape[2])).contiguous()
            query_points = p[:,0:3].contiguous()
            query_points = 0.4*query_points.unsqueeze(dim=0)
            bvh = bvh_distance_queries.BVH()

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            distance, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
            torch.cuda.synchronize()
            
            distance = torch.sqrt(distance).squeeze()

            distance,_ = torch.min(torch.reshape(distance, (-1, pointsize)), dim=1)
            #distance = distance.detach().cpu().numpy()

            #print(distance.shape)
            whole_p.append(distance)
            del p, p_list, tg_batch, distance, query_points, bvh
        
        #unsigned_distance = np.concatenate(whole_p, axis=0)
        unsigned_distance = torch.cat(whole_p, dim=0)
        #print(unsigned_distance.shape)
        
        where_d          = (unsigned_distance <=  margin) & \
                                (unsigned_distance >=  offset)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = unsigned_distance[where_d]
        
        #print(x1.shape[0])
        if(x1.shape[0]<=1):
            continue

        #print('x1 ',x0.shape)
        th_batch = scale*x1#+base
        whole_p = []
        batch_size = 80000
        for batch_id in range(math.floor(x1.shape[0]/batch_size)+1):
            if batch_id*batch_size==x1.shape[0]:
                break
            #print(batch_id)
            tg_batch = chain.forward_kinematics(
                th_batch[batch_id*batch_size:
                        min((batch_id+1)*batch_size,x1.shape[0]),:]
                        , end_only = False)

            p_list=[]
            iter = 0
            pointsize = 0
            for tg in tg_batch:
                if iter>1:
                    #print(tg)
                    v = np.load(out_path_+'/meshes/collision/'+tg+'.npy')
                    nv = np.ones((v.shape[0],4))
                    pointsize = pointsize+v.shape[0]

                    nv[:,:3]=v[:,:3]
                    m = tg_batch[tg].get_matrix()
                    #print(m.shape)
                    t=torch.from_numpy(nv).float().cuda()
                    p=torch.matmul(m[:],t.T)
                    #p=p.cpu().numpy()
                    p = torch.permute(p, (0, 2, 1)).contiguous()
                    #p=np.transpose(p,(0,2,1))
                    p_list.append(p)
                    del m,p,t,nv, v
                iter = iter+1
            
            p = torch.cat(p_list, dim=1)
            p = torch.reshape(p,(p.shape[0]*p.shape[1],p.shape[2])).contiguous()
            query_points = p[:,0:3].contiguous()
            query_points = 0.4*query_points.unsqueeze(dim=0)
            
            bvh = bvh_distance_queries.BVH()

            torch.cuda.synchronize()
            torch.cuda.synchronize()
            distance, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
            torch.cuda.synchronize()
        
            distance = torch.sqrt(distance).squeeze()

            distance,_ = torch.min(torch.reshape(distance, (-1, pointsize)), dim=1)
            
            whole_p.append(distance)
            del p, p_list, tg_batch, distance, query_points, bvh

        unsigned_distance = torch.cat(whole_p, dim=0)

        y1 = unsigned_distance

        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        print(WholeSize)
        if(WholeSize > numsamples):
            break

    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]    
    
    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    
    speed0 = velocity_max*np.clip(distance[:,0] , a_min = offset, a_max = margin)/margin
    speed1 = velocity_max*np.clip(distance[:,1] , a_min = offset, a_max = margin)/margin
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0]=speed0
    speed[:,1]=speed1
    return sampled_points, speed

def point_rand_sample_bound_points(numsamples, dim, 
                             vertices, faces,velocity_max, offset, margin):
    numsamples = int(numsamples)

    # 把输入的顶点和面数据转换为PyTorch张量，并将它们传输到GPU上，为后续操作做准备。
    vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    faces = torch.tensor(faces, dtype=torch.long, device='cuda')
    # 根据给定的顶点和面数据构造三角形，并增加一个维度以用于后续批处理。
    triangles = vertices[faces].unsqueeze(dim=0)
    print(vertices.shape)
    print(faces.shape)
    print(triangles.shape)
    #X  = torch.zeros((numsamples,2*dim)).cuda()
    #Y  = torch.zeros((numsamples,2)).cuda()
    # 初始化一个空的列表用于存储采样的点对(X_list)和它们的速度(Y_list)
    X_list = []
    Y_list = []
    OutsideSize = numsamples + 2
    WholeSize = 0
    while OutsideSize > 0:
        # 生成随机点P和nP，后者是前者根据随机位移dP（归一化后，单位长度）和长度rL得到的，rL基于点距离原点的最远距离(根号3)随机选择。
        P  = torch.rand((2*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        dP = torch.rand((2*numsamples,dim),dtype=torch.float32, device='cuda')-0.5
        rL = (torch.rand((2*numsamples,1),dtype=torch.float32, device='cuda'))*np.sqrt(3.)
        nP = P + torch.nn.functional.normalize(dP,dim=1)*rL

        # 取nP在边界内的点
        PointsInside = torch.all((nP <= 0.5),dim=1) & torch.all((nP >= -0.5),dim=1)
        

        x0 = P[PointsInside,:]
        x1 = nP[PointsInside,:]

        print(x0.shape[0])
        if(x0.shape[0]<=1):
            continue
        #print(len(PointsOutside))
        # 构建x0的查询点，用于调用BVH算法计算这些点到最近表面的距离。这里的torch.cuda.synchronize()确保所有之前的CUDA操作都已经完成。
        query_points = x0
        query_points = query_points.unsqueeze(dim=0)
        #print(query_points.shape)
        bvh = bvh_distance_queries.BVH()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        # 执行BVH算法得到x0点到最近表面的距离，去掉无关的维度，并开方得到实际距离。
        distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
        torch.cuda.synchronize()
        unsigned_distance = torch.sqrt(distances).squeeze()
        
        # 从x0中去掉那些距离超出边界的点
        where_d          = (unsigned_distance <=  margin) & \
                                (unsigned_distance >=  offset)
        x0 = x0[where_d]
        x1 = x1[where_d]
        y0 = unsigned_distance[where_d]

        if(x1.shape[0]<=1):
            continue

        # 对于残余的移动后的点x1，重复与上面相同的过程，计算其到最近表面的距离。
        query_points = x1
        query_points = query_points.unsqueeze(dim=0)
        bvh = bvh_distance_queries.BVH()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        distances, closest_points, closest_faces, closest_bcs= bvh(triangles, query_points)
        torch.cuda.synchronize()
        #unsigned_distance = abs()
        y1 = torch.sqrt(distances).squeeze()

        # 合并满足条件的所有点对，更新控制变量，准备退出循环或进行下一次迭代。
        x = torch.cat((x0,x1),1)
        y = torch.cat((y0.unsqueeze(1),y1.unsqueeze(1)),1)

        X_list.append(x)
        Y_list.append(y)
        OutsideSize = OutsideSize - x.shape[0]
        WholeSize = WholeSize + x.shape[0]
        if(WholeSize > numsamples):
            break
        
        

    X = torch.cat(X_list,0)[:numsamples]
    Y = torch.cat(Y_list,0)[:numsamples]

    sampled_points = X.detach().cpu().numpy()
    distance = Y.detach().cpu().numpy()
    speed0 = velocity_max*np.clip(distance[:,0] , a_min = offset, a_max = margin)/margin
    speed1 = velocity_max*np.clip(distance[:,1] , a_min = offset, a_max = margin)/margin
    speed  = np.zeros((distance.shape[0],2))
    speed[:,0]=speed0
    speed[:,1]=speed1
    return sampled_points, speed


def sample_speed(path, numsamples, dim):
    
    try:

        global out_path
        out_path = os.path.dirname(path)
        #print(out_path)
        global path_name 
        path_name = os.path.splitext(os.path.basename(out_path))[0]
        #print('pp',path)
        global task_name 
        task_name = out_path.split('/')[2]#os.path.splitext(os.path.basename(out_path),'/')
        if task_name=='arm':
            #dim = np.loadtxt(out_path+'/dim')
            global end_effect
            with open(out_path+'/dim') as f:
                iter = 0
                for line in f:
                    data = line.split()
                    if iter==0:
                        dim = int(data[0])
                    else:
                        end_effect = data[0]
                        #print(end_effect)
                    iter=iter+1
        file_name = os.path.splitext(os.path.basename(path))[0]
        input_file = os.path.join(out_path,file_name + '_scaled.off')
        out_file = out_path + '/sampled_points.npy'

        print(input_file)
        if os.path.exists(out_file):
            print(f'Exists: {out_file}')
            #return
   
        #out_file = out_path + '/boundary_{}_samples.npz'.format( sigma)
        limit = 0.5
        xmin=[-limit]*dim
        xmax=[limit]*dim
        velocity_max = 1
        
        if task_name=='c3d' or task_name=='test':
            margin = limit/5.0
            offset = margin/10.0 
        elif task_name=='gibson':
            margin = limit/10.0
            offset = margin/7.0 
        elif task_name=='arm':
            margin = limit/10.0
            offset = margin/15.0 
        
        v, f = igl.read_triangle_mesh(input_file)


        start = timer()
        if task_name=='arm':
            sampled_points, speed = arm_rand_sample_bound_points(numsamples, dim, 
                    v, f, velocity_max, offset, margin, out_path ,path_name,end_effect)
        else:
            sampled_points, speed = point_rand_sample_bound_points(numsamples, dim, 
                    v, f, velocity_max, offset, margin)

        end = timer()

        print(end-start)

        np.save('{}/sampled_points'.format(out_path),sampled_points)
        np.save('{}/speed'.format(out_path),speed)
    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))
    
