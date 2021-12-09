import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    求欧式距离就是利用这个公式    (a-b)^2=a^2+b^2-2ab
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # bx512x16
    new_points = points[batch_indices, idx.long(), :]

    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3] 通过FPS算法找到的点（质心）
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1]) # 4x512x1024 含义就是512个质心和所有点的距离
    "square_distance计算FPS算法得到的点（质心）和所有点的距离"
    sqrdists = square_distance(new_xyz, xyz)  # 4x512x1024 质心与所有点的距离
    group_idx[sqrdists > radius ** 2] = N  # 将不满足条件(在圆外面的点)的位置置成N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # 升序排序取前K个 # 4x512x16 就是满足条件的前k个点与所有选取质心的距离
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]  # 这一步不是很懂，为什么要吧超出范围的也算到这个面
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # Bx512的全零矩阵，每个点要找到512个质心
    distance = torch.ones(B, N).to(device) * 1e10  # 1x1024 值全部是1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # 随机从0-N生成四个随机数 尺寸为（4，）也就是随机初始四张图的四个质心的索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # 将前4个位置赋成最远的值（随机初始）
        "xyz[batch_indices, farthest, :]这个切片是为了拿到该批次中，每张图像对应质心的坐标"
        "也就是，将批次图像中随机初始的质心的索引从xyz中拿出它的坐标，具体可见草稿纸实验2"
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3) # 取4张图象中的质心坐标，并view()

        dist = torch.sum((xyz - centroid) ** 2, -1) # 每个点和初始的坐标的距离
        mask = dist < distance  # 找到小于该距离的掩码，也就是找离他比较近的点的索引
        distance[mask] = dist[mask]  # 通过索引找到距离符合要求的坐标（太远了的坐标就直接丢弃），并对distance赋值
        farthest = torch.max(distance, -1)[1]  # 找到所有点与随机初始点 最远点的索引
    return centroids
def sample_and_group(npoint, radius, nsample, xyz,points, knn,csa_p):
    """
    self.npoint, self.radius, self.nsample, xyz,self.knn
    Input:
        npoint:   质心点
        radius:    半径
        nsample:    邻居数
        xyz: input points position data, [B, N, 3]  坐标
        points: input points data, [B, N, D] # 特征点
    Return:
        new_xyz: sampled points position data,   采样得到的质心点 B Np C
        grouped_xyz_normal:   坐标邻域与质心偏移后的点  B Np  Ns C
        new_points: 采样的特征质心点   B Np C
        grouped_points_normal: 特征邻域与特征偏移后的点 B Np  Ns C
    """

    "KNN 计算一次的时间为0.0009968280792236328"
    "ball quary 计算一次时间为0.01596856117248535"
    B, N, C = xyz.shape
    Bf,Nf,Cf = points.shape
    S = npoint
    #fps_idx = tp.furthest_point_sample(xyz.contiguous(), npoint) # [B, npoint, C] # 通过FPS算法找到质心索引
    fps_idx = torch.as_tensor(np.random.choice(N, npoint, replace=True)).view(-1, npoint).repeat(B, 1)  # 1 (512,)
    new_xyz = index_points(xyz, fps_idx) # 通过索引找到质心坐标
    new_points = index_points(points,fps_idx) # 特征质心坐标
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K     2,256,3

    else:
        idx,dist= tp.ball_query(radius, nsample, xyz.contiguous(), new_xyz.contiguous())
    "idx为S个质心邻域内的K个点对应输入点的索引"
    #print(points.size())
    grouped_points = index_points(points, idx)  ### 找到质心对应特征区域 [B, npoint, nsample, C] n


    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C] n个质心 k个领域每个 3个坐标
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C) # 将质心与所有邻域的点相减 这里相当是绝对坐标转换为相对坐标 ，对xyz坐标进行平移,


    grouped_points_norm = grouped_points#  - new_points.view(Bf, S, 1, Cf)  #
    "这个方法里特征不进行偏移"

    #repeat_xyz = new_xyz[:,:,None,:].repeat(1,1,nsample,1)
    #repeat_points = new_points[:,:,None,:].repeat(1,1,16,1)

    cat_xyz = torch.cat([grouped_xyz_norm])
    if csa_p is not  None:
        csa_position = index_points(csa_p,idx)
        csa_position = torch.cat([csa_position, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
                  dim=-1)
    else:
        csa_position = cat_xyz
    csa_feature = torch.cat([grouped_points_norm, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
        dim=-1)  # B S neiber C+1

    return_all = False
    if return_all :
        return new_xyz,grouped_xyz,grouped_xyz_norm,new_points,grouped_points_norm
    else:
        return new_xyz, csa_position,csa_feature

    #new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D] 在最后一个维度进行拼接


class CSA_Layer(nn.Module):
    def __init__(self, channels,activate_function):
        super(CSA_Layer,self).__init__()
        "可以适当利用矩阵乘法降低通道数"
        self.q_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=False)

        self.v_conv = nn.Conv1d(channels, channels , 1,bias=False)
        self.trans_conv = nn.Conv1d(channels, channels, 1,bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = activate_function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature,position):
        "用 qk 来强化v  如果想要增强特征 那么qk应该是位置  如果想要增强位置，那么qk应该是特征"
        x_q = self.q_conv(position).permute(0, 2, 1)[:,:,:,None]  # b, n, c,1
        x_k = self.k_conv(position).permute(0, 2, 1)[:,:,None,:]  # b, n, 1,c
        x_v = self.v_conv(feature)
        energy = torch.matmul(x_q,x_k) # b, n, c c
        energy = torch.sum(energy,dim=2,keepdim=False) # b n c
        energy = energy / (1e-9 + energy.sum(dim=-1, keepdim=True))
        attention = self.softmax(energy).permute(0,2,1)

        x_r = torch.mul(attention,x_v)  # b, c, n

        x = feature + x_r#self.act(self.after_norm(self.trans_conv(feature)+ x_r))

        return x

class PointNetSetAbstraction(nn.Module):
    #
    def __init__(self, npoint, radius, nsample, in_channel, mlp,knn=True,activate_function=None):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint  # 中心点
        self.radius = radius  # 半径
        self.nsample = nsample  # 邻居数量
        self.knn = knn  # 是否启用knn
        self.activate_function = activate_function
        # in_channels = 3 + 1 *2
        # mlp = [8,16]

        self.feature = nn.Sequential(
            nn.Conv2d(in_channel+6,mlp[0],1,bias=False),nn.BatchNorm2d(mlp[0]),self.activate_function,
            nn.Conv2d(mlp[0], mlp[1], 1, bias=False),nn.BatchNorm2d(mlp[1]),self.activate_function,
            )
        self.position = nn.Sequential(
            nn.Conv2d(in_channel+6,mlp[0],1,bias=False),nn.BatchNorm2d(mlp[0]),self.activate_function,
            nn.Conv2d(mlp[0], mlp[1], 1, bias=False),nn.BatchNorm2d(mlp[1]),self.activate_function,
            )
        self.csa_feature = CSA_Layer(mlp[1],activate_function)
        self.csa_position = CSA_Layer(mlp[1],activate_function)


    def forward(self, xyz, points,csa_p=None):
        """
        Input:
            xyz: input points position data, [B, C, N] 第二次PointNetSetAbstraction输入的xyz为第一次找到的质心，
            points: input points data, [B, D, N] points为第一次的预测结果
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        # B C N  ---> B N C

        points = points.permute(0, 2, 1)
        if csa_p is not None:
            csa_p = csa_p.permute(0,2,1)

        new_xyz, cat_xyz, cat_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.knn,csa_p)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        "将新的特征点进行变换升维  new_points 进行转换后是对每个点的每个特征进行1x1卷积"
        new_points = cat_points.permute(0, 3, 2, 1) # [B, C+3, nsample,npoint]
        new_xyz_cat = cat_xyz.permute(0,3, 2, 1)

        feature = self.feature(new_points)
        position = self.position(new_xyz_cat)

        position = torch.max(position, dim=2)[0]
        feature = torch.max(feature, dim=2)[0]


        csa_feature = self.csa_feature(feature,position)
        csa_position = self.csa_position(position,feature)

        return new_xyz.permute(0, 2, 1), csa_feature,csa_position  # 本次寻找到的质心 和 新的特征



class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.k = 16 # 邻居数量
        self.total_points = 1024
        self.actiavte_function = nn.LeakyReLU(0.3,inplace=True)

        self.normal_channel = normal_channel

        self.first_f = nn.Sequential(nn.Conv1d(3, 32, 1, bias=False), nn.BatchNorm1d(32), self.actiavte_function
                                     , nn.Conv1d(32, 64, 1, bias=False), nn.BatchNorm1d(64), self.actiavte_function)

        self.first_p = nn.Sequential(nn.Conv1d(3, 32, 1, bias=False), nn.BatchNorm1d(32), self.actiavte_function,
                                     nn.Conv1d(32, 64, 1, bias=False), nn.BatchNorm1d(64), self.actiavte_function)

        self.sa1 = PointNetSetAbstraction(npoint=self.total_points // 4, radius=0.1, nsample=self.k,
                                          in_channel=64, mlp=[64, 128], activate_function=self.actiavte_function,
                                          knn=True)
        self.sa2 = PointNetSetAbstraction(npoint=self.total_points // 16, radius=0.2, nsample=self.k,
                                          in_channel=128, mlp=[128, 256], activate_function=self.actiavte_function,
                                          knn=True)
        self.sa3 = PointNetSetAbstraction(npoint=self.total_points // 64, radius=0.4, nsample=self.k,
                                          in_channel=256, mlp=[256, 512], activate_function=self.actiavte_function,
                                          knn=True)
        self.sa4 = PointNetSetAbstraction(npoint=self.total_points // 256, radius=0.6, nsample=self.k,
                                          in_channel=512, mlp=[512, 1024], activate_function=self.actiavte_function,
                                          knn=True)


        self.cat = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(2048,1024,1,bias=False),nn.BatchNorm1d(1024),self.actiavte_function
        )
        self.fc2 = nn.Linear(1024, 512,bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256,bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, num_class,bias=False)


    def forward(self, xyz):
        B, _, _ = xyz.shape  ############################################# 8,3,1024  # BCN
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]

        up_featrue = self.first_f(norm)
        up_position = self.first_p(xyz)

        l1_xyz, csa_feature1, csa_position1 = self.sa1(xyz, up_featrue, up_position)  # B C N 质心坐标和该层预测结果
        l2_xyz, csa_feature2,csa_position2 = self.sa2(l1_xyz, csa_feature1,csa_position1)
        l3_xyz, csa_feature3,csa_position3 = self.sa3(l2_xyz, csa_feature2,csa_position2)
        l4_xyz, csa_feature4, csa_position4 = self.sa4(l3_xyz, csa_feature3, csa_position3)

        cat_f_p = torch.cat([csa_feature4,csa_position4],dim=1)
        x = self.cat(cat_f_p)
        x = torch.max(x,dim=-1)[0]

        x = x.view(B, 1024)

        x = self.drop2(self.actiavte_function(self.bn2(self.fc2(x))))
        x = self.drop3(self.actiavte_function(self.bn3(self.fc3(x))))
        x = self.fc4(x) ##################################################### 8,10
        x = F.log_softmax(x, -1)
        return x



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    inputs = torch.randn((2,6,1024))
    o = get_model(10)
    print(o)
    print(o(inputs).size())
    list_ = [0.3,0.7,-1,2,1.2]
    tensor_list = torch.as_tensor(list_)
    softmax = nn.Softmax(dim=-1)
    print(softmax(tensor_list))
