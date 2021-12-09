import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F



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
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # bx512x16
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
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(
        [B, S, 1])  # 4x512x1024 含义就是512个质心和所有点的距离
    "square_distance计算FPS算法得到的点（质心）和所有点的距离"
    sqrdists = square_distance(new_xyz, xyz)  # 4x512x1024 质心与所有点的距离
    group_idx[sqrdists > radius ** 2] = N  # 将不满足条件(在圆外面的点)的位置置成N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # 升序排序取前K个 # 4x512x16 就是满足条件的前k个点与所有选取质心的距离
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
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # Bx512的全零矩阵，每个点要找到512个质心
    distance = torch.ones(B, N).to(device) * 1e10  # 1x1024 值全部是1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 随机从0-N生成四个随机数 尺寸为（4，）也就是随机初始四张图的四个质心的索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # 将前4个位置赋成最远的值（随机初始）
        "xyz[batch_indices, farthest, :]这个切片是为了拿到该批次中，每张图像对应质心的坐标"
        "也就是，将批次图像中随机初始的质心的索引从xyz中拿出它的坐标，具体可见草稿纸实验2"
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3)  # 取4张图象中的质心坐标，并view()

        dist = torch.sum((xyz - centroid) ** 2, -1)  # 每个点和初始的坐标的距离
        mask = dist < distance  # 找到小于该距离的掩码，也就是找离他比较近的点的索引
        distance[mask] = dist[mask]  # 通过索引找到距离符合要求的坐标（太远了的坐标就直接丢弃），并对distance赋值
        farthest = torch.max(distance, -1)[1]  # 找到所有点与随机初始点 最远点的索引
    return centroids


class _make_stages(nn.Module):
    def __init__(self, in_channels, out_channels, bin_sz, activate_function):
        super(_make_stages, self).__init__()
        self.prior = nn.AvgPool1d(kernel_size=bin_sz, padding=bin_sz // 2, stride=1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = activate_function

    def forward(self, feature):
        """
        input B N C
        output B C N
        """
        f = self.prior(feature)  # B N C 2,128,512
        f = f.permute(0, 2, 1)  # B C N  2,1 128
        f = self.conv(f)
        f = self.relu(self.bn(f))
        return f


class _PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes, activate_function):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // (len(bin_sizes))
        # 通过这个列表表达式将backboned输出进行平均池化
        # bin_sizes是池化的大小，在这里等于[1,2,3,6]
        # 每次取出这个列表中的一个值，将他传给make_stages函数，得到一个全局平均池化后的结果
        # 通过for循环将四种尺寸的全局平局池化的结果保存到列表中 传入nn.MoudleList 生成网络
        self.stages = nn.ModuleList([_make_stages(in_channels, out_channels, b_s, activate_function)
                                     for b_s in bin_sizes])
        # 对四种尺寸的全局平均池化拼接后的结果调整通道数
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels + (out_channels * len(bin_sizes)), out_channels,
                      kernel_size=1),
            nn.BatchNorm1d(out_channels),
            activate_function,
            nn.Dropout2d(0.1))

    def forward(self, features):
        "B C N  ---> B N C 进行池化 然后再进行1d卷积"
        features_pooling = features.permute(0, 2, 1)  # B N C
        pyramids = [features]
        pyramids.extend([stage(features_pooling) for stage in self.stages])  # B C N
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

import numpy as np
def sample_and_group(npoint, radius, nsample, xyz, points, knn, csa_p):
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
    Bf, Nf, Cf = points.shape
    S = npoint
    "随机采样"
    fps_idx = torch.as_tensor(np.random.choice(N, npoint, replace=True)).view(-1,npoint).repeat(B,1) # 1 (512,)
    #fps_idx = farthest_point_sample(xyz.contiguous(), npoint)  # [B, npoint, C] # 通过FPS算法找到质心索引 # 1 torch.Size([4, 512])
    #print('1',fps_idx.shape)
   # print('fps shape',fps_idx.shape,'xyzshape',xyz.shape) # fps shape torch.Size([2, 512]) xyzshape torch.Size([2, 2048, 3])
    new_xyz = index_points(xyz, fps_idx)  # 通过索引找到质心坐标

    # new_points = index_points(points,fps_idx) # 特征质心坐标
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K     2,256,3

    else:

        idx = query_ball_point(radius, nsample, xyz.contiguous(), new_xyz.contiguous())
    "idx为S个质心邻域内的K个点对应输入点的索引"

    grouped_points = index_points(points, idx)  ### 找到质心对应特征区域 [B, npoint, nsample, C] n
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C] n个质心 k个领域每个 3个坐标

    "坐标进行偏移"
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 将质心与所有邻域的点相减 这里相当是绝对坐标转换为相对坐标 ，对xyz坐标进行平移,

    grouped_points_norm = grouped_points  # - new_points.view(Bf, S, 1, Cf)  #
    "这个方法里特征不进行偏移"

    # repeat_xyz = new_xyz[:,:,None,:].repeat(1,1,nsample,1)
    # repeat_points = new_points[:,:,None,:].repeat(1,1,16,1)

    if csa_p is not None:
        csa_position = index_points(csa_p, idx)
        csa_position = torch.cat([csa_position, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
                                 dim=-1)
    else:
        csa_position = grouped_xyz_norm
    csa_feature = torch.cat(
        [grouped_points_norm, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
        dim=-1)  # B S neiber C+1

    return_all = False
    if return_all:
        return new_xyz, grouped_xyz, grouped_xyz_norm, grouped_points_norm
    else:
        return new_xyz, csa_position, csa_feature, grouped_xyz_norm,fps_idx
    # new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D] 在最后一个维度进行拼接


class CSA_Layer(nn.Module):
    def __init__(self, channels, activate_function):
        super(CSA_Layer, self).__init__()
        "可以适当利用矩阵乘法降低通道数"
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)

        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.trans_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = activate_function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, position):
        "用 qk 来强化v  如果想要增强特征 那么qk应该是位置  如果想要增强位置，那么qk应该是特征"
        x_q = self.q_conv(position).permute(0, 2, 1)[:, :, :, None]  # b, n, c,1
        x_k = self.k_conv(position).permute(0, 2, 1)[:, :, None, :]  # b, n, 1,c
        x_v = self.v_conv(feature)
        energy = torch.matmul(x_q, x_k)  # b,n c c
        energy = torch.sum(energy, dim=-2, keepdim=False)  # b n c
        #energy = self.trans_conv(energy.permute(0, 2, 1))
        energy = energy / (1e-9 + energy.sum(dim=-1, keepdim=True))
        attention = self.softmax(energy.permute(0, 2, 1)) # bcn
        #
        x_r = torch.mul(attention, x_v)  # b, c, n
        x = (x_r + feature)
        # x = self.act(self.after_norm(self.trans_conv(feature + x_r)))

        return x


class PointNetSetAbstraction(nn.Module):
    #
    def __init__(self, npoint, radius, nsample, in_channel, mlp, knn=True, activate_function=None):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint  # 中心点
        self.radius = radius  # 半径
        self.nsample = nsample  # 邻居数量
        self.knn = knn  # 是否启用knn
        self.activate_function = activate_function
        # in_channels = 3 + 1 *2
        # mlp = [8,16]

        self.feature = nn.Sequential(
            nn.Conv2d(in_channel + 6, mlp[0], 1, bias=False), nn.BatchNorm2d(mlp[0]), self.activate_function,
            nn.Conv2d(mlp[0], mlp[1], 1, bias=False), nn.BatchNorm2d(mlp[1]), self.activate_function,

        )
        self.position = nn.Sequential(
            nn.Conv2d(in_channel + 6, mlp[0], 1, bias=False), nn.BatchNorm2d(mlp[0]), self.activate_function,
            nn.Conv2d(mlp[0], mlp[1], 1, bias=False), nn.BatchNorm2d(mlp[1]), self.activate_function,

        )

        self.csa_feature = CSA_Layer(mlp[1], activate_function)
        self.csa_position = CSA_Layer(mlp[1], activate_function)
        self.at1 = Attention(mlp[1], self.activate_function)
        self.at2 = Attention(mlp[1], self.activate_function)
    def forward(self, xyz, points, csa_p=None):
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
            csa_p = csa_p.permute(0, 2, 1)

        new_xyz, cat_xyz, cat_points, grouped_xyz_norm,fps_idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz,
                                                                          points, self.knn, csa_p)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        "将新的特征点进行变换升维  new_points 进行转换后是对每个点的每个特征进行1x1卷积"
        new_points = cat_points.permute(0, 3, 2, 1)  # [B, C+3, nsample,npoint]
        new_xyz_cat = cat_xyz.permute(0, 3, 2, 1)

        feature = self.feature(new_points)

        position = self.position(new_xyz_cat)

        position = torch.max(position, dim=2)[0]
        #print(position.shape)
        position = self.at1(position)
        feature = torch.max(feature, dim=2)[0]
        feature = self.at2(feature)

        csa_feature = self.csa_feature(feature, position)
        csa_position = self.csa_position(position, feature)

        return new_xyz.permute(0, 2, 1), csa_feature, csa_position,fps_idx  # 本次寻找到的质心 和 新的特征


class PointNetFeaturePropagation(nn.Module):
    #                 in_channel=1280, mlp=[256, 256]
    def __init__(self, in_channel, mlp, last, activate_function, up_npoint):
        super(PointNetFeaturePropagation, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.activate_function = activate_function
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.transform_low_level = nn.Sequential(
            nn.Conv1d(last, in_channel, 1, bias=False), nn.BatchNorm1d(in_channel),self.activate_function
        )

    def forward(self, xyz1, xyz2, points1, points2):
        #            前面两层的质心和前面两层的输出
        """
        Input:
            利用前一层的点对后面的点进行插值
            xyz1: input points position data, [B, C, N]  l2层输出 xyz
            xyz2: sampled input points position data, [B, C, S]  l3层输出  xyz
            points1: input points data, [B, D, N]  l2层输出  points
            points2: input points data, [B, D, S]  l3层输出  points

        Return:
            new_points: upsampled points data, [B, D', N]
        """
        "  将B C N 转换为B N C 然后利用插值将高维点云数目S 插值到低维点云数目N (N大于S)"
        "  xyz1 低维点云  数量为N   xyz2 高维点云  数量为S"
        xyz1 = xyz1.permute(0, 2, 1)  # 第一次插值时 2,3,128 ---> 2,128,3 | 第二次插值时 2,3,512--->2,512,3 B N C
        xyz2 = xyz2.permute(0, 2, 1)  # 第一次插值时2,3,1  ---> 2 ,1,3    |  第二次插值时 2,3,128--->2,128,3

        points2 = points2.permute(0, 2, 1)  # 第一次插值时2,1021,1  --->2,1,1024  最后低维信息，压缩成一个点了  这个点有1024个特征
        # 第二次插值 2，256，128 --->2,128,256
        _, _, C2 = points2.shape

        B, N, C = xyz1.shape  # N = 128   低维特征的点云数  （其数量大于高维特征）
        _, S, _ = xyz2.shape  # s = 1   高维特征的点云数

        if S == 1:
            "如果最后只有一个点，就将S直复制N份后与与低维信息进行拼接"
            interpolated_points = points2.repeat(1, N, 1)  # 2,128,1024 第一次直接用拼接代替插值
        else:
            "如果不是一个点 则插值放大 128个点---->512个点"
            "此时计算出的距离是一个矩阵 512x128 也就是512个低维点与128个高维点 两两之间的距离"
            dists = square_distance(xyz1, xyz2)  # 第二次插值 先计算高维与低维的距离 2,512,128
            dists, idx = dists.sort(dim=-1)  # 2,512,128 在最后一个维度进行排序 默认进行升序排序，也就是越靠前的位置说明 xyz1离xyz2距离较近
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3] 2,512,3 此时dist里面存放的就是 xyz1离xyz2最近的3个点的距离
            dist_recip = 1.0 / (dists + 1e-8)  # 求距离的倒数 2,512,3 对应论文中的 Wi(x)
            "对dist_recip的倒数求和 torch.sum   keepdim=True 保留求和后的维度  2,512,1"
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 也就是将距离最近的三个邻居的加起来  此时对应论文中公式的分母部分
            weight = dist_recip / norm  # 2,512,3
            # print('weight size',weight.shape) #  torch.Size([16, 2048, 3])
            # print('points2 size',points2.shape) # torch.Size([16, 512, 128])
            #print('index_points(points2, idx)',index_points(points2, idx).shape) # torch.Size([16, 512,3, 128])
            # print('---------------------------')
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = (interpolated_points.permute(0, 2, 1) +  points1)
        else:
            new_points = interpolated_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.activate_function(bn(conv(new_points)))

        return new_points
class Attention(nn.Module):
    def __init__(self,in_channel,activate_function):
        super(Attention, self).__init__()
        self.trans = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,out_channels=in_channel//4,kernel_size=1,bias=False),nn.BatchNorm1d(in_channel//4),
            activate_function,
            nn.Conv1d(in_channels=in_channel// 4, out_channels=in_channel , kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        # x b c n
        #print('x',x.shape)
        y = x
        x_t = x.permute(0,2,1) # b n c
        att = torch.bmm(x,x_t) # b c c
        att = torch.sum(att,dim=-1,keepdim=True)
        att = self.trans(att)
        return y*att

class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        # in_channel = 6 if normal_channel else 3
        self.k = 16  # 邻居数量
        self.total_points = 2048
        self.actiavte_function = nn.LeakyReLU(0.3, inplace=True)
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

            nn.Conv1d(2048,256, 1, bias=False), nn.BatchNorm1d(256), self.actiavte_function,Attention(256,self.actiavte_function))
        # # self.seg_head = nn.Sequential(_PSPModule(1024,bin_sizes=self.bin_size,activate_function=self.actiavte_function),)

        self.up_sample1 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=512 * 2,
                                                     activate_function=self.actiavte_function,
                                                     up_npoint=self.total_points // 64)  # 64  - 16
        self.up_sample2 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=256 * 2,
                                                     activate_function=self.actiavte_function,
                                                     up_npoint=self.total_points // 16)  # 16 - 4
        self.up_sample3 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=128 * 2,
                                                     activate_function=self.actiavte_function,
                                                     up_npoint=self.total_points // 4)  # 4 -2
        self.up_sample4 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=64 * 2,
                                                     activate_function=self.actiavte_function,
                                                     up_npoint=self.total_points)

        self.msa0 = nn.Sequential(nn.Conv1d(64+64,128*2,1,bias=False),nn.BatchNorm1d(256),self.actiavte_function,
                                 nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), self.actiavte_function,
                                  Attention(256,self.actiavte_function))

        self.msa1 = nn.Sequential(nn.Conv1d(64*2+128*2,192,1,bias=False),nn.BatchNorm1d(192),self.actiavte_function,
                                 nn.Conv1d(192, 256, 1, bias=False), nn.BatchNorm1d(256), self.actiavte_function,Attention(256,self.actiavte_function))

        self.msa2 = nn.Sequential(nn.Conv1d(256+256+256, 512 , 1, bias=False), nn.BatchNorm1d(512),self.actiavte_function,
                                nn.Conv1d(512 , 256, 1, bias=False), nn.BatchNorm1d(256),self.actiavte_function,Attention(256,self.actiavte_function))

        self.msa3 = nn.Sequential(nn.Conv1d(256 + 512+512, 1024, 1, bias=False), nn.BatchNorm1d(1024), self.actiavte_function,
                                  nn.Conv1d(1024, 256, 1, bias=False), nn.BatchNorm1d(256),self.actiavte_function,Attention(256,self.actiavte_function) )

        self.last = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128),self.actiavte_function,
                                  nn.Dropout(0.2), nn.Conv1d(128, num_class, 1,bias=False ))
    def forward(self, xyz, trans=None):
        B, C, N = xyz.shape  ############################################# 8,3,1024  # BCN
       # assert self.total_points == N, '输入点数与输出不相等'
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]

        up_featrue = self.first_f(norm)
        up_position = self.first_p(xyz)

        l1_xyz, csa_feature1, csa_position1, fps_idx1 = self.sa1(xyz, up_featrue, up_position)  # N = 512  f1=128
        l2_xyz, csa_feature2, csa_position2, fps_idx2 = self.sa2(l1_xyz, csa_feature1, csa_position1)  # 128  # f2=256
        l3_xyz, csa_feature3, csa_position3, fps_idx3 = self.sa3(l2_xyz, csa_feature2, csa_position2)  # 32  # f3=512
        l4_xyz, csa_feature4, csa_position4, fps_idx4 = self.sa4(l3_xyz, csa_feature3, csa_position3)  #8

        msa0 = self.msa0(torch.cat([up_position,up_featrue],dim=1))

        f11 = index_points(up_featrue.permute(0,2,1),fps_idx1)
        p11 = index_points(up_position.permute(0,2,1), fps_idx1) #m=512 f=128
        fp1 = torch.cat([f11,p11],dim=-1)
        msa1 = self.msa1(torch.cat([fp1.permute(0,2,1),csa_position1,csa_feature1],dim=1)) # 64+128 --->128

        fp2 = index_points(msa1.permute(0,2,1),fps_idx2) # n =128 f=256
        msa2 = self.msa2(torch.cat([fp2.permute(0,2,1),csa_position2,csa_feature2],dim=1))

        fp3 = index_points(msa2.permute(0,2,1),fps_idx3) # n=32 f=512
        msa3 = self.msa3(torch.cat([fp3.permute(0,2,1), csa_position3 , csa_feature3], dim=1))

        x = self.cat(torch.cat([csa_feature4, csa_position4], dim=1))  # 1024

        x = self.up_sample1(l3_xyz, l4_xyz, msa3, x)  # l1_xyz, l2_xyz, l1_points, l2_points
        x = self.up_sample2(l2_xyz, l3_xyz, msa2, x)  # l1_xyz, l2_xyz, l1_points, l2_points
        x = self.up_sample3(l1_xyz, l2_xyz, msa1, x)  # l1_xyz, l2_xyz, l1_points, l2_points
        x = self.up_sample4(xyz, l1_xyz, msa0, x)  # l1_xyz, l2_xyz, l1_points, l2_points

        x = self.last(x)  # B C N
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x,None


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, tans):
        total_loss = F.nll_loss(pred, target)

        return total_loss


"s3dis时候使用"
# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#     def forward(self, pred, target, trans_feat, weight):
#         total_loss = F.nll_loss(pred, target, weight=weight)
#
#         return total_loss

if __name__ == '__main__':
    inputs = torch.randn((2, 6, 2048))
    o = get_model(50).eval()
    # print(o)
    print(o(inputs).size())
    list_ = [0.3, 0.7, -1, 2, 1.2]
    tensor_list = torch.as_tensor(list_)
    softmax = nn.Softmax(dim=-1)
    print(softmax(tensor_list))