
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"CSANet for Classification"
class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.k = 16 # neighbors
        self.total_points = 1024 #input points
        self.activate_function = nn.LeakyReLU(0.3,inplace=True)

        self.normal_channel = normal_channel

        self.first_f = nn.Sequential(nn.Conv1d(3, 32, 1, bias=False), nn.BatchNorm1d(32), self.activate_function
                                     , nn.Conv1d(32, 64, 1, bias=False), nn.BatchNorm1d(64), self.activate_function)

        self.first_p = nn.Sequential(nn.Conv1d(3, 32, 1, bias=False), nn.BatchNorm1d(32), self.activate_function,
                                     nn.Conv1d(32, 64, 1, bias=False), nn.BatchNorm1d(64), self.activate_function)

        self.sa1 = CSANetSetAbstraction(npoint=self.total_points // 4, radius=0.1, nsample=self.k,
                                          in_channel=64, mlp=[64, 128], activate_function=self.activate_function,
                                          knn=True)
        self.sa2 = CSANetSetAbstraction(npoint=self.total_points // 16, radius=0.2, nsample=self.k,
                                          in_channel=128, mlp=[128, 256], activate_function=self.activate_function,
                                          knn=True)
        self.sa3 = CSANetSetAbstraction(npoint=self.total_points // 64, radius=0.4, nsample=self.k,
                                          in_channel=256, mlp=[256, 512], activate_function=self.activate_function,
                                          knn=True)
        self.sa4 = CSANetSetAbstraction(npoint=self.total_points // 256, radius=0.6, nsample=self.k,
                                          in_channel=512, mlp=[512, 1024], activate_function=self.activate_function,
                                          knn=True)


        self.cat = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(2048,1024,1,bias=False),nn.BatchNorm1d(1024),self.activate_function
        )
        self.fc2 = nn.Linear(1024, 512,bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(512, 256,bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, num_class,bias=False)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]

        up_feature = self.first_f(norm)
        up_position = self.first_p(xyz)

        l1_xyz, csa_feature1, csa_position1 = self.sa1(xyz, up_feature, up_position)  # B C N 质心坐标和该层预测结果
        l2_xyz, csa_feature2,csa_position2 = self.sa2(l1_xyz, csa_feature1,csa_position1)
        l3_xyz, csa_feature3,csa_position3 = self.sa3(l2_xyz, csa_feature2,csa_position2)
        l4_xyz, csa_feature4, csa_position4 = self.sa4(l3_xyz, csa_feature3, csa_position3)
        #cat postion and feature
        cat_f_p = torch.cat([csa_feature4,csa_position4],dim=1)
        x = self.cat(cat_f_p)
        x = torch.max(x,dim=-1)[0]

        x = x.view(B, 1024)

        x = self.drop2(self.activate_function(self.bn2(self.fc2(x))))
        x = self.drop3(self.activate_function(self.bn3(self.fc3(x))))
        x = self.fc4(x) ##################################################### 8,10
        x = F.log_softmax(x, -1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

class CSA_Layer(nn.Module):
    def __init__(self, channels,activate_function):
        super(CSA_Layer,self).__init__()

        self.q_conv = nn.Conv1d(channels, channels , 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels , 1, bias=False)

        self.v_conv = nn.Conv1d(channels, channels , 1,bias=False)
        self.trans_conv = nn.Conv1d(channels, channels, 1,bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = activate_function
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature,position):
        """
        feature:   projected features
        position:  projected position
        """
        x_q = self.q_conv(position).permute(0, 2, 1)[:,:,:,None]  # b, n, c,1
        x_k = self.k_conv(position).permute(0, 2, 1)[:,:,None,:]  # b, n, 1,c
        x_v = self.v_conv(feature)
        energy = torch.matmul(x_q,x_k) # b, n, c c
        energy = torch.sum(energy,dim=2,keepdim=False) # b n c
        energy = energy / (1e-9 + energy.sum(dim=-1, keepdim=True))
        attention = self.softmax(energy).permute(0,2,1)

        x_r = torch.mul(attention,x_v)  # b, c, n

        x = feature + x_r

        return x

class CSANetSetAbstraction(nn.Module):
    #
    def __init__(self, npoint, radius, nsample, in_channel, mlp,knn=True,activate_function=None):
        super(CSANetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius # Radius of the ball query
        self.nsample = nsample
        self.knn = knn # Whether to use knn to find neighbors
        self.activate_function = activate_function

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


    def forward(self, xyz, csa_f,csa_p=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            csa_f: input points data, [B, D, N] features
            csa_p: input points data, [B, D, N] position
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        # B C N  ---> B N C

        points = csa_f.permute(0, 2, 1)
        if csa_p is not None:
            csa_p = csa_p.permute(0,2,1)

        new_xyz, cat_xyz, cat_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.knn,csa_p)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = cat_points.permute(0, 3, 2, 1) # [B, C+3, nsample,npoint]
        new_xyz_cat = cat_xyz.permute(0,3, 2, 1)

        feature = self.feature(new_points)
        position = self.position(new_xyz_cat)

        position = torch.max(position, dim=2)[0]
        feature = torch.max(feature, dim=2)[0]

        csa_feature = self.csa_feature(feature,position)
        csa_position = self.csa_position(position,feature)

        return new_xyz.permute(0, 2, 1), csa_feature,csa_position

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
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
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
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
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :]
        centroid = centroid.view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def sample_and_group(npoint, radius, nsample, xyz,points, knn,csa_p):
    """
    self.npoint, self.radius, self.nsample, xyz,self.knn
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data,    B Np C
        grouped_xyz_normal:     B Np  Ns C
        new_points:    B Np C
        grouped_points_normal:  B Np  Ns C
    """
    B, N, C = xyz.shape
    Bf,Nf,Cf = points.shape
    S = npoint

    fps_idx = torch.as_tensor(np.random.choice(N, npoint, replace=True)).view(-1, npoint).repeat(B, 1)  # 1 (512,)
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points,fps_idx)
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K     2,256,3

    else:
        idx,dist= query_ball_point(radius, nsample, xyz.contiguous(), new_xyz.contiguous())

    grouped_points = index_points(points, idx)

    grouped_xyz = index_points(xyz, idx)

    #original coordinate offset
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_points_norm = grouped_points   #  - new_points.view(Bf, S, 1, Cf)  #

    if csa_p is not  None:
        csa_position = index_points(csa_p,idx)
        # position cat original coordinate offset
        csa_position = torch.cat([csa_position, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
                  dim=-1)
    else:
        csa_position = grouped_xyz_norm
        #feature cat original coordinate offset
    csa_feature = torch.cat([grouped_points_norm, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
        dim=-1)  # B S neiber C+1

    return_all = False
    if return_all :
        return new_xyz,grouped_xyz,grouped_xyz_norm,new_points,grouped_points_norm
    else:
        return new_xyz, csa_position,csa_feature






if __name__ == '__main__':
    inputs = torch.randn((2,6,1024))
    o = get_model(10)
    print(o)
    print(o(inputs).size())
    list_ = [0.3,0.7,-1,2,1.2]
    tensor_list = torch.as_tensor(list_)
    softmax = nn.Softmax(dim=-1)
    print(softmax(tensor_list))
