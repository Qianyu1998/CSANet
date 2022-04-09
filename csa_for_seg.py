import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"CSANet for  semantic segmentation"
class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(get_model, self).__init__()
        # in_channel = 6 if normal_channel else 3
        self.k = 16  # number of neighbors
        self.total_points = 2048 # input points
        self.activate_function = nn.LeakyReLU(0.3, inplace=True)
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
        # cat feature and position with FA attention
        self.cat = nn.Sequential(
            nn.Conv1d(2048,256, 1, bias=False), nn.BatchNorm1d(256), self.activate_function,Attention(256,self.activate_function))


        # upsample with distance-based interpolation (Borrowed and improved from PointNet++)
        self.up_sample1 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=512 * 2,
                                                     activate_function=self.activate_function,
                                                     up_npoint=self.total_points // 64)  # 64  - 16
        self.up_sample2 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=256 * 2,
                                                     activate_function=self.activate_function,
                                                     up_npoint=self.total_points // 16)  # 16 - 4
        self.up_sample3 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=128 * 2,
                                                     activate_function=self.activate_function,
                                                     up_npoint=self.total_points // 4)  # 4 -2
        self.up_sample4 = PointNetFeaturePropagation(in_channel=256, mlp=[256,256], last=64 * 2,
                                                     activate_function=self.activate_function,
                                                     up_npoint=self.total_points)


        # msa: Multi-Scale fusion with attention
        self.msa0 = nn.Sequential(nn.Conv1d(64+64,128*2,1,bias=False),nn.BatchNorm1d(256),self.activate_function,
                                 nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), self.activate_function,
                                  Attention(256,self.activate_function))

        self.msa1 = nn.Sequential(nn.Conv1d(64*2+128*2,192,1,bias=False),nn.BatchNorm1d(192),self.activate_function,
                                 nn.Conv1d(192, 256, 1, bias=False), nn.BatchNorm1d(256), self.activate_function,Attention(256,self.activate_function))

        self.msa2 = nn.Sequential(nn.Conv1d(256+256+256, 512 , 1, bias=False), nn.BatchNorm1d(512),self.activate_function,
                                nn.Conv1d(512 , 256, 1, bias=False), nn.BatchNorm1d(256),self.activate_function,Attention(256,self.activate_function))

        self.msa3 = nn.Sequential(nn.Conv1d(256 + 512+512, 1024, 1, bias=False), nn.BatchNorm1d(1024), self.activate_function,
                                  nn.Conv1d(1024, 256, 1, bias=False), nn.BatchNorm1d(256),self.activate_function,Attention(256,self.activate_function) )

        self.last = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128),self.activate_function,
                                  nn.Dropout(0.2), nn.Conv1d(128, num_class, 1,bias=False ))

    def forward(self, xyz, trans=None):
        B, C, N = xyz.shape  ############################################# 8,3,1024  # BCN
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]

        # Backbone Network
        up_feature = self.first_f(norm)
        up_position = self.first_p(xyz)
        l1_xyz, csa_feature1, csa_position1, fps_idx1 = self.sa1(xyz, up_feature, up_position)  # N = 512  f1=128
        l2_xyz, csa_feature2, csa_position2, fps_idx2 = self.sa2(l1_xyz, csa_feature1, csa_position1)  # 128  # f2=256
        l3_xyz, csa_feature3, csa_position3, fps_idx3 = self.sa3(l2_xyz, csa_feature2, csa_position2)  # 32  # f3=512
        l4_xyz, csa_feature4, csa_position4, fps_idx4 = self.sa4(l3_xyz, csa_feature3, csa_position3)  #8

        # Multi-Scale fusion
        msa0 = self.msa0(torch.cat([up_position,up_feature],dim=1))
        # down sample
        f11 = index_points(up_feature.permute(0,2,1),fps_idx1)
        p11 = index_points(up_position.permute(0,2,1), fps_idx1) #m=512 f=128
        fp1 = torch.cat([f11,p11],dim=-1)
        msa1 = self.msa1(torch.cat([fp1.permute(0,2,1),csa_position1,csa_feature1],dim=1)) # 64+128 --->128

        fp2 = index_points(msa1.permute(0,2,1),fps_idx2) # n =128 f=256
        msa2 = self.msa2(torch.cat([fp2.permute(0,2,1),csa_position2,csa_feature2],dim=1))

        fp3 = index_points(msa2.permute(0,2,1),fps_idx3) # n=32 f=512
        msa3 = self.msa3(torch.cat([fp3.permute(0,2,1), csa_position3 , csa_feature3], dim=1))

        x = self.cat(torch.cat([csa_feature4, csa_position4], dim=1))  # 1024
        # decoder
        x = self.up_sample1(l3_xyz, l4_xyz, msa3, x)  # l1_xyz, l2_xyz, l1_points, l2_points
        x = self.up_sample2(l2_xyz, l3_xyz, msa2, x)  # l1_xyz, l2_xyz, l1_points, l2_points
        x = self.up_sample3(l1_xyz, l2_xyz, msa1, x)  # l1_xyz, l2_xyz, l1_points, l2_points
        x = self.up_sample4(xyz, l1_xyz, msa0, x)  # l1_xyz, l2_xyz, l1_points, l2_points

        x = self.last(x)  # B C N
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, tans):
        total_loss = F.nll_loss(pred, target)

        return total_loss

#cross self-attention layer
class CSA_Layer(nn.Module):
    def __init__(self, channels, activate_function):
        super(CSA_Layer, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)

        #self.trans_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = activate_function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature, position):

        x_q = self.q_conv(position).permute(0, 2, 1)[:, :, :, None]  # b, n, c,1
        x_k = self.k_conv(position).permute(0, 2, 1)[:, :, None, :]  # b, n, 1,c
        x_v = self.v_conv(feature)
        energy = torch.matmul(x_q, x_k)  # b,n c c
        energy = torch.sum(energy, dim=-2, keepdim=False)  # b n c
        energy = energy / (1e-9 + energy.sum(dim=-1, keepdim=True))
        attention = self.softmax(energy.permute(0, 2, 1)) # bcn
        x_r = torch.mul(attention, x_v)  # b, c, n
        x = (x_r + feature)
        # x = self.act(self.after_norm(self.trans_conv(feature + x_r)))

        return x


class CSANetSetAbstraction(nn.Module):
    #
    def __init__(self, npoint, radius, nsample, in_channel, mlp, knn=True, activate_function=None):
        super(CSANetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn  # whether to use knn to find neighbors
        self.activate_function = activate_function

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
        self.fa1 = Attention(mlp[1], self.activate_function)
        self.fa2 = Attention(mlp[1], self.activate_function)
    def forward(self, xyz, csa_f, csa_p=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            csa_f: input points data, [B, D, N] features
            csa_p: input points data, [B, D, N] position
        Return:
            new_xyz: sampled points position data, [B, C, S]
            csa_feature: after csa_layer feature
            csa_position: after csa_layer position
            fps_idx: down-sample idx
        """
        xyz = xyz.permute(0, 2, 1)
        # B C N  ---> B N C

        points = csa_f.permute(0, 2, 1)
        if csa_p is not None:
            csa_p = csa_p.permute(0, 2, 1)

        new_xyz, cat_xyz, cat_points, grouped_xyz_norm,fps_idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz,
                                                                          points, self.knn, csa_p)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]

        new_points = cat_points.permute(0, 3, 2, 1)  # [B, C+3, nsample,npoint]
        new_xyz_cat = cat_xyz.permute(0, 3, 2, 1)

        feature = self.feature(new_points)

        position = self.position(new_xyz_cat)

        position = torch.max(position, dim=2)[0]
        # FA Attention
        position = self.fa1(position)

        feature = torch.max(feature, dim=2)[0]
        #FA Attention
        feature = self.fa2(feature)

        csa_feature = self.csa_feature(feature, position)
        csa_position = self.csa_position(position, feature)

        return new_xyz.permute(0, 2, 1), csa_feature, csa_position,fps_idx


class PointNetFeaturePropagation(nn.Module):
    # upsample with distance-based interpolation (Borrowed and improved from PointNet++)
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

        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]

        Return:
            new_points: upsampled points data, [B, D', N]
        """

        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        _, _, C2 = points2.shape

        B, N, C = xyz1.shape  # N = 128
        _, S, _ = xyz2.shape  # s = 1

        if S == 1:

            interpolated_points = points2.repeat(1, N, 1)
        else:

            dists = square_distance(xyz1, xyz2)  # 2,512,128
            dists, idx = dists.sort(dim=-1)  # 2,512,128
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3] 2,512,3
            dist_recip = 1.0 / (dists + 1e-8)  #
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # 2,512,3
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = (interpolated_points.permute(0, 2, 1) +  points1)
        else:
            new_points = interpolated_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.activate_function(bn(conv(new_points)))

        return new_points

# FA attention
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
        y = x
        x_t = x.permute(0,2,1) # b n c
        att = torch.bmm(x,x_t) # b c c
        att = torch.sum(att,dim=-1,keepdim=True)
        att = self.trans(att)
        return y*att



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

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
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat(
        [B, S, 1])
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


def sample_and_group(npoint, radius, nsample, xyz, points, knn, csa_p):
    B, N, C = xyz.shape
    Bf, Nf, Cf = points.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz.contiguous(), npoint)  # [B, npoint, C] # 
    new_xyz = index_points(xyz, fps_idx)

    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K     2,256,3
    else:
        idx = query_ball_point(radius, nsample, xyz.contiguous(), new_xyz.contiguous())

    grouped_points = index_points(points, idx)
    grouped_xyz = index_points(xyz, idx)

    # original coordinate offset
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  #

    if csa_p is not None:
        # position cat original coordinate offset
        csa_position = index_points(csa_p, idx)
        csa_position = torch.cat([csa_position, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
                                 dim=-1)
    else:

        csa_position = grouped_xyz_norm
    # feature cat original coordinate offset
    csa_feature = torch.cat(
        [grouped_points, grouped_xyz_norm, new_xyz.view(B, S, 1, C).expand_as(grouped_xyz_norm)],
        dim=-1)  # B S neiber C+1

    return_all = False
    if return_all:
        return new_xyz, grouped_xyz, grouped_xyz_norm, grouped_points
    else:
        return new_xyz, csa_position, csa_feature, grouped_xyz_norm,fps_idx



if __name__ == '__main__':
    inputs = torch.randn((2, 6, 2048))
    o = get_model(50).eval()
    # print(o)
    print(o(inputs).size())
    list_ = [0.3, 0.7, -1, 2, 1.2]
    tensor_list = torch.as_tensor(list_)
    softmax = nn.Softmax(dim=-1)
    print(softmax(tensor_list))
