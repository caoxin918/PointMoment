import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.emb_dims = 1024

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()

        return x

# class Tnet(nn.Module):
#     def __init__(self, k=3):
#         super(Tnet, self).__init__()
#         self.conv1 = nn.Conv1d(k, 64, 1)
#         self.conv2 = nn.Conv1d(64, 128, 1)
#         self.conv3 = nn.Conv1d(128, 1024, 1)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, k*k)
#
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.bn5 = nn.BatchNorm1d(256)
#
#         self.k = k
#
#     def forward(self, x):
#         batchsize = x.size()[0]
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#
#         x = F.relu(self.bn4(self.fc1(x)))
#         x = F.relu(self.bn5(self.fc2(x)))
#         x = self.fc3(x)
#
#         iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).\
#             repeat(batchsize, 1)
#         if x.is_cuda:
#             iden = iden.cuda(x.device, non_blocking=True)
#         x = x + iden
#         x = x.view(-1, self.k, self.k)
#         return x


# class PointNet(nn.Module):
    # """
    # 提取点云的全局特征
    # """
    # def __init__(self, global_feat=True, feature_transform=False):
    #     super(PointNet, self).__init__()
    #     self.global_feat = global_feat
    #     self.feature_transform = feature_transform
    #     # 旋转出一个有利于分类或分割的角度，处理输入点云
    #     # self.input_tnet = Tnet(k=3)
    #     # 对提取出的特征进行变换，得到一个有利于分类的特征角度
    #     if self.feature_transform:
    #         self.feat_tnet = Tnet(k=64)
    #
    #     self.conv1 = nn.Conv1d(3, 64, 1)
    #     self.conv2 = nn.Conv1d(64, 128, 1)
    #     self.conv3 = nn.Conv1d(128, 1024, 1)
    #     self.bn1 = nn.BatchNorm1d(64)
    #     self.bn2 = nn.BatchNorm1d(128)
    #     self.bn3 = nn.BatchNorm1d(1024)
    #
    # def forward(self, x):
    #     num_points = x.size()[2]
    #     # trans_input = self.input_tnet(x)
    #     # x = x.transpose(2, 1)
    #     # x = torch.bmm(x, trans_input)
    #     # x = x. transpose(2, 1)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #
    #     if self.feature_transform:
    #         trans_feat = self.feat_tnet(x)
    #         x = x.transpose(2, 1)
    #         x = torch.bmm(x, trans_feat)
    #         x = x.transpose(2, 1)
    #     else:
    #         trans_feat = None
    #
    #     pointfeat = x  # 用于与全局特征拼接(分割)
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = self.bn3(self.conv3(x))
    #     x = torch.max(x, 2, keepdim=True)[0]
    #     x = x.view(-1, 1024)
    #     if self.global_feat:
    #         # 分类
    #         return x
    #     else:
    #         # 分割
    #         x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
    #         return torch.cat([x, pointfeat], 1)


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32, 3, 2500))
    pointfeat = PointNet(global_feat=True)
    print(pointfeat)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

