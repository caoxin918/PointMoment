import argparse
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.distributed as dist

from model.pointnet import PointNet


class PointHome(nn.Module):
    """
    feature encoder: PointNet
    projector: 3 layer mlp
    loss :loss(home) + loss(reg)
    """
    def __init__(self, args):
        super(PointHome, self).__init__()
        self.args = args
        # feature encoder
        self.backbone = PointNet()
        # projector
        sizes = [1024] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def loss_inv(self, z1, z2):
        c = (z1 * z2).sum(dim=0)
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        sim = c.add_(-1).pow_(2).mean()
        loss = self.args.lambd1 * sim

        return loss

    def loss2(self, z):
        c = z.T @ z

        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        off_diag = off_diagonal(c).pow_(2).mean()
        loss = self.args.lambd2 * off_diag

        return loss

    def loss3(self, z):
        c = torch.einsum('na,nb,nc->abc', [z, z, z])

        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        off_diag = off_diagonal3(c).pow_(2).mean()
        loss = self.args.lambd3 * off_diag

        return loss

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        z1 = self.bn(z1)
        z2 = self.bn(z2)

        if torch.randn(1).item() > 0.5:
            z = z1
        else:
            z = z2

        loss = self.loss_inv(z1, z2) + self.loss2(z)
        if self.args.lambd3 > 0:
            loss = loss + self.loss3(z)

        return loss

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
     n, m = x.shape
     assert n == m
     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def off_diagonal3(x):
    n, m, l = x.shape
    assert n == m
    assert n == l
    return x.flatten()[:-1].view(n - 1, n * n + n + 1)[:, 1:].flatten()


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)


    # def forward(self, y1, y2):
    #     z1 = self.bn(self.projector(self.backbone(y1)))
    #     z2 = self.bn(self.projector(self.backbone(y2)))
    #
    #     # 协方差矩阵
    #     c = z1.T @ z2
    #     # 自协方差矩阵
    #     c1 = z1.T @ z2
    #     c2 = z2.T @ z2
    #     # 三阶混合矩
    #     m1 = torch.einsum('ij,ik,im->jkm', [z1, z1, z1])
    #     m2 = torch.einsum('ij,ik,im->jkm', [z2, z2, z2])
    #
    #     # 协方差矩阵对角线元素
    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     on_diag = on_diag / self.args.D
    #
    #     # 两种变换后的图像分别计算自协方差矩阵后取非对角线元素
    #     off_diag_c1, off_diag_c2 = off_diagonal_square(c1, c2)
    #     off_diag_c1 = (off_diag_c1 / self.args.batch_size).pow_(2).sum()
    #     off_diag_c2 = (off_diag_c2 / self.args.batch_size).pow_(2).sum()
    #     # 两种变换后的图像分别计算自三阶混合矩后取非对角线元素
    #     off_diag_m1, off_diag_m2 = off_diagonal_cube(m1, m2)
    #     off_diag_m1 = (off_diag_m1 / self.args.batch_size).pow_(2).sum()
    #     off_diag_m2 = (off_diag_m2 / self.args.batch_size).pow_(2).sum()
    #
    #     off_diag = ((off_diag_c1 + off_diag_m1) / self.args.M + (off_diag_c2 + off_diag_m2) / self.args.M) / self.args.T
    #
    #     loss = on_diag + self.args.lambd * off_diag
    #     return loss


    # def forward(self, y1, y2):
    #     z_a = self.projector(self.backbone(y1))
    #     z_b = self.projector(self.backbone(y2))
    #
    #     z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
    #     z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
    #
    #     N = z_a.size(0)
    #     D = z_a.size(1)
    #
    #     # cross-correlation matrix
    #     c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
    #     # loss
    #     c_diff = (c - torch.eye(D, device='cuda')).pow(2)  # DxD
    #     # multiply off-diagonal elems of c_diff by lambda
    #     c_diff[~torch.eye(D, dtype=bool)] *= self.args.lambd
    #     loss = c_diff.sum()
    #     return loss

    # def forward(self, y1, y2, y3):
    #     z_a = self.projector(self.backbone(y1))
    #     z_b = self.projector(self.backbone(y2))
    #     z_c = self.projector(self.backbone(y3))
    #
    #     z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
    #     z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
    #     z_c_norm = (z_c - z_c.mean(0)) / z_c.std(0)  # NXD
    #
    #     N = z_a.size(0)
    #     D = z_a.size(1)
    #
    #     c1 = torch.mm(z_a_norm.T, z_b_norm) / N
    #     c2 = torch.mm(z_a_norm.T, z_c_norm) / N
    #     c3 = torch.mm(z_b_norm.T, z_c_norm) / N
    #
    #     c_diff_1 = (c1 - torch.eye(D, device='cuda')).pow(2)
    #     c_diff_2 = (c2 - torch.eye(D, device='cuda')).pow(2)
    #     c_diff_3 = (c3 - torch.eye(D, device='cuda')).pow(2)
    #
    #     c_diff_1[~torch.eye(D, dtype=bool)] *= self.args.lambd
    #     c_diff_2[~torch.eye(D, dtype=bool)] *= self.args.lambd
    #     c_diff_3[~torch.eye(D, dtype=bool)] *= self.args.lambd
    #
    #     loss = c_diff_1.sum() + c_diff_2.sum() + c_diff_3.sum()
    #     return loss


# def off_diagonal(x):
#     # return a flattened view of the off-diagonal elements of a square matrix
#     n, m = x.shape
#     assert n == m
#     # 首先利用flatten()拉直向量，然后去掉最后一个元素，
#     # 然后构造为一个维度为[N-1, N+1]的矩阵。在这个矩阵中，之前所有的对角线元素全部出现在第1列
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
#
#
# def off_diagonal_square(x1, x2):
#     # return a flattened view of the off-diagonal elements of a square matrix and cube matrix
#     n, m = x1.shape
#     i, j = x2.shape
#     assert n == m
#     assert i == j
#     off_diag_x1 = x1.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
#     off_diag_x2 = x2.flatten()[:-1].view(i-1, i+1)[:, 1:].flatten()
#     return off_diag_x1, off_diag_x2
#
#
# def off_diagonal_cube(x1, x2):
#     # return a flattened view of the off-diagonal elements of a cube matrix
#     n, m, q = x1.shape
#     assert n == m == q
#     i, j, k = x2.shape
#     assert i == j == k
#     off_diag_x1 = x1.flatten()[:-1].view(n-1, -1)[:, 1:].flatten()
#     off_diag_x2 = x2.flatten()[:-1].view(i-1, -1)[:, 1:].flatten()
#     return off_diag_x1, off_diag_x2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test point_home')
    parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--D', default=256, type=int,
                        help='the embedding feature dimension')
    parser.add_argument('--T', default=2, type=int,
                        help='the number of transform')
    parser.add_argument('--M', default=535822848, type=int,
                        help='the total number of combinations for all orders of moments')
    parser.add_argument('--projector', default='512-512-256', type=str, metavar='MLP',
                        help='projector MLP')
    parser.add_argument('--lambd', default=1, type=float, metavar='L',
                        help='weight on mixed moments')
    args = parser.parse_args()
    data1 = Variable(torch.rand(32, 3, 2500))
    data2 = Variable(torch.rand(32, 3, 2500))
    model = PointHome(args)
    print(model)
    loss = model.forward(data1, data2)
    print(loss)

