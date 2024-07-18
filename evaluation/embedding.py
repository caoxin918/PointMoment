from __future__ import print_function
import os
import random
import argparse
from pathlib import Path

import torch
import math
import numpy as np
import wandb
from lightly.loss.ntx_ent_loss import NTXentLoss
import time
from sklearn.svm import SVC

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from torch.utils.data import DataLoader

from data.modelnet40 import ModelNet40SVM
from data.modelnet10 import Dataset
from data.dataset import ScanObjectNNSVM, ShapeNet
from model.dgcnn import DGCNN
from model.pointnet import PointNet

def train(args):

    point_model = PointNet().cuda()

    # 加载训练的模型参数
    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = point_model.load_state_dict(state_dict, strict=False)  # 缺失的关键字,多余的关键字
    print(missing_keys)
    print(unexpected_keys)

    # point_model.load_state_dict(torch.load(args.model_path))
    # print("Model Loaded !!")

    # Testing
    root = '../datasets'
    dataset_name = 'modelnet10'
    split = 'test'

    datasets = Dataset(root=root, dataset_name=dataset_name, num_points=2048, split=split)  # modelnet10
    test_val_loader = DataLoader(datasets, batch_size=128, shuffle=False)

    feats_train = []
    point_model.eval()

    for i, (data, label, _, _) in enumerate(test_val_loader):
        print('Epoch (%d), Batch(%d/%d)' % (1, i, len(test_val_loader)))
        data = data.permute(0, 2, 1).cuda()
        with torch.no_grad():
            feats = point_model(data)
            feats = nn.functional.normalize(feats, dim=1)
            feats_train.append(feats.cpu())

    feats_train = torch.cat(feats_train, dim=0)
    feats_train = feats_train.numpy()

    np.save("../results/three_order.npy", feats_train)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--pretrained', default='path/to/model.pth', type=Path, metavar='FILE',
                        help='path to pretrained model')
    args = parser.parse_args()

    train(args)