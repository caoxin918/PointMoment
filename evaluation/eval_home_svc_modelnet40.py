from __future__ import print_function
import os
import random
import argparse
from pathlib import Path

import torch
import math
import numpy as np
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
from data.dataset import ScanObjectNNSVM
from model.dgcnn import DGCNN
from model.pointnet import PointNet


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'dgcnn':
        point_model = DGCNN().to(device)
    else:
        point_model = PointNet().to(device)

    # 加载训练的模型参数
    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = point_model.load_state_dict(state_dict, strict=False)  # 缺失的关键字,多余的关键字
    print(missing_keys)
    print(unexpected_keys)

    # point_model.load_state_dict(torch.load(args.model_path))
    # print("Model Loaded !!")

    # Testing

    train_val_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=1024), batch_size=32, shuffle=True)
    test_val_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=1024), batch_size=32, shuffle=True)

    feats_train = []
    labels_train = []
    point_model.eval()

    for i, (data, label) in enumerate(train_val_loader):
        labels = list(map(lambda x: x[0], label.numpy().tolist()))
        data = data.permute(0, 2, 1).to(device)
        with torch.no_grad():
            feats = point_model(data)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_train.append(feat)
        labels_train += labels

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)

    feats_test = []
    labels_test = []

    for i, (data, label) in enumerate(test_val_loader):
        labels = list(map(lambda x: x[0], label.numpy().tolist()))
        data = data.permute(0, 2, 1).to(device)
        with torch.no_grad():
            feats = point_model(data)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_test.append(feat)
        labels_test += labels

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)

    model_tl = SVC(C=0.1, kernel='linear')
    model_tl.fit(feats_train, labels_train)
    test_accuracy = model_tl.score(feats_test, labels_test)
    print(f"Linear Accuracy : {test_accuracy}")


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')

    parser.add_argument('--pretrained', default='../Ablation_analysis/one/dgcnn/two_order/backbone/model_015.pth', type=Path, metavar='FILE',
                        help='path to pretrained model')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)


