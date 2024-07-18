from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import wandb
from sklearn.svm import SVC

import torch
from torch import nn
import torch.optim as opt
from torch.utils.data import DataLoader

import builtins
import torch.distributed as dist
import torch.multiprocessing as mp

# from model.point_home import PointHome
from data.modelnet40 import ModelNet40
from data.dataset import ShapeNet, ModelNet40SVM
from model.pointnet import PointNet
from model.dgcnn import DGCNN

"""
time:05/23
version:v1
author:hanxinxin
description: +the module of selecting best acc on crosspoint
             +shapenet(pretraining) and modelnet40(test)
"""

def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a pointnet model with HOME", add_help=False)

    # Running
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--dist-url', default='tcp://localhost:10001',
                        help='url used to set up distributed training')

    # Data
    # parser.add_argument("--data-dir", type=Path, default="./datasets/cifar10",
    #                     help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="../exp_pointmoment_shapenet/dgcnn/mlp_4096_512_batchsize_8_epoch_200_augument_strl(strl_utils)_dgcnn_opt_adam_lr_0.001_CosineAnnealingLR_v0",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=100,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Optim
    parser.add_argument("--epochs", type=int, default=200,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=8,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.001,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Model
    parser.add_argument('--lambd1', default=10, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd2', default=10, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--lambd3', default=10, type=float, metavar='L',
                        help='weight on off-diagonal terms')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--projector', default='4096-512', type=str, metavar='MLP',
                        help='projector MLP')

    return parser


def main(args):
    # args.distributed = True

    ngpus_per_node = torch.cuda.device_count()

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    # args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    # mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    wandb.init(project="PointMoment_ShapeNet", name="DGCNN_1")

    args.gpu = gpu

    if "SLURM_NODEID" in os.environ:
        args.rank = int(os.environ["SLURM_NODEID"])

    # suppress printing if not first GPU on each node
    if args.gpu != 0 or args.rank != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    if "MASTER_PORT" in os.environ:
        args.dist_url = 'tcp://{}:{}'.format(args.dist_url, int(os.environ["MASTER_PORT"]))
    print(args.dist_url)

    print(args.rank, args.gpu)
    # args.rank = args.rank * ngpus_per_node + gpu
    # dist.init_process_group(backend='nccl', init_method=args.dist_url,
    #                         world_size=args.world_size, rank=args.rank)
    # torch.distributed.barrier()

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.exp_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    dataset = ShapeNet()
    # dataset = ModelNet40(args.num_points, partition='train')
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    per_device_batch_size = int(args.batch_size / args.world_size)
    print(args.batch_size, args.world_size, per_device_batch_size)
    loader = DataLoader(dataset, batch_size=per_device_batch_size, num_workers=args.num_workers,
                        pin_memory=True,shuffle=True, drop_last=True)
    # loader = DataLoader(dataset, batch_size=per_device_batch_size, num_workers=args.num_workers,
    #                     pin_memory=True, sample=sample)

    model = PointHome(args).cuda()
    print(model)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    wandb.watch(model)

    optimizer = opt.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)  # before:wd=1e-6  now:wd=1e-4(crosspoint)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, last_epoch=-1, verbose=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,
    #                             momentum=0.9, weight_decay=5e-4)

    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        msg = model.load_state_dict(ckpt["model"])
        print(msg)
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    best_acc = 0
    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        # #######################################
        # Train
        #########################################
        train_losses = AverageMeter()

        model.train()
        wandb_log={}
        # sampler.set_epoch(epoch)
        for step, (y1, y2) in enumerate(loader, start=epoch * len(loader)):

            batch_size = y1.size()[0]

            y1 = y1.transpose(2, 1)
            y2 = y2.transpose(2, 1)
            # y1 = y1.cuda(gpu, non_blocking=True)
            # y2 = y2.cuda(gpu, non_blocking=True)
            y1 = y1.cuda()
            y2 = y2.cuda()

            lr = adjust_learning_rate(args, optimizer, loader, step)
            # lr_scheduler.step()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2, False)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            train_losses.update(loss.item(), batch_size)

            current_time = time.time()
            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time

        wandb_log['Train Loss'] = train_losses.avg

        # Testing
        train_val_loader = DataLoader(ModelNet40SVM(partition='train', num_points=1024), batch_size=32, shuffle=True)
        test_val_loader = DataLoader(ModelNet40SVM(partition='test', num_points=1024), batch_size=32, shuffle=True)

        feats_train = []
        labels_train = []
        model.eval()

        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).cuda()
            with torch.no_grad():
                feats = model(data, None, True)
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
            data = data.permute(0, 2, 1).cuda()
            with torch.no_grad():
                feats = model.forward(data, None, True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)

        model_tl = SVC(C=0.1, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        wandb_log['Linear Accuracy'] = test_accuracy
        print(f"Linear Accuracy : {test_accuracy}")

        if args.rank ==0 and test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> epoch{:03d},Saving Best Model...'.format(epoch))

            torch.save(model.backbone.state_dict(), args.exp_dir / "model_best.pth")

        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )
            torch.save(state, args.exp_dir / "model_{:03d}.pth".format(epoch))

        wandb.log(wandb_log)

    if args.rank == 0:
        print('==> Saving Last Model...')
        torch.save(model.backbone.state_dict(), args.exp_dir / "model_final.pth")

    wandb.finish()


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


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
        # self.backbone = PointNet()
        self.backbone = DGCNN()
        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))  # 2048-DGCNN 1024-pointnet
        # sizes = [1024] + list(map(int, args.projector.split('-')))
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
        # torch.distributed.all_reduce(c)

        sim = c.add_(-1).pow_(2).mean()
        loss = self.args.lambd1 * sim

        return loss

    def loss2(self, z):
        c = z.T @ z

        c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        off_diag = off_diagonal(c).pow_(2).mean()
        loss = self.args.lambd2 * off_diag

        return loss

    def loss3(self, z):
        c = torch.einsum('na,nb,nc->abc', [z, z, z])

        c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        off_diag = off_diagonal3(c).pow_(2).mean()
        loss = self.args.lambd3 * off_diag

        return loss

    def forward(self, x1, x2, eval):
        z1 = self.backbone(x1)
        feat = z1
        if eval:
            return feat
        else:
            z1 = self.projector(z1)
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
    return x.flatten()[:-1].view(n - 1, n*n + n + 1)[:, 1:].flatten()


def off_diagonal_idx(dim):
    idx1, idx2 = torch.meshgrid(torch.arange(dim), torch.arange(dim))
    idx_select = idx1.flatten() != idx2.flatten()
    idx1_select = idx1.flatten()[idx_select]
    idx2_select = idx2.flatten()[idx_select]
    return [idx1_select, idx2_select]


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
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HOME training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
