import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch import optim, nn

from model.pointnet import PointNet
from data.modelnet40_eval import ModelNet40

parser = argparse.ArgumentParser(description='Evaluate resnet18 features on CiFar10')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', default='./exp/home_modelnet40_1/model_final.pth', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./exp/home_modelnet40_1/eval_hxx', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)  # 输出到文件

    model = PointNet().cuda()
    linear = torch.nn.Linear(1024, 40).cuda()
    print(model)

    # 加载训练的模型参数
    state_dict = torch.load(args.pretrained, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)  # 缺失的关键字,多余的关键字
    print(missing_keys)
    print(unexpected_keys)

    if args.weights == 'freeze':  # 只训练最后的线性层
        model.requires_grad_(False)

    optimizer = optim.Adam(params=linear.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss().cuda()

    # 创建数据集及其loader
    train_dataset = ModelNet40(num_points=2048, partition='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, drop_last=True,
                        pin_memory=False, sampler=None)
    test_dataset = ModelNet40(num_points=2048, partition='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # 从checkpoint恢复
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        start_epoch = 0
        best_acc = argparse.Namespace(top1=0, top5=0)

    # 开始训练
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        train_top1 = AverageMeter('Acc@1')
        train_top5 = AverageMeter('Acc@5')
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
            linear.train()
        else:
            assert False

        for step, (pointcloud, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            pointcloud = pointcloud.transpose(2, 1)
            pointcloud = pointcloud.cuda()
            target = target.cuda()  # ([161])
            out = model(pointcloud)
            out = linear(out)  # (16, 40)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc1, acc5 = accuracy(out, target, topk=(1, 5))
            train_top1.update(acc1[0].item(), 16)
            train_top5.update(acc5[0].item(), 16)

            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step, loss=loss.item(),
                             time=int(time.time() - start_time), train_acc1=train_top1.avg, train_acc5=train_top5.avg)
                print(json.dumps(stats))  # 将python对象转换为json对象
                print(json.dumps(stats), file=stats_file)

        scheduler.step()
        state = dict(
                epoch=epoch + 1, model=linear.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        torch.save(state, args.checkpoint_dir / 'checkpoint_linear_{:03d}.pth'.format(epoch))

        # sanity check
        if args.weights == 'freeze':
            reference_state_dict = torch.load(args.pretrained, map_location='cpu')
            model_state_dict = model.state_dict()
            for k in reference_state_dict:
                assert torch.equal(model_state_dict[k].cpu(), reference_state_dict[k]), k

    state = dict(
        epoch=epoch + 1, model=linear.state_dict(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
    torch.save(state, args.checkpoint_dir / 'checkpoint_linear_final.pth')

    # evaluate
    model.eval()
    linear.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    with torch.no_grad():
        for step, (pointcloud, target) in enumerate(test_loader, start=epoch * len(test_loader)):
            pointcloud = pointcloud.transpose(2, 1)
            pointcloud = pointcloud.cuda()
            target = target.cuda()
            out = model(pointcloud)
            out = linear(out)

            acc1, acc5 = accuracy(out, target.cuda(), topk=(1, 5))
            top1.update(acc1[0].item(), 16)
            top5.update(acc5[0].item(), 16)

            stats = dict(acc1=top1.avg, acc5=top5.avg)
            print(json.dumps(stats))  # 将python对象格式化成json字符
            print(json.dumps(stats), file=stats_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()