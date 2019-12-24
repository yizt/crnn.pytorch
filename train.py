# -*- coding: utf-8 -*-
"""
 @File    : train.py
 @Time    : 2019/12/4 下午7:47
 @Author  : yizuotian
 @Description    :
"""

import argparse
import sys
import os
import numpy as np
import torch
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import crnn
from generator import Generator
from config import cfg


# import torchvision.transforms as transforms

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    args.distributed = False
    if args.device == 'cuda' and torch.cuda.device_count() > 1:
        args.distributed = True
        args.world_size = torch.cuda.device_count()
        args.rank = os.environ['RANK']
        torch.cuda.set_device(args.local_rank)

        print("args.local_rank:{},RANK:{},WORLD_SIZE:{}".format(args.local_rank, os.environ["RANK"],
                                                                os.environ['WORLD_SIZE']))
        # args.local_rank, os.environ["RANK"],os.environ['WORLD_SIZE'] 会自动赋值
        print('| distributed init (rank {}): {}'.format(
            args.rank, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)
        setup_for_distributed(args.rank == 0)


def train(args):
    device = torch.device(
        'cuda:{}'.format(args.local_rank) if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    init_distributed_mode(args)
    # data loader
    data_set = Generator(cfg.word.get_all_words())
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(data_set)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.workers)
    # model
    model = crnn.CRNN(len(data_set.alpha))
    criterion = CTCLoss()
    criterion = criterion.to(device)
    # 加载与训练模型
    if args.init_epoch > 0:
        model.load_state_dict(torch.load('crnn.{:03d}.pth'.format(args.init_epoch)))
    model = model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module
    # train
    model.train()

    for epoch in range(args.init_epoch, args.epochs):
        epoch_loss = 0.0
        if args.distributed:
            train_sampler.set_epoch(epoch)
        for image, target, input_len, target_len in tqdm(data_loader):
            image = image.to(device)
            # print(target, target_len, input_len)
            outputs = model(image.to(torch.float32))  # [B,N,C]
            outputs = torch.log_softmax(outputs, dim=2)
            outputs = outputs.permute([1, 0, 2])  # [N,B,C]
            loss = criterion(outputs[2:], target, input_len, target_len)
            # 梯度更新
            model.zero_grad()
            loss.backward()
            optimizer.step()
            # 当前轮的loss
            epoch_loss += loss.item() * image.size(0)
            if np.isnan(loss.item()):
                print(target, input_len, target_len)
        # 打印日志,保存权重
        print('Epoch: {}/{} loss: {:03f}'.format(epoch + 1, args.epochs, epoch_loss / len(data_set)))
        torch.save(model_without_ddp.state_dict(), 'crnn.{:03d}.pth'.format(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu', help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=40, help="epochs")
    parser.add_argument("--init-epoch", type=int, default=0, help="init epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--workers", type=int, default=4, help="number of workers")

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    arguments = parser.parse_args(sys.argv[1:])
    train(arguments)
