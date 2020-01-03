# -*- coding: utf-8 -*-
"""
 @File    : utils.py
 @Time    : 2019-12-25 11:39
 @Author  : yizuotian
 @Description    :
"""
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def _add_weight_history(writer, net, epoch):
    for name, param in net.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


def add_weight_history_on_master(writer, net, epoch):
    if is_main_process():
        _add_weight_history(writer, net, epoch)


def add_scalar_on_master(writer, tag, scalar_value, global_step):
    if is_main_process():
        writer.add_scalar(tag, scalar_value, global_step)
