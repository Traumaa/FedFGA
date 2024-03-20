import torch

import os
import json
import random
import numpy as np
from collections import Counter
from spikingjelly.activation_based import functional, monitor, neuron


class AverageMeter(object):
    """Record metrics information"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count



def test_firerate(model, img):
    # model.eval()
    firerate_ = AverageMeter()
    # SNN
    functional.reset_net(model)

    # 监视器
    # 监视点火率函数
    def cal_firing_rate(s_seq: torch.Tensor):
        if len(s_seq.shape) == 5 and s_seq.shape[2] == 128:
            # 选择所有组中每一列的元素
            all_groups_all_columns = s_seq[:, :, :, :, :]
            # 初始化一个空的列表，用于存储每一列的平均值
            average_per_column_list = []

            # 遍历每一列
            for col_index in range(all_groups_all_columns.size(2)):
                # 选择所有组中每一列的元素
                column_elements = all_groups_all_columns[:, :, col_index, :, :]

                # 求元素和
                sum_column_elements = column_elements.sum()

                # 获取元素个数
                num_elements = column_elements.numel()

                # 求平均值
                average_column = sum_column_elements / num_elements

                # 将平均值添加到列表中
                average_per_column_list.append(average_column)

            # average_per_column_list = average_per_column_list[-128:] # 取倒数第二层的128通道的点火率
            # 返回每一列的平均值列表
            return average_per_column_list

    # def cal_firing_rate(s_seq: torch.Tensor):
    #     while len(s_seq.shape) == 5 and s_seq.shape[2] == 32:
    #         s_seq_1 = s_seq.mean(0)
    #         s_seq_2 = s_seq_1.mean(0)
    #         s_seq_3 = s_seq_2.mean(-1)
    #         average_per_column_list = s_seq_3.mean(-1)
    #
    #         return average_per_column_list.tolist()




    # def cal_firing_rate(s_seq: torch.Tensor):
    #     # print(s_seq)
    #     print(s_seq.shape)
    #     return s_seq.flatten(1).mean(1)
        # 设置监视器
    # fr_monitor = monitor.OutputMonitor(model, neuron.IFNode)
    fr_monitor = monitor.OutputMonitor(model, neuron.IFNode, cal_firing_rate)

    functional.reset_net(model)
    fr_monitor.enable()
    model(img)

    # record = fr_monitor.records
    # record1 = record
    # firerate = torch.sum(fr_monitor.records[-1])
    firerate = fr_monitor.records
    # --> fr_monitor.records shape:[torch.Size([10, 500, 64, 28, 28]), torch.Size([10, 500, 32, 28, 28]), torch.Size([10, 500, 32, 14, 14]), torch.Size([10, 500, 10])]，然后将每个tensor展平，求平均得到每个时间步的点火率。依次按照32通道递减，如果将32个通道看作一组，如何输出不同组的点火率？
    firerate_np = np.array([t.cpu().item() for sublist in firerate if sublist is not None for t in sublist])  # 去除cuda

    functional.reset_net(model)
    # del fr_monitor
    fr_monitor.remove_hooks()

    # return firerate_.sum
    return firerate_np


    # # 监视器
    # # 监视点火率函数
    # def cal_firing_rate(s_seq: torch.Tensor):
    #     # s_seq.shape = [T, N, *]
    #     return s_seq.flatten(1).mean(1)
    #
    # # 设置监视器
    # fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, cal_firing_rate)
    # # 监视
    # with torch.no_grad():
    #     functional.reset_net(net)   # 重置网络
    #     fr_monitor.disable()        # 暂停记录
    #     net(images)                 # 运行一次网络
    #     functional.reset_net(net)   # 重置网络
    #     # print(f'after call fr_monitor.disable(), fr_monitor.records=\n{fr_monitor.records}')
    #     fr_monitor.enable()         # 进行记录
    #     a = net(images)             # 运行一次网络
    #     # print(a)
    #     print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
    #     functional.reset_net(net)   # 重置网络
    #     del fr_monitor              # 删除监视器，释放资源