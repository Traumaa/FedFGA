from collections import defaultdict
from dataclasses import replace
from email.policy import strict
import math
import random

import utility
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as tu
from tqdm import tqdm
import wandb
import collections
import copy


class Trainer:
    def __init__(self, args, loader, agent_list, ckp):

        self.args = args
        self.ckp = ckp  # save need modified
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loaders_train = loader.loaders_train
        self.agent_list = agent_list[:-1]  # last one is the tester
        self.tester = agent_list[-1]
        self.epoch = 0
        for agent in self.agent_list:
            agent.make_optimizer_all(ckp=ckp)
            agent.make_scheduler_all()
            # self.top_channels_indices = agent.top_channels_indices
        self.tester.make_optimizer_all(ckp=ckp)
        self.tester.make_scheduler_all()
        # self.top_channels_indices = agent_list[-1].top_channels_indices

        self.run = wandb.init(project=args.project)
        self.run.name = args.save

        self.device = torch.device('cpu' if args.cpu else 'cuda')

        self.sync_at_init()

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def train(self):
        # epoch, _ = self.start_epoch()
        self.epoch += 1
        epoch = self.epoch

        for agent in self.agent_list:
            agent.make_optimizer_all()
            agent.make_scheduler_all(reschedule=epoch - 1)  # pls make sure if need resume
        # Step 1: sample a list of agent w/o replacement
        agent_joined = np.sort(np.random.choice(range(self.args.n_agents), self.args.n_joined, replace=False))
        # Step 2: sample a list of associated budgets
        while True:
            # agent_budget = np.full(10, 0.25) # 极端情况
            # break

            # 非极端情况
            agent_budget = np.random.choice(self.args.fraction_list, self.args.n_joined)
            # For implementation simiplicity, we sample all model size for all methods
            _, unique_counts = np.unique(agent_budget, return_counts=True)
            if len(unique_counts) == len(self.args.fraction_list):
                break
        # Step 3: create a buget -> client dictionary for syncing later
        budget_record = collections.defaultdict(list)  # need to make sure is not empty
        for k, v in zip(agent_budget, agent_joined):
            budget_record[k].append(v)

        for i in agent_joined:
            self.agent_list[i].begin_all(epoch, self.ckp)  # call move all to train()
            self.agent_list[i].start_loss_log()  # need check!

        self.start_epoch()  # get current lr
        timer_model = utility.timer()

        for idx, i in enumerate(agent_joined):
            timer_model.tic()

            loss, loss_orth, log_train = self.agent_list[i].train_local(self.loaders_train[i],
                                                                        agent_budget[idx], self.args.local_epochs)

            timer_model.hold()
            tt = timer_model.release()

            self.ckp.write_log(
                '{}/{} ({:.0f}%)\t'
                'agent {}\t'
                'model {}\t'
                'NLL: {:.3f}\t'
                'Top1: {:.2f} / Top5: {:.2f}\t'
                'Total {:<2.4f}/ Orth: {:<2.5f} '
                'Time: {:.1f}s'.format(
                    idx + 1,
                    len(agent_joined),
                    100.0 * (idx + 1) / len(agent_joined), i, agent_budget[idx],
                    # *(log_train),
                    *log_train,
                    loss, loss_orth,
                    tt
                )
            )

        for i in agent_joined:
            self.agent_list[i].log_all(self.ckp)
            for loss in self.agent_list[i].loss_list:
                loss.end_log(len(self.loader_train.dataset) * self.args.local_epochs)  # should be accurate



        self.sync(budget_record, agent_joined, agent_budget)
        self.budget_record = budget_record
        self.agent_joined = agent_joined
        self.agent_budget = agent_budget
        # 训练完成后，中心服务器和客户端中的网络参数进行同步（聚合并下发）
        # 这里的sync函数负责生成两个字典【将filter加权聚合，将其他参数加权聚合】，然后分别发送给服务器和客户端





    def test(self):
        # epoch = self.scheduler.last_epoch + 1
        epoch = self.epoch
        self.ckp.write_log('\nEvaluation:')
        timer_test = utility.timer()
        # 对每个网络进行测试，并根据点火率筛选出通道
        self.tester.test_all(self.loader_test, timer_test, self.run, epoch)
        self.top_channels_indices = self.tester.top_channels_indices  # 需要在完成test_all后用变量接受一下tester的top_channels_indices属性
        # 测试完成后根据点火率选出的通道索引，对各个规模的客户端进行参数赋值
        self.sync_firerate()

    def sync_at_init(self):
        if self.args.resume_from:
            for i in range(len(self.args.fraction_list)):
                print("resume from checkpoint")
                self.tester.model_list[i].load_state_dict(torch.load(
                    '../experiment/' + self.args.save + '/model/model_m' + str(
                        i) + '_' + self.args.resume_from + '.pt'))
        # Sync all agents' parameters with tester before training
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            state_dict = self.tester.model_list[model_id].state_dict()
            for agent in self.agent_list:
                agent.model_list[model_id].load_state_dict(copy.deepcopy(state_dict), strict=True)

    def sync(self, budget_record, agent_joined, agent_budget):
        # Step 1: gather all filter banks
        # This step runs across network fractions
        print("训练完成，开始聚合通道")
        filter_banks = {}
        for k, v in self.tester.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape)

        for k in filter_banks:
            for b, i in zip(agent_budget, agent_joined):
                model_id = self.agent_list[i].budget_to_model(b)
                state_dict = self.agent_list[i].model_list[model_id].state_dict()
                filter_banks[k] += state_dict[k] * (1. / self.args.n_joined)
        # Step 2: gather all other parameters
        # This step runs within each network fraction
        anchors = {}
        weights_dict = {}
        for net_f in self.args.fraction_list:
            n_models = len(budget_record[net_f])
            agent_list_at_net_f = budget_record[net_f]

            '''
            这里是对不同网络的参数进行平均，举例：参与聚合的客户端中有两个0.5的网络，则对他们参数进行平均，放入anchor字典
            修改，将不同网络的参数聚合到1.0网络里面去，实现通道聚合
            其中 conv2.conv.1.weight参数形状：
            0.25 网络参数形状：(32, 64, 1, 1)

            0.5 网络参数形状：(64, 128, 1, 1)
            
            0.75  网络参数形状：(96, 192, 1, 1)
            
            1.0 网络参数形状：(128, 256, 1, 1)
            '''
            anchor = {}
            model_id = self.tester.budget_to_model(net_f)
            for k, v in self.tester.model_list[model_id].state_dict().items():
                if 'filter_bank' not in k:
                    anchor[k] = torch.zeros(v.shape)


            for k in anchor:
                for i in agent_list_at_net_f:
                    model_id = self.agent_list[i].budget_to_model(net_f)
                    state_dict = self.agent_list[i].model_list[model_id].state_dict()
                    anchor[k] += state_dict[k] * (1. / n_models)

            # 将每个网络的conv2.conv.1.weight参数取出
            weights_dict[net_f] = anchor['conv2.conv.1.weight'].clone().detach().cpu().numpy()
            anchors[net_f] = anchor

        # 对每个网络的conv2.conv.1.weight参数进行合并
        merged_tensor = self.merge_tensors(weights_dict)
        # print(merged_tensor.shape())
        # 合并好的参数赋值给1.0网络的conv2.conv.1.weight参数
        anchors[1.0]['conv2.conv.1.weight'] = merged_tensor
        self.anchors = anchors

        # Step 3: distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)

        # Last step: update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)

        print("这一轮的通道参数聚合完成")
        return filter_banks, anchors

    def merge_tensors(self,weights_dict):
        # 现在1.0网络多余的参数相当于除以了4，重新优化聚合策略
        # 创建一个列表，用于存储插值或直接赋值后的张量
        # expanded_tensors = []
        #
        # target_shape = (128, 256, 1, 1)
        #
        # # 遍历字典中的键值对，对张量进行拓展操作
        # for key, tensor in weights_dict.items():
        #     # 创建目标形状的零张量
        #     expanded_tensor = np.zeros(target_shape)
        #
        #     # 计算需要复制的原始张量的形状范围
        #     shape_ranges = []
        #     for i in range(4):
        #         shape_ranges.append(min(target_shape[i], tensor.shape[i]))
        #
        #     # 将原始张量的值复制到新的张量中
        #     expanded_tensor[:shape_ranges[0], :shape_ranges[1], :shape_ranges[2], :shape_ranges[3]] = tensor
        #     # 将numpy数组转换为PyTorch张量
        #     expanded_tensor = torch.tensor(expanded_tensor)
        #     # 更新字典中的张量为扩展后的张量
        #     expanded_tensors.append(expanded_tensor)
        # print(key, "conv2.conv.1.weight原始张量形状：", tensor.shape)
        # print(key, "conv2.conv.1.weight扩展后张量形状：", expanded_tensor.shape)
        #
        #
        # # 将所有张量相加
        # merged_tensor = sum(expanded_tensors)
        # # 求平均值
        # merged_tensor /= len(expanded_tensors)
        # # print("merged_tensor形状：", expanded_tensor.shape)
        keys_shapes = {
            0.25: (32, 64, 1, 1),
            0.5: (64, 128, 1, 1),
            0.75: (96, 192, 1, 1),
            1.0: (128, 256, 1, 1)
        }

        # 创建一个空字典来存储新的张量
        merged_tensors = {}

        # Step 1: 处理 0.25 的张量
        sum_tensor_025 = torch.zeros(keys_shapes[0.25])
        for k in keys_shapes:
            tensor = weights_dict[k][:32, :64, :, :]  # 根据最小形状裁剪
            sum_tensor_025 += tensor
        merged_tensors[0.25] = sum_tensor_025 / 4

        # Step 2: 处理 0.5 的张量
        sum_tensor_05 = torch.zeros(keys_shapes[0.5])
        for k in [0.5, 0.75, 1.0]:
            tensor = weights_dict[k][:64, :128, :, :]  # 根据 0.5 的形状裁剪
            sum_tensor_05 += tensor
        merged_tensors[0.5] = sum_tensor_05 / 3

        # Step 3: 处理 0.75 的张量
        sum_tensor_075 = torch.zeros(keys_shapes[0.75])
        for k in [0.75, 1.0]:
            tensor = weights_dict[k][:96, :192, :, :]  # 根据 0.75 的形状裁剪
            sum_tensor_075 += tensor
        merged_tensors[0.75] = sum_tensor_075 / 2

        # Step 4: 处理 1.0 的张量
        # 直接使用1.0的张量，因为无需合并
        merged_tensors[1.0] = weights_dict[1.0]

        final_tensor = weights_dict[1.0]  #

        # 替换操作
        final_tensor[:96, :192, :, :] = merged_tensors[0.75]  # 替换来自 0.75 的区域
        final_tensor[:64, :128, :, :] = merged_tensors[0.5]  # 替换来自 0.5 的区域
        final_tensor[:32, :64, :, :] = merged_tensors[0.25] # 替换 0.25
        final_tensor = torch.tensor(final_tensor)
        print("final_tensor形状：", final_tensor.shape)
        return final_tensor


    def sync_firerate(self):

        '''
        '''
        '''
        功能：将1.0网络的conv2.conv.1.weight参数根据点火率通道索引进行拆分，分别赋值给不同客户端，使其在下一轮继续训练
        1. 初始化空字典anchors，
        2. 对每一个网络进行处理，初始化为与tester中的参数形状相同的全零张量
        3. 对每一个参数进行加权平均
        1.0 网络Tensor：(128，256，1，1)

        0.75 	Tensor：(96，192，1，1)

        0.5  	Tensor：(64，128，1，1)

        0.25 	Tensor：(32，64，1，1)

        第二个维度暂时按照顺序进行选取
        '''

        # 筛选出的通道
        top_channels_indices_25 = self.top_channels_indices[:32]
        top_channels_indices_50 = self.top_channels_indices[:64]
        top_channels_indices_75 = self.top_channels_indices[:96]


        # print(top_channels_indices)

        anchors = self.anchors

        '''
               目的是将1.0网络中conv2.conv.1.weight拆分成4个张量，分别对应0.25，0.5，0.75，1.0，赋值给不同规模网络
               得先把1.0 网络中的conv2.conv.1.weight取出来，然后根据点火率进行拆分操作（封装成函数）
        '''
        # 初始化一个空字典，用于存储网络分数为1.0的网络的模型参数
        anchor_1_0 = {}

        # 遍历测试器的模型参数字典，找到网络分数为1.0的模型
        for k, v in self.tester.model_list[self.tester.budget_to_model(1.0)].state_dict().items():
            if k == 'conv2.conv.1.weight':  # 如果当前键是'conv2.conv.1.weight'
                anchor_1_0[k] = v.clone().detach()  # 将参数值存储到anchor_1_0字典中
                break  # 找到后立即退出循环


        # 打印anchor_1_0字典中的参数形状
        self.anchor_1_0 = anchor_1_0
        print("取出来的conv2.conv.1.weight tensor in anchor_1_0 shape:", anchor_1_0['conv2.conv.1.weight'].shape)

        for net_f in self.args.fraction_list:  # 对不同分数的网络进行处理

            model_id = self.tester.budget_to_model(net_f)
            print("model_id:",model_id)

            if net_f == 0.25:  # 对0.25网络执行最后一层参数拆分复制操作
                for k in anchors[net_f]:
                    if k == 'conv2.conv.1.weight':
                        anchors[net_f][k] = self.split_tensor_by_indices_25(anchor_1_0, top_channels_indices_25)
                        print("Shape of 0.25 tensor:", anchors[net_f][k].shape)
            if net_f == 0.5:
                for k in anchors[net_f]:
                    if k == 'conv2.conv.1.weight':
                        # 对于'conv2.conv.1.weight'键，执行赋值0.50操作
                        anchors[net_f][k] = self.split_tensor_by_indices_50(anchor_1_0, top_channels_indices_50)
                        print("Shape of 0.50 tensor:", anchors[net_f][k].shape)
            if net_f == 0.75:
                for k in anchors[net_f]:
                    if k == 'conv2.conv.1.weight':
                        # 对于'conv2.conv.1.weight'键，执行赋值0.75操作
                        anchors[net_f][k] = self.split_tensor_by_indices_75(anchor_1_0, top_channels_indices_75)
                        print("Shape of 0.75 tensor:", anchors[net_f][k].shape)
            if net_f == 1:
                for k in anchors[net_f]:
                    if k == 'conv2.conv.1.weight':
                        # 对于'conv2.conv.1.weight'键，执行赋值1.0操作
                        anchors[net_f][k] = self.split_tensor_by_indices_10(anchor_1_0, top_channels_indices_75)
                        print("Shape of 1.00 tensor:", anchors[net_f][k].shape)

            # anchors[net_f] = anchor

        # Step 3: distribute anchors and filter banks to all agents
        for agent in self.agent_list:
            for net_f in self.args.fraction_list:
                model_id = agent.budget_to_model(net_f)
                agent.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
                # agent.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)

        # Last step: update tester
        for net_f in self.args.fraction_list:
            model_id = self.tester.budget_to_model(net_f)
            self.tester.model_list[model_id].load_state_dict(copy.deepcopy(anchors[net_f]), strict=False)
            print("按照通道点火率分发参数完成")
            # self.tester.model_list[model_id].load_state_dict(copy.deepcopy(filter_banks), strict=False)
        return anchors

    def split_tensor_by_indices_25(self, anchor_1_0, top_channels_indices):
        # 获取 anchor_1_0['conv2.conv.1.weight'] 的形状
        original_shape = anchor_1_0['conv2.conv.1.weight'].shape
        tensor = anchor_1_0['conv2.conv.1.weight']

        # 创建一个新的张量，形状为 [32, 64, 1, 1]
        new_shape = (32, 64, original_shape[2], original_shape[3])
        new_tensor = torch.zeros(new_shape)

        # 根据 top_channels_indices 进行拆分
        for i, idx in enumerate(top_channels_indices):
            new_tensor[i, :, :, :] = tensor[idx, :64, :, :]

        return new_tensor

    def split_tensor_by_indices_50(self, anchor_1_0, top_channels_indices):
        # 获取 anchor_1_0['conv2.conv.1.weight'] 的形状
        original_shape = anchor_1_0['conv2.conv.1.weight'].shape
        tensor = anchor_1_0['conv2.conv.1.weight']

        # 创建一个新的张量，形状为 [64, 128, 1, 1]
        new_shape = (64, 128, original_shape[2], original_shape[3])
        new_tensor = torch.zeros(new_shape)

        # 根据 top_channels_indices 进行拆分
        for i, idx in enumerate(top_channels_indices):
            new_tensor[i, :, :, :] = tensor[idx, :128, :, :]

        return new_tensor

    def split_tensor_by_indices_75(self, anchor_1_0, top_channels_indices):
        # 获取 anchor_1_0['conv2.conv.1.weight'] 的形状
        original_shape = anchor_1_0['conv2.conv.1.weight'].shape
        tensor = anchor_1_0['conv2.conv.1.weight']

        # 创建一个新的张量，形状为 [96, 192, 1, 1]
        new_shape = (96, 192, original_shape[2], original_shape[3])
        new_tensor = torch.zeros(new_shape)

        # 根据 top_channels_indices 进行拆分
        for i, idx in enumerate(top_channels_indices):
            new_tensor[i, :, :, :] = tensor[idx, :192, :, :]

        return new_tensor

    def split_tensor_by_indices_10(self, anchor_1_0, top_channels_indices):
        # 获取 anchor_1_0['conv2.conv.1.weight'] 的形状
        original_shape = anchor_1_0['conv2.conv.1.weight'].shape
        tensor = anchor_1_0['conv2.conv.1.weight']

        # 创建一个新的张量，形状为 (128, 256, 1, 1)
        new_shape = (128, 256, original_shape[2], original_shape[3])
        new_tensor = torch.zeros(new_shape)

        # 根据 top_channels_indices 进行拆分
        for i, idx in enumerate(top_channels_indices):
            new_tensor[i, :, :, :] = tensor[idx, :256, :, :]

        return new_tensor

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):

        lr = self.agent_list[0].scheduler_list[0].get_lr()[0]

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}'.format(self.epoch, lr))

        return self.epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # epoch = self.scheduler.last_epoch + 1
            epoch = self.epoch
            return epoch >= self.args.epochs

    def _analysis(self):
        flops = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules()
        ])
        flops_conv = torch.Tensor([
            getattr(m, 'flops', 0) for m in self.model.modules() if isinstance(m, nn.Conv2d)
        ])
        flops_ori = torch.Tensor([
            getattr(m, 'flops_original', 0) for m in self.model.modules()
        ])

        print('')
        print('FLOPs: {:.2f} x 10^8'.format(flops.sum() / 10 ** 8))
        print('Compressed: {:.2f} x 10^8 / Others: {:.2f} x 10^8'.format(
            (flops.sum() - flops_conv.sum()) / 10 ** 8, flops_conv.sum() / 10 ** 8
        ))
        print('Accel - Total original: {:.2f} x 10^8 ({:.2f}x)'.format(
            flops_ori.sum() / 10 ** 8, flops_ori.sum() / flops.sum()
        ))
        print('Accel - 3x3 original: {:.2f} x 10^8 ({:.2f}x)'.format(
            (flops_ori.sum() - flops_conv.sum()) / 10 ** 8,
            (flops_ori.sum() - flops_conv.sum()) / (flops.sum() - flops_conv.sum())
        ))
        input()
