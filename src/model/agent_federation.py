import enum
import fractions
import os
from importlib import import_module
from sched import scheduler
import torch
import torch.nn as nn
from IPython import embed
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision.utils as tu
from tqdm import tqdm
import copy
from spikingjelly.activation_based import functional, monitor, neuron
import torch.nn.functional as F
from .fun import test_firerate
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    def __init__(self, *args):
        super(Agent, self).__init__()
        print('Init Agent {} and making models...'.format(args[2]))

        self.args = args[0] # args should contain slim rate
        self.ckp = args[1]
        self.my_id = args[2]
        self.crop = self.args.crop
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.precision = self.args.precision
        self.n_GPUs = self.args.n_GPUs
        self.save_models = self.args.save_models
        self.fractions = self.args.fraction_list
        self.num_classes = 100 if "cifar100" in self.args.data_train else 10
        self.top_channels_indices = None

        # To do... If not use resume, share ckp may be safe

        print("Init a List of Models")
        model_list = []
        self.budget_model = {net_f:i for i,net_f in enumerate(self.fractions)}

        # 根据传入的fractions，将moudle拆分成不同规模的网络
        for net_f in self.fractions:
            module = import_module('model.' + self.args.model.lower())
            self.module = module
            new_args = self.args
            new_args.net_fraction = net_f
            model_list.append(module.make_model(new_args))
        self.model_list = model_list

        print("Filter bank synced at initalization!")
        self.sync_at_init()
        if not self.args.cpu:
            print('CUDA is ready!')
            torch.cuda.manual_seed(self.args.seed)

        # temporarily disable data parallel

        self.load_all(
            self.ckp.dir,
            pretrained=self.args.pretrained,
            load=self.args.load,
            resume=self.args.resume,
            cpu=self.args.cpu
        )

        for i, m in enumerate(self.model_list):
            print(self.get_model(i), file=self.ckp.log_file)

        self.summarize(self.ckp)


    def test_all(self, loader_test, timer_test, run, epoch):
        timer_test = timer_test
        num_class = self.num_classes # if "cifar100" in args.data_train else 10
        for i, model in enumerate(self.model_list):  # 对每个模型进行测试
            self.model_list[i] = self.model_list[i].to(self.device)
            self.loss_list[i].start_log(train=False)
            model.eval()
            with torch.no_grad():  # torch.no_grad() 在 PyTorch 中关闭梯度计算
                for img, label in tqdm(loader_test, ncols=80):   # 需要处理lable  img（500,1,28,28）

                    img, label = self.prepare(img, label)
                    label_onehot = F.one_hot(label, num_class).float()  # label(500,10)
                    torch.cuda.synchronize()
                    timer_test.tic()

                    functional.reset_net(model)

                    prediction = model(img)  # (10,500,10)
                    prediction = torch.mean(prediction, dim=0)  # (10,500,10) -> (500,10)
                    if i ==3:
                        firerate = test_firerate(model, img)  # 点火率
                        firerate = firerate[-128:]
                    # firerate_shape = [tensor.shape for tensor in firerate]

                    torch.cuda.synchronize()
                    timer_test.hold()

                    self.loss_list[i](prediction, label_onehot, label, train=False)
                    functional.reset_net(model)

            self.loss_list[i].end_log(len(loader_test.dataset), train=False)
            best = self.loss_list[i].log_test.min(0)
            self.model_list[i] = self.model_list[i].to('cpu')



            for j, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
                self.ckp.write_log(
                    'model {} {}: {:.3f} (Best: {:.3f} from epoch {})'.format(  # model 0 Loss: 2.200 (Best: 2.200 from epoch 0)
                        i,  # 0
                        measure, # loss
                        self.loss_list[i].log_test[-1, j], # 2.200，最后一行的0，1，2分别是当前的loss，Top1 error和Top5
                        best[0][j], # 2.200，best的第一行0，1，2元素分别是最优的loss，Top1 error和Top5
                        best[1][j] + 1 if len(self.loss_list[i].log_test) == len(self.loss_list[i].log_train) else best[1][j], #轮次
                        )
                    )
            if i == 3:
                # 当开始测试1.0模型时，测试倒数第二层的点火率
                # print(print(f"Firerate type: {type(firerate)}"))
                # 在 firerate 计算后，选择通道：
                max_firerate_channels = sorted(enumerate(firerate), key=lambda x: x[1], reverse=True)[:32]
                # 打印或使用这些通道
                print("Top 32 channels with highest firerate:")
                for channel_idx, firerate_value in max_firerate_channels:
                    print(f"Channel {channel_idx}: {firerate_value}")

                # 将通道保存至top_channels_indices
                top_channels_indices = [channel_idx for channel_idx, _ in max_firerate_channels]
                self.top_channels_indices = top_channels_indices
                print("\nfirerate:\n", firerate)
                print(self.top_channels_indices)
                self.plot_firing_rate(firerate)

            # todo: 拆分完整网络参数并赋值给0.25网络
            run.log({"acc @ {}".format(self.fractions[i]): 100-self.loss_list[i].log_test[-1, self.args.top]},step=epoch-1)
            total_time = timer_test.release()
            is_best = self.loss_list[i].log_test[-1, self.args.top] <= best[0][self.args.top]
            self.ckp.save(self, i, epoch, is_best=is_best)
            #self.ckp.save_results(epoch, i, model)
            self.scheduler_list[i].step()

            # return self.top_channels_indices

    def plot_firing_rate(self, firerate):
        # Generate x axis positions
        x_positions = np.arange(len(firerate))
        # Plot the bar chart
        plt.bar(x_positions, firerate, color='blue', alpha=0.7)

        # Set title and labels
        plt.title('Firing Rate Distribution')
        plt.xlabel('Channel')
        plt.ylabel('Model 3 Firing Rate')

        # Save the plot to the 'firerate' folder in the current working directory
        firerate_folder = os.path.join(os.getcwd(), 'firerate')

        # Ensure 'firerate' folder exists, create it if not
        os.makedirs(firerate_folder, exist_ok=True)

        # Find the next available filename in the format 'firerate+[数字从1递增]'
        i = 1
        while True:
            filename = f'epoch_{i}_firerate.png'
            filepath = os.path.join(firerate_folder, filename)
            if not os.path.exists(filepath):
                break
            i += 1

        # Save the plot as an image
        plt.savefig(filepath)
        plt.clf()



    # def plot_firing_rate(self, firerate_data):
    #     # 确保 firerate 文件夹存在
    #     if not os.path.exists("firerate"):
    #         os.makedirs("firerate")
    #
    #     # 遍历每次循环的点火率列表
    #     for n, firerate in enumerate(firerate_data, start=1):
    #         # 创建柱状图
    #         # firerate_list = list(firerate)
    #         plt.bar(range(1, len(firerate) + 1), firerate)
    #         plt.title(f"Epoch {n} Firerate")
    #         plt.xlabel("Channel")
    #         plt.ylabel("Firerate")
    #         plt.savefig(f"firerate/epoch_{n}_firerate.png")
    #         plt.clf()  # 清除图形，准备下一次循环
    #
    #     print("柱状图已保存到 firerate 文件夹中。")

    def budget_to_model(self, budget):
        return self.budget_model[budget]

    def train_local(self, loader_train, budget, epochs):

        num_classes = self.num_classes
        model_id = self.budget_to_model(budget)
        loss_list = []
        loss_orth_list = []
        n_samples = 0
        self.model_list[model_id] = self.model_list[model_id].to(self.device)
        for epoch in range(epochs):
            for batch, (img, label) in enumerate(loader_train):
                img, label = self.prepare(img, label)  # .half()取半精度
                n_samples += img.size(0)  # 32
                label_onehot = F.one_hot(label, num_classes).float() # label_onehot（32，10）

                self.optimizer_list[model_id].zero_grad()  # 用来将特定模型（model_id）对应的优化器的梯度清零。
                prediction = self.forward(img, model_id)    # prediction（10，32，10）
                prediction = torch.mean(prediction, dim=0)  # prediction（32，10） 在第一个维度上取平均
                # prediction = prediction
                # prediction = prediction/10 #args.T

                loss, _ = self.loss_list[model_id](prediction, label_onehot, label)


                loss_orth = self.args.lambdaR*self.module.orth_loss(self.model_list[model_id],self.args,'L2')
                loss = loss_orth + loss
                loss_orth_list.append(loss_orth.item())

                loss.backward()  # 计算网络参数相对于损失函数的梯度

                self.optimizer_list[model_id].step()

                loss_list.append(loss.item())  # 376
                functional.reset_net(self.model_list[model_id])


        log_train = self.loss_list[model_id].log_train[-1,:]/n_samples
        self.model_list[model_id] = self.model_list[model_id].to('cpu')


        return sum(loss_list)/len(loss_list), sum(loss_orth_list)/len(loss_orth_list), log_train
        # loss, loss_orth, log_train


    def train_one_step(self, img, label):
        loss_list = []
        loss_orth_list = []
        for i, _ in enumerate(self.model_list):
            self.optimizer_list[i].zero_grad()
            prediction = self.forward(img, i)
            loss, _ = self.loss_list[i](prediction, label,)


            loss_orth = self.args.lambdaR*self.module.orth_loss(self.model_list[i],self.args,'L2')
            loss = loss_orth + loss
            loss_orth_list.append(loss_orth.item())

            loss.backward()
            self.optimizer_list[i].step()

            loss_list.append(loss.item())
            functional.reset_net(self.model)

        if self.args.sync: self.sync_filter()

        return loss_list, loss_orth_list

    def sync_at_init(self):   # 初始化阶段同步滤波器的权重kv
        n_models = len(self.model_list)
        filter_banks = {}
        for k,v in self.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = v

        for i in range(n_models):
            self.model_list[i].load_state_dict(copy.deepcopy(filter_banks), strict=False)


    def sync_filter(self):   # 初始化滤波器
        n_models = len(self.model_list)
        filter_banks = {}
        for k,v in self.model_list[0].state_dict().items():
            if 'filter_bank' in k:
                filter_banks[k] = torch.zeros(v.shape).cuda()

        for k in filter_banks:
            for model in self.model_list:
                state_dict = model.state_dict()
                filter_banks[k]+=state_dict[k]*(1./n_models)

        for i in range(n_models):

            self.model_list[i].load_state_dict(copy.deepcopy(filter_banks), strict=False)

    def forward(self, x, i):
        if self.crop > 1:
            b, n_crops, c, h, w = x.size()
            x = x.view(-1, c, h, w)
        # from IPython import embed; embed(); exit()
        x = self.model_list[i](x)  # img放在model_list[i]（不同分数的模型进行训练）

        if self.crop > 1: x = x.view(b, n_crops, -1).mean(1)

        return x

    def get_model(self, i):
        if self.n_GPUs == 1:
            return self.model_list[i]
        else:
            return self.model_list[i].module

    def state_dict_all(self, **kwargs):
        ret = []
        for i, _ in enumerate(self.model_list):
            ret.append(self.state_dict(i))
        return ret

    def state_dict(self, i, **kwargs):
        return self.get_model(i).state_dict(**kwargs)

    def save_all(self, apath, epoch, is_best=False):
        for i, _ in enumerate(self.model_list):
            self.save(i, apath, epoch, is_best)

    def save(self, i, apath, epoch, is_best=False):
        target = self.get_model(i).state_dict()

        conditions = (True, is_best, self.save_models)
        names = ('latest', 'best', '{}'.format(epoch))

        for c, n in zip(conditions, names):
            if c:
                torch.save(
                    target,
                    os.path.join(apath, 'model', 'model_m{}_{}.pt'.format(i,n))
                )

    def load_all(self, apath, pretrained='', load='', resume=-1, cpu=False):
        for i, _ in enumerate(self.model_list):
            self.load(i, apath, pretrained, load, resume, cpu)

    def load(self, i, apath, pretrained='', load='', resume=-1, cpu=False):
        f = None
        if pretrained:
            if pretrained != 'download':
                print('Load pre-trained model from {}'.format(pretrained))
                f = pretrained
                # from IPython import embed; embed(); exit()
        else:
            if load:
                if resume == -1:
                    print('Load model {} after the last epoch'.format(i))
                    resume = 'latest'
                else:
                    print('Load model {} after epoch {}'.format(i,resume))

                f = os.path.join(apath, 'model', 'model_m{}_{}.pt'.format(i,resume))

        if f:
            kwargs = {}
            if cpu:
                kwargs = {'map_location': lambda storage, loc: storage}

            state = torch.load(f, **kwargs)
            # from IPython import embed; embed(); exit()

            self.get_model(i).load_state_dict(state, strict=False)

    def begin_all(self, epoch, ckp):
        for i, _ in enumerate(self.model_list):
            self.begin(i, epoch, ckp)

    def begin(self, i, epoch, ckp):
        self.model_list[i].train()
        m = self.get_model(i)
        if hasattr(m, 'begin'): m.begin(epoch, ckp)

    def start_loss_log(self):
        for loss in self.loss_list:
            loss.start_log() #create a tensor

    def log_all(self, ckp):
        for i, _ in enumerate(self.model_list):
            self.log(i, ckp)

    def log(self, i, ckp):
        m = self.get_model(i)
        if hasattr(m, 'log'): m.log(ckp)

    def summarize(self, ckp):
        for i, _ in enumerate(self.model_list):
            ckp.write_log('# parameters of model {}: {:,}'.format(i,
                sum([p.nelement() for p in self.model_list[i].parameters()])
            ))

            kernels_1x1 = 0
            kernels_3x3 = 0
            kernels_others = 0
            gen = (c for c in self.model_list[i].modules() if isinstance(c, nn.Conv2d))
            for m in gen:
                kh, kw = m.kernel_size
                n_kernels = m.in_channels * m.out_channels
                if kh == 1 and kw == 1:
                    kernels_1x1 += n_kernels
                elif kh == 3 and kw == 3:
                    kernels_3x3 += n_kernels
                else:
                    kernels_others += n_kernels

            linear = sum([
                l.weight.nelement() for l in self.model_list[i].modules() \
                if isinstance(l, nn.Linear)
            ])

            ckp.write_log(
                '1x1: {:,}\n3x3: {:,}\nOthers: {:,}\nLinear:{:,}\n'.format(
                    kernels_1x1, kernels_3x3, kernels_others, linear
                ),
                refresh=True
            )
    def make_optimizer_all(self, ckp=None, lr=None):
        ret = []
        for i, _ in enumerate(self.model_list):
            ret.append(self.make_optimizer(i, ckp, lr))
        self.optimizer_list = ret

    def make_optimizer(self, i, ckp=None, lr=None):
        trainable = filter(lambda x: x.requires_grad, self.model_list[i].parameters())

        if self.args.optimizer == 'SGD':
            optimizer_function = optim.SGD
            kwargs = {'momentum': self.args.momentum, 'nesterov': self.args.nesterov}

        kwargs['lr'] = self.args.lr if lr is None else lr
        kwargs['weight_decay'] = self.args.weight_decay  #
        # embed()
        optimizer = optimizer_function(trainable, **kwargs)

        if self.args.load != '' and ckp is not None:
            print('Loading the optimizer from the checkpoint...')
            optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )

        return optimizer
    def make_loss_all(self, Loss):
        self.loss_list = [Loss(self.args, self.ckp) for _ in self.fractions]  # 对不同分数的模型创建一个loss

    def make_scheduler_all(self, resume=-1, last_epoch=-1, reschedule=-1):
        ret = []
        for s in self.optimizer_list:
            ret.append(self.make_scheduler(s, resume, last_epoch, reschedule))
        self.scheduler_list = ret

    def make_scheduler(self, target, resume=-1, last_epoch=-1, reschedule=0):  # 学习率衰减
        if self.args.decay.find('step') >= 0:
            milestones = list(map(lambda x: int(x), self.args.decay.split('-')[1:]))
            kwargs = {'milestones': milestones, 'gamma': self.args.gamma}

            scheduler_function = lrs.MultiStepLR
            # embed()
            kwargs['last_epoch'] = last_epoch
            scheduler = scheduler_function(target, **kwargs)

        if self.args.load != '' and resume > 0:
            for _ in range(resume): scheduler.step()
        if reschedule>0:
            for _ in range(reschedule): scheduler.step()
        return scheduler
    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half': x = x.half()  # 取半精度，加快训练和减少显存
            return x

        return [_prepare(a) for a in args]

