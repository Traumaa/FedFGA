import torch
import os
import utility
from data import Data
from model import Model
from loss import Loss
from trainer_FL import Trainer
from option import args
import random
import numpy as np
from model.agent_federation import Agent

# login wandb
import wandb
wandb.login(key='abe03d00c9ead3fd3daeb086692ff66c692a79e3')

# # 选择gpu
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# if args.gpus is None:
#     gpus = "0"
#     os.environ["CUDA_VISIBLE_DEVICES"]= gpus
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]


random.seed(0)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
np.random.seed(0)
#print('Flag')
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.enabled=True
torch.backends.cudnn.deterministics=True

# This file create a list of agents...
# It also create a tester that sync at each communication for benchmarking
if checkpoint.ok:
    loader = Data(args)
    agent_list = [Agent(args, checkpoint, my_id) for my_id in range(args.n_agents)] #share ckp...need check if save
    tester = Agent(args, checkpoint, 1828) #a tester for runing test. Assign it a fixed id
    for agent in agent_list:
        agent.make_loss_all(Loss)
    tester.make_loss_all(Loss)
    #loss = Loss(args, checkpoicifar-10-python.tar.gznt)
    t = Trainer(args, loader, agent_list+[tester], checkpoint)
    while not t.terminate():
        if agent_list[0].scheduler_list[0].last_epoch == -1 and not args.test_only:
            t.test()
        t.train() # 训练完成后进行参数同步【包括两部分参数，一部分是共享参数，另一部分是各自参数】，赋值给tester一份参数，然后进行测试
        t.test()

    checkpoint.done()
