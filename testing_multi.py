import os


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import wandb
import argparse
import pickle
import datetime
import random
import pdb


# 定义网络模型
class Net(nn.Module):
    def __init__(self, embed_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(embed_size, 1)

    def forward(self, x):
        return self.fc(x)

# 定义训练函数


# def train(local_rank, args):
#     ###todo 关键或许在wandb.init
#
#         print("statr ddp")
#
#
#         with open(args.config_file_name, 'rb') as file:
#            config = pickle.load(file)
#
#         dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456',
#                                 world_size=args.world_size, rank=local_rank)
#
#
#
#         if local_rank == 0:
#             wandb.init()
#         torch.manual_seed(0)
#
#         #config = args.config
#         #embed_size = int(os.environ.get('embed_size'))
#
#
#         embed_size = config['embed_size']
#         model = Net(embed_size)
#
#         # 创建模型并将其移到指定设备
#         model = model.to(local_rank)
#         model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
#
#         # 定义损失函数和优化器
#         criterion = nn.MSELoss()
#         optimizer = optim.SGD(model.parameters(), lr=0.01)
#
#         # 模拟输入数据
#         inputs = torch.randn(64, embed_size).to(local_rank)
#         labels = torch.randn(64, 1).to(local_rank)
#
#         # 训练模型
#         for epoch in range(100000):
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             if local_rank == 0:
#                 wandb.log({"epoch": epoch, "loss": loss})
#
#             optimizer.step()
#
#             print(f"Rank {local_rank}, Epoch {epoch}: Loss {loss.item()}")
#
#         # 释放进程组资源
#         dist.destroy_process_group()
#         os.remove(args.config_file_name)

    # 初始化进程组


def train(local_rank, args):
    ###todo 关键或许在wandb.init

        print("statr ddp")


        with open(args['config_file_name'], 'rb') as file:
           config = pickle.load(file)



        #pdb.set_trace()
        dist.init_process_group(backend='nccl', init_method=args['ddp_address'],  ##todo: 端口需要进行修改
                                world_size=args['world_size'], rank=local_rank)
        # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456',  ##todo: 端口需要进行修改
        #                         world_size=args['world_size'], rank=local_rank)



        if local_rank == 0:
            wandb.init()
        torch.manual_seed(0)

        #config = args.config
        #embed_size = int(os.environ.get('embed_size'))


        embed_size = config['embed_size']
        model = Net(embed_size)

        # 创建模型并将其移到指定设备
        model = model.to(local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 模拟输入数据
        inputs = torch.randn(64, embed_size).to(local_rank)
        labels = torch.randn(64, 1).to(local_rank)

        # 训练模型
        for epoch in range(100000):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if local_rank == 0:
                wandb.log({"epoch": epoch, "loss": loss})

            optimizer.step()

            print(f"Rank {local_rank}, Epoch {epoch}: Loss {loss.item()}")

        # 释放进程组资源
        dist.destroy_process_group()
        os.remove(args.config_file_name)

#
#         # mp.spawn(train, args=(num_gpus,), nprocs=num_gpus)

#
