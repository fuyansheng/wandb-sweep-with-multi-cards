import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,5'
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import wandb
from testing_multi import train
import pickle
import datetime
import random
import pdb
import argparse
import socket

wandb.login()


sweep_id = 'ohp2dwod'
world_size = 2
num_gpus = 2
#

def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定到一个临时端口
    sock.bind(('localhost', 0))
    # 获取绑定后的端口号
    port = sock.getsockname()[1]
    # 关闭socket
    sock.close()
    # 返回闲置端口号
    return port


def main():
    with wandb.init(config = None):
        config = wandb.config
        print(config)
        config_dict = config._items

        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S_%f")

        unique_filename = f"output_{formatted_time}.pkl"
        with open(unique_filename, 'wb') as file:
            pickle.dump(config_dict, file)
        print(unique_filename)

        port = find_free_port()
        ddp_add = 'tcp://127.0.0.1:' + str(port)

        print(ddp_add)

        args = {}
        args['config_file_name'] = unique_filename
        args['world_size'] = world_size
        args['ddp_address'] = ddp_add

        # 启动多进程训练
        mp.spawn(
            train,
            nprocs=num_gpus,
            args=(args,)
        )


if __name__ == '__main__':

    # 启动多个训练进程
    # mp.spawn(multi_card_run, args=(num_gpus,), nprocs=num_gpus)
    wandb.agent(sweep_id, main, project='FYS_test_multi_gpu')

