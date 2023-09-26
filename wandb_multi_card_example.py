import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0，1'
import sys
import argparse
import random
from pathlib import Path
import torch.multiprocessing as mp

import wandb
import launchpad as lp
import pickle
import datetime
import random
import pdb
import socket
from testing_multi import train
# from testing_multi import main

#from config import sweep_config
#from multi_together_train_pipeline import model_pipeline


sweep_id = 'test'
num_gpus = 2
world_size = 2

def parse(args):
    parser = argparse.ArgumentParser(
        description='wandb_usage', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team_name", type=str, default='')
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--experiment_name", type=str,default='test_sweep')
    parser.add_argument("--scenario_name", type=str,default='test_sweeping')
    parser.add_argument("--wandb_log_path", type=str, default="../../wandb_results/")
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--sweep_worker_num",type=int,default=2)
    all_args = parser.parse_known_args(args)[0]
    return all_args

class SweepWorker():
    def __init__(self, args, sweep_id):
        self.sweep_id = sweep_id
        self.args = args

    def run(self):
        all_args = parse(self.args)
        random.seed(all_args.seed)
        wandb.agent(self.sweep_id, function=main)


def make_program(sweep_worker_num,args,sweep_id):
    program = lp.Program('wandb_sweep')
    with program.group('sweep_worker'):
        for _ in range(sweep_worker_num):
            program.add_node(lp.CourierNode(SweepWorker,args,sweep_id))
    return program

def set_wandb(all_args):
    # run_dir = Path("../../../wandb_results") / all_args.project_name / all_args.experiment_name
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    os.environ["WANDB_ENTITY"] = all_args.team_name
    os.environ["WANDB_PROJECT"] = all_args.project_name

    # os.environ["WANDB_DIR"] = str(run_dir)


def test_sweep(args):
    all_args = parse(args)
    set_wandb(all_args)

    program = make_program(all_args.sweep_worker_num,args,sweep_id)
    lp.launch(program, launch_type='local_mp')



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
    test_sweep(sys.argv[1:])
