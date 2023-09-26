import os
import sys
import argparse
import random
from pathlib import Path

import wandb
import launchpad as lp


from pipeline import pipeline  #此处为了举例，实际使用时该函数要自行添加引用

def parse(args):
    parser = argparse.ArgumentParser(
        description='wandb_usage', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team_name", type=str, default='')
    parser.add_argument("--project_name", type=str, default=")
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
        wandb.agent(self.sweep_id, function=model_pipeline)


def make_program(sweep_worker_num,args,sweep_id):
    program = lp.Program('wandb_sweep')
    with program.group('sweep_worker'):
        for _ in range(sweep_worker_num):
            program.add_node(lp.CourierNode(SweepWorker,args,sweep_id))
    return program

def set_wandb(all_args):

    os.environ["WANDB_ENTITY"] = all_args.team_name
    os.environ["WANDB_PROJECT"] = all_args.project_name



def test_sweep(args):
    all_args = parse(args)
    set_wandb(all_args)

    sweep_id = 'test'
    program = make_program(all_args.sweep_worker_num,args,sweep_id)
    lp.launch(program, launch_type='local_mp')

if __name__ == '__main__':
    test_sweep(sys.argv[1:])
