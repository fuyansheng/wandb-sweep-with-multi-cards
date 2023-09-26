import wandb
import sys

wandb.login()

sweep_config = {
    'name': 'test_wandb',
    'method': 'grid'
}
metric = {
    'name': 'loss',
    'goal': 'minimize'
}
sweep_config ['metric'] = metric

parameters_dict = {
    'embed_size': {
        'values': [100,200,300]
    }
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="FYS_using_multi_hitter_testing_multi_pross_gpu")

print(sweep_id)
