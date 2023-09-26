import wandb
import sys

## todo 模型结构问题或loss定义方式问题  ; 可能出现问题的地方：attention——mask构成 ；   masked_label的集合


#from HittER import HiTTERModel
# import sys
# sys.path.append(r'../dataset/coke_reader.py')


from torch.utils.data import Dataset
from torchvision import transforms

import math
import collections


wandb.login()

sweep_config = {
    'name': 'mkg_on_dataset',
    'method': 'grid'
}
metric = {
    'name': 'train_loss_avg_batch_of_one_epoch',
    'goal': 'minimize'
}
sweep_config ['metric'] = metric

parameters_dict = {
    'experiment_name':{
      'value': 'testing_multi_gpu,multi_process'
    },
    'batch_size': {
      'value': 128
    },
    'epoch_num':{
        'values': [300]
    },
    'embed_size': {
        'values': [320]
    },
    'num_layers': {
        'values': [6]
    },
    'n_head': {
        'values': [8]
    },
    'intermediate_size': {
        'values': [1280]
    },
    'embed_drop': {
        'values': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85]
    },
    'attn_drop': {
        'values': [0.1]
    },
    'ffn_drop': {
        'values': [0.1]
    },
    'initializer_range': {
        'values': [0.02]
    },
    'context_neighbor_drop': {
        'values': [0.5]
    },
    'src_mlm_drop': {
        'values': [0.8]
    },
    'src_mask_rate': {
        'values': [0.6]
    },
    'src_replace_rate': {
        'values': [0.3]
    },
    'max_neighbor':{
        'values': [12]
    },
    'label_smoothing':{
        'values': [0.3]
    },
    'add_mlm_loss':{
        'value': True
    },
    'learning_rate': {
        'values': [2e-4]
    },
    'weight_decay':{
        'values': [1e-4]
    },
    'warm_up_epochs':{
        'value': 25
    },
    'multi_fusion_choose':{
        'value': 'concat_linear'
    },
    'multi_fusion_linear_drop':{
        'values': [0.1]
    },
    'multi_fusion_concat_drop':{
        'values': [0.1]
    },
    'model_choose':{
        'value': 'without_linear'
    },


    'contrastive_negative_sum':{
        'values': [0]
    },
    'add_contrastive_loss':{
        'value': True
    },
    'contrastive_add_neighbor':{
        'value': False
    },
    'contrastive_loss_rate':{
        'values': [1]
    }

}

#
# ##todo: multi_fusion_choose的取值可以选择： concat_linear // element_product, model_choose可取值包括： without_linear // with linear
#
#
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="FYS_using_multi_hitter_testing_multi_pross_gpu")

print(sweep_id)
