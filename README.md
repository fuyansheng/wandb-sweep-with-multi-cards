# wandb-sweep-with-multi-cards
在单机多卡的情况下使用wandb中的sweep进行多进程的超参数搜索，以及实验过程数据记录。

1. 在不同gpu卡上共同执行一个sweep调参。首先规定好sweep寻找参数的范围，然后执行脚本，输出并记录对应的sweep_id，代码示例：sweep_generate.py
   然后，打开不同的终端窗口，命令行指定特定cuda_id执行程序，例如：CUDA_VISIBLE_DEVICES=0 python model_pipeline.py

2. 实现并行调参数加速，即可以实现在单个gpu同时跑多个process（基于Launchpad）。代码示例：multi_process_example.py

3.实现wandb中sweep的单机多卡并行（基于PyTorch 中的nn.parallel.DistributedDataParallel)。代码示例：wandb_multi_card_example.py

4.实现单机多卡多process的wandb中的sweep。代码示例：multi_process_cards_wandb_example.py
