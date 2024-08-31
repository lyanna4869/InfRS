import os, os.path as osp
import torch
from torch import nn as nn
from fsdet.engine import (
    default_argument_parser,
    launch, LaunchArguments
)

from fs.utils import train_entry
import logging

logger = logging.getLogger("fsdet.fs.train")


def main(args=None):
    import fs_dior.core as dior_core
    dior_core.init()
    parser = default_argument_parser()
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(args)  # 调用default_argument_parser()函数会得到一个类，parse_args()是这个类的方法

    print("Command Line Args:", args)
    launch_args = LaunchArguments(
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    # Trainer
    launch(train_entry, launch_args)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:] # sys.argv[1:]则保存的是在命令行中传递的实际参数列表
    if len(args) == 0:
        ## base
        args = [
            "--config-file", "configs/DIOR/base_training/split3.yml", "--start-iter", "0"
            # "--config-file", "configs/DIOR/all/all.yml", "--start-iter", "0"
        ]

    main(args)
