import os, os.path as osp
import torch, torch.nn as nn
from fsdet.engine import (
    default_argument_parser,
    launch, LaunchArguments
)
from fsdet.evaluation import (
    verify_results,
)
from fs.utils import check_save_git_info, train_entry
import logging

logger = logging.getLogger("fsdet.nwpu.train")


def main(args=None):
    import fs_nwpu.core as nwpu_core
    nwpu_core.init()
    parser = default_argument_parser()
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(args)  

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

    args = sys.argv[1:] 
    if len(args) == 0:
        args = [
            "--config-file", "configs/NWPU/base_training/split1.yml", "--start-iter", "0"
        ]

    main(args)
