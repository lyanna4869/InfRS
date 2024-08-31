from fs.utils import entry_test
from fsdet.engine import (
    default_argument_parser,
    launch, LaunchArguments
)


def main(args = None):
    import fs_nwpu.core as nwpu_core
    from fsdet.config import globalvar as gv
    # gv.instances_per_annotations = 1
    gv.collect_roi_feature_stage = None # "rpn"

    nwpu_core.init()
    parser = default_argument_parser()
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(args) #调用default_argument_parser()函数会得到一个类，parse_args()是这个类的方法
    
    print("Command Line Args:", args)
    launch_args = LaunchArguments(
        num_gpus_per_machine= args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    #Trainer
    launch(entry_test, launch_args)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            # "--config-file", "configs/NWPU/base_training/split1.yml", 
            "--config-file", "configs/NWPU/split1/inc/10shot_dmk.yml", "--eval-iter", "model_best"
            # "--config-file", "configs/NWPU/inc/10shot_CLP_baseline.yml", "--start-iter", "0",
            # "--config-file", "configs/NWPU/prototype/select_shot.yml", "--start-iter", "0",
        ]
    main(args)
