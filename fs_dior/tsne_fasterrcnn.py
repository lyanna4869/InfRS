from fs.utils import entry_prototype
from fsdet.engine import (
    default_argument_parser,
    launch, LaunchArguments
)
import torch
import numpy as np
from fs.utils.tsne import TSNEPloter
import os
all_tsne_config = {
    "noms": {
       "dst": "tsne_noms_1_9",
       # "title": "NWPU WITHOUT MSLoss",
        "title":"",
        "model": "noms_model_best_novel",
        "config": "tsne_wf101_10shot_CL_IoU.yml",
    },
    "ms": {
        "dst": "tsne_ms_1_9",
    #    "title": "NWPU WITH MSLoss",
        "title":"",
        "model": "ms_model_best_novel",
        "config": "tsne_wf101_10shot_CL_IoU.yml",
    },
    "tfa": {
       "dst": "tsne_tfa_1_plus",
 #       "title": "NWPU TFA",
        "title":"",
        "model": "model_best_novel",
        "config": "tsne_TFA101_10shot_CL_IoU.yml",
    }
}
def plot_tsne(res, ALL_CLASS):
    tsne_config = all_tsne_config[os.environ.get('TSNE_CONFIG', 'tfa')]

    num_class = len(ALL_CLASS)
    gt = []
    det = []
    try:
        for x in res['features']:
            gt.append(x['label'][0])
            det.append(x['feature'][0])
    except Exception as e:
        print(e)
        print(x['file'])
    gt = np.asarray(gt)
    det = np.asarray(det)
    tsneploter = TSNEPloter(num_class, 30000)
    tsneploter.start(det, gt, tsne_config['title'],
        dst=f"checkpoints/tsne/dior/{tsne_config['dst']}")
    
def main(args = None):
    import fs_dior.core as dior_core
    from fsdet.config import globalvar as gv
    
    gv.instances_per_annotations = 1
    gv.collect_roi_feature_stage = "tsne"
    gv.tsne_manaual_select_instances = False 
    gv.tsne_manaual_select_instances = True  
    gv.tsne_select_instance_count = 30
    gv.tsne_desired_annos = []
    gv.rcnn_inference_post_process = False
    # gv.instances_per_annotations = 1

    dior_core.init()
    ALL_CLASS = dior_core.meta.ALL_CATEGORIES[1]
    plot_only = True  # 从缓存中读取数据
    if plot_only:
        res = torch.load("checkpoints/nwpu/tsne/split1/results.feature")
        plot_tsne(res, ALL_CLASS)
        return
    parser = default_argument_parser()
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(args) 
    
    print("Command Line Args:", args)
    launch_args = LaunchArguments(
        num_gpus_per_machine= args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    #Trainer
    launch(entry_prototype, launch_args)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            "--config-file", "configs/DIOR/tsne/split1.yml", "--start-iter", "0",
            # "--config-file", "configs/DIOR/prototype/select_shot.yml", "--start-iter", "0",
        ]
    main(args)
