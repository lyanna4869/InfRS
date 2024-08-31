from fs.utils import entry_prototype, setup_arg_config
from fsdet.engine import (
    default_argument_parser,
    launch, LaunchArguments
)
import torch
import numpy as np
from fs.utils.tsne import TSNEPloter
import os, os.path as osp

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
def plot_tsne(res: "torch.Tensor", ALL_CLASS: "list[str]", tsne_config: "dict"):
    # tsne_config = all_tsne_config[os.environ.get('TSNE_CONFIG', 'tfa')]

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
    tsne_fig_loc = osp.join(tsne_config['OUTPUT_DIR'], tsne_config['DST'])
    tsneploter.start(det, gt, tsne_config['TITLE'],
        dst=tsne_fig_loc)
    
def main(args = None):
    import fs_nwpu.core as nwpu_core
    from fsdet.config import globalvar as gv
    ## bsf.c tsne 是从 ROI Heads 最后一部分收集 feature
    gv.instances_per_annotations = 1
    gv.collect_roi_feature_stage = "tsne"
    gv.tsne_manaual_select_instances = False # 自动选取 instance
    gv.tsne_manaual_select_instances = True  # 手动选取
    gv.tsne_select_instance_count = 30
    gv.tsne_desired_annos = []
    gv.rcnn_inference_post_process = False
    # gv.instances_per_annotations = 1

    nwpu_core.init()
    ALL_CLASS = nwpu_core.meta.ALL_CATEGORIES[1]
    
    parser = default_argument_parser()
    parser.add_argument("--only-plot", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=-1)
    args = parser.parse_args(args) #调用default_argument_parser()函数会得到一个类，parse_args()是这个类的方法
    
    config = setup_arg_config(args)
    config.defrost()
    config.TSNE.OUTPUT_DIR = config.OUTPUT_DIR
    # config.freeze()
    plot_only = args.only_plot  # 从缓存中读取数据
    feat_loc = osp.join(config.OUTPUT_DIR, "results.feature")
    if plot_only:
        # res = torch.load("checkpoints/nwpu/tsne/split1/results.feature")
        res = torch.load(feat_loc)
        plot_tsne(res, ALL_CLASS, config.TSNE)
        return

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

    res = torch.load(feat_loc)
    plot_tsne(res, ALL_CLASS, config.TSNE)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 0:
        args = [
            "--config-file=configs/NWPU/tsne/split1_tfa.yml", "--start-iter", "0",
            # "--config-file", "configs/NWPU/prototype/select_shot.yml", "--start-iter", "0",
        ]
    main(args)
