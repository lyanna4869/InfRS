from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import (
    default_argument_parser,
    default_setup,
    launch, LaunchArguments
)
import fsdet.utils.comm as comm  # multi-gpu communication
from fsdet.evaluation import (
    verify_results,
)
from fsdet.checkpoint import DetectionCheckpointer

from .utils import *
from fs.faster_rcnn import Trainer, PrototypeTrainer, IncTrainer

def setup_arg_config(args):
    """
    Create configs and perform basic setups.
    cfg (CfgNode): the full config to be used
    args (argparse.NameSpace): the command line arguments to be logged
    """
    cfg = get_cfg()  # 获取一个基本的配置：default.py
    cfg.merge_from_file(args.config_file)  
    cfg.merge_from_list(args.opts)  
    if args.seed == -1:  # 这是默认设置的seed
        # args.seed = 42483128 
        args.seed = 3407
        cfg.SEED = args.seed  
    print(f"===> cfg.seed: {cfg.SEED}")
    cfg.freeze()  
    set_global_cfg(cfg)  
    default_setup(cfg, args)
    return cfg

def create_val_model(cfg, args, trainer: "Trainer") -> "nn.Module":
    model = trainer.model  # 则调用build_model生成一个model
    if args.eval_iter != -1:
        # load checkpoint at specified iteration
        try:
            it = int(args.eval_iter)
            model_name = f'model_{it:07d}.pth'
        except:
            model_name = f'{args.eval_iter}.pth'
        ckpt_file = os.path.join(cfg.OUTPUT_DIR, model_name)
    else:
        # load checkpoint at last iteration
        ckpt_file = cfg.MODEL.WEIGHTS
    trainer.checkpointer.resume_or_load(
        ckpt_file, resume=False
    )  # 将上个检查点的参数加载进model里
    return model


def entry_prototype(args):
    cfg = setup_arg_config(args)  # 配置参数
    trainer = PrototypeTrainer(cfg)
    model = create_val_model(cfg, args, trainer)

    res = trainer.prototype(cfg, model)
    verify_results(cfg, res)
    feat_dst = osp.join(cfg.OUTPUT_DIR, "results.feature")
    logger.info(f"Feature will be save: {feat_dst}")
    torch.save(res, feat_dst)

    return res

def entry_test(args):
    cfg = setup_arg_config(args)  # 配置参数
    trainer = Trainer(cfg)
    model = create_val_model(cfg, args, trainer)
    # model = trainer.model
    res = trainer.test(cfg, model)
    verify_results(cfg, res)
    return res

def train_entry(args):
    from fs.meta_path import ROOT_PATH
    cfg = setup_arg_config(args)  # 配置参数
    # pdb.set_trace()
    if cfg.TRAINER == "default":
        trainer = Trainer(cfg)
    elif cfg.TRAINER == "IncTrainer":
        trainer = IncTrainer(cfg)
    else:
        trainer = Trainer(cfg)
        
    if args.eval_only:  # 如果设置只评估不训练（eval_only是只评估最后一个检查点）
        model = create_val_model(cfg, args, trainer)
        res = trainer.test(cfg, model)  # 可以开始运行test了，会获得res
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer.resume_or_load(resume=args.resume)  # False
    if args.start_iter != -1:
        trainer.start_iter = args.start_iter
    return trainer.train()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)