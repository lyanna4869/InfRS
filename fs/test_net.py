# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from datetime import datetime
import logging
import os
import json
import numpy as np
from collections import OrderedDict
from fsdet.evaluation.evaluator import inference_on_dataset, tsne_on_dataset
from fsdet.evaluation.testing import print_csv_format

import fsdet.utils.comm as comm
from fsdet.checkpoint import DetectionCheckpointer
from fsdet.config import get_cfg, set_global_cfg
from fsdet.data import MetadataCatalog
from fsdet.engine import (
    default_setup,
    hooks,
    launch,
)
from fsdet.evaluation import (
    verify_results,
)

from fsdet.engine import DefaultTrainer
from fs.core.evaluation import FsDetectionEvaluator
class Tester(DefaultTrainer):
    # build_model = Trainer.build_model
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR)

        self.best_res = None
        self.best_file = None
        self.all_res = {}
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        return FsDetectionEvaluator(dataset_name)
    
    def test_(self, ckpt):
        m = self.check_pointer._load_file(ckpt)
        self.check_pointer._load_model(m)
        print('evaluating checkpoint {}'.format(ckpt))
        res = DefaultTrainer.test(self.cfg, self.model)

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)

            key = 'nAP50'
            if key not in res['bbox']:
                key = 'AP'

            if (self.best_res is None) or (self.best_res is not None and
                    self.best_res['bbox'][key] < res['bbox'][key]):  # best result by the above key
                self.best_res = res
                self.best_file = ckpt
            print('best results from checkpoint {}'.format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, 'inference'), exist_ok=True)
            with open(os.path.join(self.cfg.OUTPUT_DIR, 'inference',
                                   'all_res.json'), 'w') as fp:
                json.dump(self.all_res, fp)

    @classmethod
    def tsne(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        dataset_name = cfg.DATASETS.TEST[0]
        data_loader = cls.build_test_loader(cfg, dataset_name)        
        results = tsne_on_dataset(model, data_loader, cfg.LOGGING_INTERVAL)
        logger.info("Evaluation TSNE results for {} in csv format:".format(dataset_name))
        return results

import os.path as osp
def setup(args, time_str='test'):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    logger = logging.getLogger("fs.test")
    if args.seed == -1:
        cfg.SEED = 25861963
    elif args.seed == 0:
        pass
    elif args.seed is not None:
        cfg.SEED = args.seed
    cfg.CURRENT_TIME_STR = time_str
    config_file = osp.abspath(args.config_file)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    if args.seed == -1 or args.seed is not None:
        logger.info("Using a generated random seed {}".format(cfg.SEED))
    return cfg


