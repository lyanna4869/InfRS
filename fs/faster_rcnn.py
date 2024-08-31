import logging
from fsdet.modeling.backbone import build_backbone
from fsdet.modeling.proposal_generator import build_proposal_generator
from fsdet.modeling.roi_heads import build_roi_heads
from fsdet.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from fsdet.engine import (
    DefaultTrainer,
)
from fsdet.structures.anno import FeatureResultDict
import os, os.path as osp
from fsdet.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    tsne_on_dataset,
    print_csv_format,
    verify_results,
    DatasetEvaluators,
    FsDetectionEvaluator,
)
from collections import OrderedDict
from fsdet.utils import comm
from fsdet.utils.events import EventStorage
import torch.nn as nn
from fsdet.data.dataset_mapper import AlbumentationMapper, PrototypeDatasetMapper
from fsdet.config import CfgNode

logger = logging.getLogger("fsdet.fs.trainer")


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    def __init__(self, cfg: "CfgNode"):
        self.storage: "EventStorage" = None
        super().__init__(cfg)
        self._best_ap = 0
        self._best_nap = 0

    @classmethod
    def build_model(cls, cfg):
        from fsdet.modeling.meta_arch import GeneralizedRCNN
        # meta_arch = cfg.MODEL.META_ARCHITECTURE
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())  # RPN
        roi_heads = build_roi_heads(cfg, backbone.output_shape())  # specify roi_heads name in yaml

        model = GeneralizedRCNN(backbone, proposal_generator, roi_heads, cfg)

        return model

    @classmethod  # 不用实例化build_evaluator即可调用
    def build_evaluator(cls, cfg, dataset_name: "str", output_folder: "str" = None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "pascal_voc":
            return FsDetectionEvaluator(dataset_name)
        if evaluator_type in ["nwpu", "dior", "dota", "rsod"]:
            evaluator = FsDetectionEvaluator(dataset_name)
            cls_txts = os.path.join(cfg.OUTPUT_DIR, "results")
            evaluator.set_output_dir_name(cls_txts)
            return evaluator

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = None
        # if cfg.INPUT.USE_ALBUMENTATIONS:
        #     mapper = AlbumentationMapper(cfg, is_train=True)
        enable_prototype = cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.ENABLED and \
             cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.SAMPLE
        if enable_prototype:
            mapper = PrototypeDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`fsdet.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = None
        enable_prototype = cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.ENABLED and \
             cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.SAMPLE
        if enable_prototype:
            mapper = PrototypeDatasetMapper(cfg)
        return build_detection_test_loader(cfg, dataset_name, mapper)

    def test(self, cfg, model: "nn.Module", evaluators: "list[DatasetEvaluator]" = None):
        """bsf.c 相比于 DefaultTrainer ,增加了保存最佳模型的功能
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        results = super().test(cfg, model, evaluators)
        if self.storage is None:
            # bsf.c 当单独测试时，不需要保存
            return results
        iter = self.storage.iter + 1
        for key, result in results.items():
            if key == 'bbox':
                ap50 = result['AP50']

                if ap50 > self._best_ap:
                    self._best_ap = ap50
                    self.checkpointer.save("model_best", iteration=iter, result=result)
                    logger.info(f"save best mAP model: {ap50:.3f}")
                nAp50 = result.get('nAP50', 0)
                if nAp50 > self._best_nap:
                    self._best_nap = nAp50
                    self.checkpointer.save("model_best_novel", iteration=iter, result=result)
                    logger.info(f"save best nAP model: {nAp50:.3f}")
        return results



class PrototypeTrainer(Trainer):
    @classmethod
    def build_model(cls, cfg):
        from fsdet.modeling.meta_arch import PrototypeRCNN
        # meta_arch = cfg.MODEL.META_ARCHITECTURE
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())  # RPN
        roi_heads = build_roi_heads(cfg, backbone.output_shape())  # specify roi_heads name in yaml

        model = PrototypeRCNN(backbone, proposal_generator, roi_heads, cfg)

        return model
    
    def prototype(self, cfg, model: "nn.Module", evaluators=None):
        """bsf.c 创建 Prototype 所需的对象
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        idx = 0
        dataset_name = cfg.DATASETS.TEST[idx]
        data_loader = self.build_test_loader(cfg, dataset_name)

        results: "FeatureResultDict" = tsne_on_dataset(model, data_loader, cfg.LOGGING_INTERVAL)
        return results
    
class IncTrainer(Trainer):
    @classmethod
    def build_model(cls, cfg):
        from fsdet.modeling.meta_arch.rcnn import IncrementalRCNN
        # meta_arch = cfg.MODEL.META_ARCHITECTURE
        backbone = build_backbone(cfg)
        proposal_generator = build_proposal_generator(cfg, backbone.output_shape())  # RPN
        roi_heads = build_roi_heads(cfg, backbone.output_shape())  # specify roi_heads name in yaml

        model = IncrementalRCNN(backbone, proposal_generator, roi_heads, cfg)

        return model