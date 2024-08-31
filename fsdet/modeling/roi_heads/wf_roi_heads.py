import torch
import torch.nn as nn
import torch.nn.functional as F
from fsdet.layers import ShapeSpec
from fsdet.modeling.poolers import ROIPooler
from fsdet.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from fsdet.structures import Instances
from fsdet.modeling.roi_heads.box_head import build_box_head

from fsdet.modeling.roi_heads import StandardROIHeads
from fs.core.multi_similarity_loss import MultiSimilarityLoss
from fsdet.modeling.roi_heads.wf_fast_rcnn import CustomFastRCNNOutputs, CustomFastRCNNOutputsV1, GeneralizedFastRCNN
from fsdet.modeling.wf_utils import ClassSplit
from fsdet.modeling.losses.contrastive_loss import ContrastiveHead, SupConLoss, SupConLossV2
from fsdet.utils.events import get_event_storage
from fsdet.config import globalvar as gv

from .wf_fast_rcnn import CustomFastRCNNOutputs
from fsdet.modeling.roi_heads.roi_heads import StandardROIHeads
import torch
import torch.nn as nn

class BaseRedetectROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        # super().__init__(cfg, input_shape)
        ### bsf.c Note NWPU
        num_cls = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if num_cls == 10:
            self.class_split = ClassSplit(7, 10)
        ### bsf.c Note DIOR
        elif num_cls == 20:
            self.class_split = ClassSplit(15, 20)
        ### RSOD
        elif num_cls == 4:
            self.class_split = ClassSplit(3, 4)

        super().__init__(cfg, input_shape)
        self.basedet_bonus = cfg.MODEL.ROI_HEADS.BASEDET_BONUS
        self.consistency_coeff = cfg.MODEL.ROI_HEADS.COEFF
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self._init_loss(cfg)

    def _init_loss(self, cfg):
        loss_ver = cfg.MODEL.ROI_BOX_HEAD.REDETECT.LOSS
        
        if loss_ver == "KL":
            self.criterion = nn.KLDivLoss(reduction="batchmean") # (n, 15) n objects 15 novel class
        elif loss_ver == "WS":
            from fsdet.modeling.losses.wasserstein_loss import WassersteinLoss
            self.criterion = WassersteinLoss(1, reduction="mean") # (n, 15) n objects 15 novel class
        elif loss_ver == "MS":
            self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
            self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC
            sp = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.SCALE_POS
            sn = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.SCALE_NEG
            lam = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.LAMBDA
            self.criterion = MultiSimilarityLoss(sp, sn, self.contrast_iou_thres, lam, self.reweight_func,)
        else:
            raise ValueError("Not Supported Yet")

    def _init_box_head(self, cfg):
        ## bsf.c see StandardROIHeads._init_box_head
        # fmt: off
        freeze_redetector = cfg.MODEL.ROI_HEADS.FREEZE_REDETECT
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.novel_box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        self.train_on_pred_boxes = False # cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        

        input_size = self.box_head.output_size
        ## bsf.c change box_predictor as always newest, redetector keeps base weight
        self.box_predictor = GeneralizedFastRCNN(cfg, input_size,
                self.class_split.num_classes(),  self.cls_agnostic_bbox_reg, 
                cosine_on=cfg.MODEL.ROI_HEADS.COSINE_ON)
 
        self.redetector = GeneralizedFastRCNN(cfg, input_size,
                self.class_split.num_classes("base"),
                self.cls_agnostic_bbox_reg, 
                cosine_on=False)
        if freeze_redetector:
            self.redetector.requires_grad_ = False
            
    def _forward_base_box(self, features, proposals):
        box_features = self.box_pooler(features,
                                       [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        return box_features, pred_class_logits, pred_proposal_deltas

    def _forward_novel_box(self, box_features, base_class_logits):
        probs = torch.softmax(base_class_logits, dim=1)
        pickup_indices = probs[:, :-1].max(dim=1)[0] <= self.confidence_thresh
        box_features = box_features[pickup_indices]
        pred_class_logits, pred_proposal_deltas = self.redetector(box_features)
        return pickup_indices, pred_class_logits, pred_proposal_deltas
    
    def forward(self, images, features, proposals, targets=None):
        if not self.training:
            return self.inference(images, features, proposals)
        del images
        assert targets
        proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        features_list = [features[f] for f in self.in_features]
        losses = self._forward_box(features_list, proposals)
        return proposals, losses
        
    def _forward_box(self, features, proposals):
        box_features = self.box_pooler(features,
                                       [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform,
            novel_class_logits,
            novel_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        losses = outputs.losses()
        if self.consistency_coeff > 0:
            base_class_logits = self.redetector(box_features, True)
            losses["loss_con_cls"] = self._cls_consistency_loss(
                base_class_logits, novel_class_logits)
        return losses

    def inference(self, images: "torch.Tensor", features: "list[torch.Tensor]", 
                proposals: "list[Instances]", targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        # novel_class_logits, novel_proposal_deltas = self.redetector(box_features)
 
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
            self.basedet_bonus
        )
        return pred_instances, {}
    
    def _cls_consistency_loss(self, base_logits, novel_logits):
        base_mapping = self.class_split.to_all_classes_bool("base", device=base_logits.device, include_bg=True)
        
        novel_logits = novel_logits[:, base_mapping] # (n, 11) n objects 11 base class
        base_logits = base_logits[:, :-1] # (m, 11) n == m
        base_log_probs = self.logsoftmax(base_logits)
        novel_probs = torch.softmax(novel_logits, dim=-1)
        c = self.criterion(base_log_probs, novel_probs)
        return c * self.consistency_coeff


@ROI_HEADS_REGISTRY.register()
class RedetectROIHeads(BaseRedetectROIHeads):

    def inference_2(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals]) # BFP_SHAPE Tensor(500, 256, 7, 7)
        box_features = self.box_head(box_features)  # BFP_SHAPE T(500, 1024)
        base_class_logits, base_proposal_deltas = self.redetector(box_features) # BFP_SHAPE T(500, 8) BFP_SHAPE T(500, 28)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features) # BFP_SHAPE T(500, 11) BFP_SHAPE T(500, 40)

        # resize base results
        temp_logits = base_class_logits.new_zeros(novel_class_logits.size()) # BFP_SHAPE T(500, 11)
        base_mapping = self.class_split.to_all_classes_bool("base", device=temp_logits.device, include_bg=True)
        temp_logits[:, base_mapping] = base_class_logits[:, :-1]
        temp_logits[:, -1] = base_class_logits[:, -1]
        base_class_logits = temp_logits

        temp_deltas = base_proposal_deltas.new_zeros(novel_proposal_deltas.size()) # BFP_SHAPE T(500, 40)
        temp_deltas = temp_deltas.view(temp_deltas.size(0),self.class_split.num_classes(), 4)
        base_proposal_deltas = base_proposal_deltas.view(base_proposal_deltas.size(0), -1, 4)
        temp_deltas[:, base_mapping[:-1]] = base_proposal_deltas
        base_proposal_deltas = temp_deltas.view(temp_deltas.size(0), -1)

        assert base_class_logits.size(0) == novel_class_logits.size(0)
        final_logits = torch.cat([base_class_logits, novel_class_logits],dim=0)
        final_deltas = torch.cat([base_proposal_deltas, novel_proposal_deltas], dim=0)
        final_proposals = [Instances.cat([p, p]) for p in proposals]

        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            final_logits, final_deltas, final_proposals, 
            # base_class_logits, base_proposal_deltas, proposals, 
            self.smooth_l1_beta
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh,
            self.test_detections_per_img,
            self.basedet_bonus
        )
        return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class RedetectROIHeadsV1(RedetectROIHeads):
    """添加 Contrastive
    """
    def __init__(self, cfg, input_shape):
        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self._contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.contrast_loss_weight = 0
        self.contrast_loss_start  = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_START
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.loss_version         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.cl_head_only         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        super().__init__(cfg, input_shape)
        ## bsf.c see ContrastiveROIHeads
        # fmt: off
        # 增加 CPE loss 和 Head
        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
    
    def _init_loss(self, cfg):
        if self.loss_version == 'V1':
            self.criterion = SupConLoss(self.temperature, self.contrast_iou_thres, self.reweight_func)
        elif self.loss_version == 'V2':
            self.criterion = SupConLossV2(self.temperature, self.contrast_iou_thres)
        elif self.loss_version == "MS":
            sp = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.SCALE_POS
            sn = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.SCALE_NEG
            lam = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.LAMBDA
            self.criterion = MultiSimilarityLoss(sp, sn, self.contrast_iou_thres, lam, self.reweight_func,)
        self.criterion.num_classes = self.num_classes  # to be used in protype version
        ### WF_on Base
        # self.inference = self.inference_base
        # self._forward_box = 
        

    def _forward_box(self, features, proposals):
        # 增加 CPE loss
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        novel_box_features = self.novel_box_head(box_features)

        box_features = self.box_head(box_features)
        # novel_class_logits, novel_proposal_deltas = self.redetector(novel_box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(novel_box_features) # cls_logits [1024, 11] novel_proposal_deltas [1024, 40]

        box_features_contrast = self.encoder(novel_box_features)
        
        if self.training and self.weight_decay:
            storage = get_event_storage()
            it = int(storage.iter)
            if it == self.contrast_loss_start:
                self.contrast_loss_weight = self._contrast_loss_weight
            if it in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate
        

        outputs = CustomFastRCNNOutputsV1(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta,
            # box_features_contrast, 
            F.normalize(novel_box_features, dim=1), # bf23.05.18 测试不映射 MLP
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.cl_head_only,
        )
        losses = outputs.losses()
        # 不使用 consistency loss ，将下面两行注释掉
        # base_class_logits = self.redetector(box_features, True)
        # losses["loss_con_cls"] = self._cls_consistency_loss(base_class_logits, novel_class_logits)
        return losses
    # 只有 novel/base branch 进行推理
    def inference(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        # box_features = self.box_head(box_features)
        # novel_class_logits, novel_proposal_deltas = self.redetector(box_features)
        box_features = self.novel_box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features) # redetector 和 box_head 共用 box_pooler， base class效果不好
        # box_features_contrast = self.encoder(box_features)
 
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta,
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            # , self.basedet_bonus
        )
        return pred_instances, {}
    

    # 这个是 Without Forgetting 和 Contrastive 结合的推理
    def inference_compose(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]
        # 增加 CPE loss
        old_box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(old_box_features)
        base_class_logits, base_proposal_deltas = self.redetector(box_features)
        box_features = self.novel_box_head(old_box_features)
        # box_features_contrast = self.encoder(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)

        # resize base results
        temp_logits = base_class_logits.new_zeros(novel_class_logits.size())
        base_mapping = self.class_split.to_all_classes_bool("base", device=temp_logits.device, include_bg=True)
        temp_logits[:, base_mapping] = base_class_logits[:, :-1]  # base class
        temp_logits[:, -1] = base_class_logits[:, -1] # bg
        base_class_logits = temp_logits

        temp_deltas = base_proposal_deltas.new_zeros(novel_proposal_deltas.size())
        temp_deltas = temp_deltas.view(temp_deltas.size(0), self.class_split.num_classes(), 4)
        if base_proposal_deltas.shape[0] > 0:
            base_proposal_deltas = base_proposal_deltas.view(base_proposal_deltas.size(0), -1, 4)
            temp_deltas[:, base_mapping[:-1]] = base_proposal_deltas
            base_proposal_deltas = temp_deltas.view(temp_deltas.size(0), -1)

        assert base_class_logits.size(0) == novel_class_logits.size(0)
        final_logits = torch.cat([base_class_logits, novel_class_logits],
                                 dim=0)
        final_deltas = torch.cat([base_proposal_deltas, novel_proposal_deltas],
                                 dim=0)
        final_proposals = [Instances.cat([p, p]) for p in proposals]

        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            final_logits, final_deltas, final_proposals, 
            # base_class_logits, base_proposal_deltas, proposals, 
            # novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta,
            )
        ## 提高性能时，可将 if 暂时去掉
        if gv.collect_roi_feature_stage is None:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                self.basedet_bonus
            )
        elif gv.collect_roi_feature_stage == "tsne":
            pred_instances, _ = outputs.inference_tsne(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
                box_features
            )
        # elif gv.collect_roi_feature_stage == "flops":
        #### if gv.for_calculate_flops:
        #     return torch.Tensor((3.1,)), {}
        return pred_instances, {}
    


@ROI_HEADS_REGISTRY.register()
class MultisimROIHeads(StandardROIHeads):
    """添加 Contrastive
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self._contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.contrast_loss_weight = 0
        self.contrast_loss_start  = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_START
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.loss_version         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.cl_head_only         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY
        # fmt: off
        # 增加 CPE loss 和 Head
        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)
        if self.loss_version == 'V1':
            self.criterion = SupConLoss(self.temperature, self.contrast_iou_thres, self.reweight_func)
        elif self.loss_version == 'V2':
            self.criterion = SupConLossV2(self.temperature, self.contrast_iou_thres)
        elif self.loss_version == "MS":
            sp = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.SCALE_POS
            sn = cfg.MODEL.ROI_BOX_HEAD.MULTI_SIMILARITY_LOSS.SCALE_NEG
            self.criterion = MultiSimilarityLoss(sp, sn, self.contrast_iou_thres, self.reweight_func)
        self.criterion.num_classes = self.num_classes  # to be used in protype version
        ### WF_on Base
    def forward(self, images, features, proposals, targets=None):
        if not self.training:
            return self.inference(images, features, proposals)
        del images
        assert targets
        proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        features_list = [features[f] for f in self.in_features]
        losses = self._forward_box(features_list, proposals)
        return proposals, losses
    
    def _forward_box(self, features, proposals):
        # 增加 CPE loss
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        class_logits, proposal_deltas = self.box_predictor(box_features)

        box_features_contrast = self.encoder(box_features)
        
        if self.training and self.weight_decay:
            storage = get_event_storage()
            it = int(storage.iter)
            if it == self.contrast_loss_start:
                self.contrast_loss_weight = self._contrast_loss_weight
            if it in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate
        

        outputs = CustomFastRCNNOutputsV1(
            self.box2box_transform, 
            class_logits, proposal_deltas, proposals, 
            self.smooth_l1_beta,
            box_features_contrast,
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.cl_head_only,
        )

        return outputs.losses()
    
    # 只有 novel/base branch 进行推理
    def inference(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        # box_features_contrast = self.encoder(box_features)
 
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta,
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
        )

        return pred_instances, {}
    
from .fast_rcnn import (
    FastRCNNContrastOutputs, PrototypeFastRCNNContrastOutputs
)


@ROI_HEADS_REGISTRY.register()
class IncrementalROIHeads(BaseRedetectROIHeads):

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # fmt: on
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.box_cls_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_CLS_WEIGHT

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        
        self.prototype_enabled    = cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.ENABLED
        self.prototype_dataset    = cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.DATASET
        self.prototype_path       = cfg.MODEL.ROI_BOX_HEAD.PROTOTYPE.PATH
        # fmt: off
        if self.prototype_enabled:  
            prototype_tensor = torch.load(self.prototype_path)  # [num_classes+1, 1024]
            # prototype_tensor = prototype_tensor[:-1, :]  # [num_classes, 1024]
            prototype_label = torch.arange(prototype_tensor.shape[0])
            self.register_buffer('prototype', prototype_tensor)
            self.register_buffer('prototype_label', prototype_label)


    def _forward_box(self, features, proposals):
        prop_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, prop_boxes)
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)

        # box_features_normalized = F.normalize(box_features)

        outputs = CustomFastRCNNOutputs(
            self.box2box_transform,
            novel_class_logits,
            novel_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        losses = outputs.losses()

        ## use prototype
        proto_box_features = self.prototype # torch.cat([self.prototype, box_features])
        novel_class_logits = self.box_predictor(proto_box_features, True)
        base_class_logits = self.redetector(proto_box_features, True)
        losses["loss_con_cls"] = self._cls_consistency_loss(
            base_class_logits, novel_class_logits)
        return losses
    
    def inference(self, images: "torch.Tensor", features: "list[torch.Tensor]", 
                proposals: "list[Instances]", targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        # novel_class_logits, novel_proposal_deltas = self.redetector(box_features)
 
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
        )
        return pred_instances, {}
    
    def _cls_consistency_loss(self, base_logits, novel_logits):
        base_mapping = self.class_split.to_all_classes_bool("base", device=base_logits.device, include_bg=True)
        
        novel_logits = novel_logits[:, base_mapping] # (n, 11) n objects 11 base class
        base_logits = base_logits[:, :-1] # (m, 11) n == m
        base_log_probs = self.logsoftmax(base_logits)
        novel_probs = torch.softmax(novel_logits, dim=-1)
        c = self.criterion(base_log_probs, novel_probs)
        return c * self.consistency_coeff
    
@ROI_HEADS_REGISTRY.register()
class IncContrastROIHeads(IncrementalROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.loss_version         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.cl_head_only         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY

        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)

        self.con_criterion = SupConLoss(self.temperature)
        self.con_criterion.num_classes = self.num_classes

    def _forward_box(self, features: "list[torch.Tensor]", proposals: "list[Instances]"):
        prop_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, prop_boxes)
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        box_features = torch.cat([box_features, self.prototype, ])
        
        box_features_normalized = self.encoder(box_features)

        if self.weight_decay:
            storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = PrototypeFastRCNNContrastOutputs(
            self.box2box_transform, novel_class_logits, novel_proposal_deltas, proposals,
            self.smooth_l1_beta,
            box_features_normalized,
            self.con_criterion, self.contrast_loss_weight, 
            self.box_reg_weight, self.box_cls_weight, self.cl_head_only
        )
        outputs.base_class_num = self.prototype.shape[0]
        losses = outputs.losses()
        if self.prototype_enabled:
            ## use prototype
            proto_box_features = self.prototype # T[7, 1024]
            # proto_box_features = torch.cat([self.prototype, box_features])
            novel_class_logits = self.box_predictor(proto_box_features, True)
            base_class_logits = self.redetector(proto_box_features, True)
            losses["loss_con_cls"] = self._cls_consistency_loss(
                base_class_logits, novel_class_logits)
            # base_class_logits = self.redetector(box_features, True)
            # losses["loss_con_cls"] = self._cls_consistency_loss(
            #     base_class_logits, novel_class_logits)
        return losses

    def inference(self, images: "torch.Tensor", features: "list[torch.Tensor]", 
                proposals: "list[Instances]", targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        prop_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features_list, prop_boxes)
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        # novel_class_logits, novel_proposal_deltas = self.redetector(box_features)
        box_features_normalized = self.encoder(box_features)

        outputs = FastRCNNContrastOutputs(
           self.box2box_transform, novel_class_logits, novel_proposal_deltas, proposals,
            self.smooth_l1_beta,
            box_features_normalized,
            self.criterion, self.contrast_loss_weight, 
            self.box_reg_weight, self.box_cls_weight, self.cl_head_only
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
        )
        return pred_instances, {}
    def _cls_consistency_loss(self, base_logits, novel_logits):
        # base_mapping = self.class_split.to_all_classes_bool("base", device=base_logits.device, include_bg=True)
        
        # novel_logits = novel_logits[:, base_mapping] # (n, 11) n objects 11 base class
        base_logits = base_logits[:, :-1] # (m, 11) n == m
        novel_logits = novel_logits[:, :-1] # (m, 11) n == m
        # base_log_probs = self.logsoftmax(base_logits)
        # novel_probs = torch.softmax(novel_logits, dim=-1)
        c = self.criterion(base_logits, novel_logits)
        return c * self.consistency_coeff


@ROI_HEADS_REGISTRY.register()
class IncContrastROIHeadsV0(IncrementalROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.fc_dim               = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.mlp_head_dim         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.MLP_FEATURE_DIM
        self.temperature          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.TEMPERATURE
        self.contrast_loss_weight = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.box_reg_weight       = cfg.MODEL.ROI_BOX_HEAD.BOX_REG_WEIGHT
        self.weight_decay         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.ENABLED
        self.decay_steps          = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.STEPS
        self.decay_rate           = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.DECAY.RATE

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

        self.loss_version         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.LOSS_VERSION
        self.contrast_iou_thres   = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        self.reweight_func        = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.REWEIGHT_FUNC

        self.cl_head_only         = cfg.MODEL.ROI_BOX_HEAD.CONTRASTIVE_BRANCH.HEAD_ONLY

        self.encoder = ContrastiveHead(self.fc_dim, self.mlp_head_dim)

        self.criterion = SupConLoss(self.temperature)
        self.criterion.num_classes = self.num_classes

    def _forward_box(self, features: "list[torch.Tensor]", proposals: "list[Instances]"):
        prop_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, prop_boxes)
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)

        box_features_normalized = self.encoder(box_features)

        if self.weight_decay:
            storage = get_event_storage()
            if int(storage.iter) in self.decay_steps:
                self.contrast_loss_weight *= self.decay_rate

        outputs = FastRCNNContrastOutputs(
            self.box2box_transform, novel_class_logits, novel_proposal_deltas, proposals,
            self.smooth_l1_beta,
            box_features_normalized,
            self.criterion, self.contrast_loss_weight, 
            self.box_reg_weight, self.box_cls_weight, self.cl_head_only
        )
        losses = outputs.losses()
        if self.prototype_enabled:
            ## use prototype
            proto_box_features = self.prototype # torch.cat([self.prototype, box_features])
            novel_class_logits = self.box_predictor(proto_box_features, True)
            base_class_logits = self.redetector(proto_box_features, True)
            losses["loss_con_cls"] = self._cls_consistency_loss(
                base_class_logits, novel_class_logits)
            # base_class_logits = self.redetector(box_features, True)
            # losses["loss_con_cls"] = self._cls_consistency_loss(
            #     base_class_logits, novel_class_logits)
        return losses

    def inference(self, images: "torch.Tensor", features: "list[torch.Tensor]", 
                proposals: "list[Instances]", targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        prop_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features_list, prop_boxes)
        box_features = self.box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)
        # novel_class_logits, novel_proposal_deltas = self.redetector(box_features)
        box_features_normalized = self.encoder(box_features)

        outputs = FastRCNNContrastOutputs(
           self.box2box_transform, novel_class_logits, novel_proposal_deltas, proposals,
            self.smooth_l1_beta,
            box_features_normalized,
            self.criterion, self.contrast_loss_weight, 
            self.box_reg_weight, self.box_cls_weight, self.cl_head_only
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
        )
        return pred_instances, {}
    def _cls_consistency_loss(self, base_logits, novel_logits):
        base_mapping = self.class_split.to_all_classes_bool("base", device=base_logits.device, include_bg=True)
        
        novel_logits = novel_logits[:, base_mapping] # (n, 11) n objects 11 base class
        base_logits = base_logits[:, :-1] # (m, 11) n == m
        base_log_probs = self.logsoftmax(base_logits)
        novel_probs = torch.softmax(novel_logits, dim=-1)
        c = self.criterion(base_log_probs, novel_probs)
        return c * self.consistency_coeff
        