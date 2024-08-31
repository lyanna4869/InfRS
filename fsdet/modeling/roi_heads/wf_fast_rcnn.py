import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fsdet.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
                                                  FastRCNNOutputs,
                                                  ROI_HEADS_OUTPUT_REGISTRY,
                                                  fast_rcnn_inference, fast_rcnn_inference_tsne)

## without forgetting 的原版代码
class CustomFastRCNNOutputs(FastRCNNOutputs):
    # def predict_probs(self, score_thresh=0.05, base_bonus=0.):
    #     probs = F.softmax(self.pred_class_logits, dim=-1) # BFP_SHAPE 
    #     bonus = probs.new_zeros(probs.size())
    #     bonus[probs > score_thresh] = base_bonus
    #     # only the first half comes from base detector
    #     num_basedet = probs.size(0) // 2
    #     bonus[num_basedet:, :] = 0
    #     probs += bonus
    #     return probs.split(self.num_preds_per_image, dim=0)
    def predict_probs(self, score_thresh=0.05, base_bonus=0., sizes = None):
        # 当前 batch 只能为 1
        probs = F.softmax(self.pred_class_logits, dim=-1) # BFP_SHAPE Tensor([4800, 21])
        if base_bonus > 0:
            # only the first half comes from base detector
            if sizes is not None:
                bonus = probs.new_zeros(probs.size())
                bonus[probs > score_thresh] = base_bonus  # 必须先 thresh 过滤，然后再加 bonus
                for i in range(0, len(sizes), 2):
                    nidx1 = sizes[i]
                    nidx2 = sizes[i+1]
                    bonus[nidx1: nidx2, :] = 0
                probs += bonus
            else:
                bonus = probs.new_zeros(probs.size())
                bonus[probs > score_thresh] = base_bonus
                num_basedet = probs.size(0) // 2
                # mask = probs[:num_basedet, :] > score_thresh
                # bonus[mask] = base_bonus  # 必须先 thresh 过滤，然后再加 bonus
                bonus[num_basedet:, :] = 0
                probs += bonus
        return probs.split(self.num_preds_per_image, dim=0)
        
    def inference(self, score_thresh, nms_thresh, 
                  topk_per_image, bonus=0.):
        boxes = self.predict_boxes()
        scores = self.predict_probs(score_thresh, bonus)
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh,
            nms_thresh, topk_per_image
        )
    
    def inference_tsne(self, score_thresh, nms_thresh, 
                  topk_per_image, box_features=None):
        boxes = self.predict_boxes()  # BFP_SHAPE Tuple(Tensor(), )
        scores = self.predict_probs(score_thresh)
        image_shapes = self.image_shapes

        return fast_rcnn_inference_tsne(
            boxes, scores, image_shapes, score_thresh,
            nms_thresh, topk_per_image, (box_features, )
        )

## Without Forgetting + Contrastive
class CustomFastRCNNOutputsV1(CustomFastRCNNOutputs):
    """
    Add a multi-task contrastive loss branch for FastRCNNOutputs
    """
    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
        box_cls_feat_con,
        criterion,
        contrast_loss_weight,
        box_reg_weight,
        cl_head_only,
        **kwargs
    ):
        """
        Args:
            box_cls_feat_con (Tensor): the projected features
                to calculate supervised contrastive loss upon
            criterion (SupConLoss <- nn.Module): SupConLoss is implemented in fsdet/modeling/contrastive_loss.py
        """
        super().__init__(box2box_transform, pred_class_logits, pred_proposal_deltas, proposals, smooth_l1_beta)
        
        self.box_cls_feat_con = box_cls_feat_con
        self.criterion = criterion
        self.contrast_loss_weight = contrast_loss_weight
        self.box_reg_weight = box_reg_weight

        self.cl_head_only = cl_head_only
        self.kwargs = kwargs

        # cat(..., dim=0) concatenates over all images in the batch
        # self.proposals = List[Boxes]
        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.ious = torch.cat([p.iou for p in proposals], dim=0)

    def supervised_contrastive_loss(self):
        contrastive_loss = self.criterion(self.box_cls_feat_con, self.gt_classes, self.ious)
        return contrastive_loss

    def losses(self):
        if self.cl_head_only:
            return {'loss_contrast': self.supervised_contrastive_loss()}
        else:
            return {
                'loss_cls': self.softmax_cross_entropy_loss(),
                'loss_box_reg': self.box_reg_weight * self.smooth_l1_loss(),
                'loss_contrast': self.contrast_loss_weight * self.supervised_contrastive_loss(),
            }
logger = logging.getLogger("fs.model.fast_rcnn")
@ROI_HEADS_OUTPUT_REGISTRY.register()
class GeneralizedFastRCNN(FastRCNNOutputLayers):
    def __init__(self, cfg, input_size, num_classes: "int", cls_agnostic_bbox_reg,
                 box_dim=4, cosine_on = False):
        """
        if box_dim is 0, bbox reg is not enabled
        """
        super().__init__(cfg, input_size, num_classes, 
                         cls_agnostic_bbox_reg, box_dim=4)
        if not cosine_on:
            logger.info("cosine_on not given, using fc classifier by default")
            self.cosine = False
        else:
            self.cosine = cosine_on

        self.num_classes = num_classes
        nn.init.normal_(self.cls_score.weight, std=0.01)
        if not self.cosine:
            nn.init.constant_(self.cls_score.bias, 0)
            self.cls_score = nn.Linear(input_size, num_classes + 1, bias=True) # 
        else:
            self.cls_score = nn.Linear(input_size, num_classes + 1, bias=False) # 
            assert hasattr(cfg.MODEL.ROI_HEADS, "COSINE_SCALE")
            self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
            if self.scale == -1:
                # learnable global scaling factor
                self.scale = nn.Parameter(torch.ones(1) * 10.0)
        if not box_dim:
            self.bbox_pred = None

    def forward(self, x, cls_only=False):
        if x.size(0) == 0:
            device = x.device
            score_size = self.num_classes + 1
            delta_size = 4
            return torch.zeros(0, score_size, device=device), \
                torch.zeros(0, delta_size, device=device)

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.cosine:
            # normalize the input x along the `input_size` dimension
            x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
            x_normalized = x.div(x_norm + 1e-5)

            # normalize weight
            temp_norm = torch.norm(self.cls_score.weight.data, p=2, dim=1)\
                .unsqueeze(1).expand_as(self.cls_score.weight.data)
            self.cls_score.weight.data = self.cls_score.weight.data\
                .div(temp_norm + 1e-5)
            cos_dist = self.cls_score(x_normalized)
            scores = self.scale * cos_dist
        else:
            scores = self.cls_score(x)
        if cls_only:
            return scores

        if self.bbox_pred:
            proposal_deltas = self.bbox_pred(x)
        else:
            proposal_deltas = torch.zeros(scores.size(0), 4).to(scores.device)

        return scores, proposal_deltas
