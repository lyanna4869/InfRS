import torch
import torch.nn as nn
import torch.nn.functional as F
# from fs.core.model.roi_heads import BaseRedetectROIHeads
# from fs.core.multi_similarity_loss import MultiSimilarityLoss
from fsdet.layers import ShapeSpec
from fsdet.layers.nms import batched_nms
from fsdet.modeling.poolers import ROIPooler
from fsdet.modeling.roi_heads.roi_heads import  ROI_HEADS_REGISTRY
from fsdet.structures import Instances
from fsdet.modeling.roi_heads.box_head import build_box_head
from fsdet.modeling.roi_heads.wf_fast_rcnn import CustomFastRCNNOutputs, CustomFastRCNNOutputsV1, GeneralizedFastRCNN
from fsdet.modeling.wf_utils import ClassSplit
from fsdet.modeling.losses.contrastive_loss import ContrastiveHead, SupConLoss, SupConLossV2
from fsdet.utils.events import get_event_storage
from fsdet.modeling.roi_heads.fast_rcnn import FastRCNNContrastOutputs

@ROI_HEADS_REGISTRY.register()
class RedetectROIHeads(BaseRedetectROIHeads):

    def _init_box_head(self, cfg):
        
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        self.class_split = ClassSplit(15, 20)

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
        self.box_predictor = GeneralizedFastRCNN(cfg, input_size,
                                                 self.class_split.num_classes("base"),
                                                 self.cls_agnostic_bbox_reg)
 
        self.redetector = GeneralizedFastRCNN(cfg, input_size,
                                              self.class_split.num_classes(),
                                              self.cls_agnostic_bbox_reg, 
                                              cosine_on=cfg.MODEL.ROI_HEADS.COSINE_ON)

    def forward(self, images, features, proposals, targets=None):
        if not self.training:
            return self.inference(images, features, proposals)
        del images
        assert targets
        proposals = self.label_and_sample_proposals(proposals, targets) # 添加 ious predict_bboxes
        del targets
        features_list = [features[f] for f in self.in_features]
        outputs, losses = self._forward_box(features_list, proposals)
        ### BFP_ADDED
        def get_proposal(i):
            proposal = proposals[i]
            prop_shape = len(proposal)
            pred_class_logits = outputs.pred_class_logits[:prop_shape]
            t = torch.max(pred_class_logits, axis=1)  # BFP_SHAPE (512, 16)  -> (512, )
            scores = F.softmax(pred_class_logits, dim = -1)
            scores = scores[:, :-1]
            boxes = outputs.predict_boxes()[0]
            proposal.set("pred_cls", t[1])  # BFP_SHAPE (512, 5)
            num_bbox_reg_classes = boxes.shape[1] // 4
            boxes = boxes.view(-1, num_bbox_reg_classes, 4)
            filter_mask = scores > 0.05  # R x K  BFP_VALUE: score_thresh 0.2
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero()  
            boxes = boxes[filter_mask]
            scores = scores[filter_mask]
            # Apply per-class NMS
            keep = batched_nms(boxes, scores, filter_inds[:, 1], 0.5)
            boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
            proposal.proposal_boxes.tensor = boxes.clone().detach()
        # for i in range(len(proposals)):
        #     get_proposal(i)
        # with torch.no_grad():
        #     get_proposal(0)
        # proposals = {
        #     "bg_idx": outputs.pred_class_logits.shape[1] - 1,
        #     "prop": proposals
        # }
        del outputs
        ### BFP_ADDED_END
        return proposals, losses

    def inference_2(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals]) # BFP_SHAPE Tensor(500, 256, 7, 7)
        box_features = self.box_head(box_features)  # BFP_SHAPE T(500, 1024)
        base_class_logits, base_proposal_deltas = self.box_predictor(box_features) # BFP_SHAPE T(500, 8) BFP_SHAPE T(500, 28)
        novel_class_logits, novel_proposal_deltas = self.redetector(box_features) # BFP_SHAPE T(500, 11) BFP_SHAPE T(500, 40)

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

    def _cls_consistency_loss(self, base_logits, novel_logits):
        base_mapping = self.class_split.to_all_classes_bool("base", device=base_logits.device, include_bg=True)
        
        novel_logits = novel_logits[:, base_mapping] # (n, 11) n objects 11 base class
        base_logits = base_logits[:, :-1] # (m, 11) n == m
        base_log_probs = self.logsoftmax(base_logits)
        novel_probs = torch.softmax(novel_logits, dim=-1)
        c = self.loss(base_log_probs, novel_probs)
        return c * self.consistency_coeff


@ROI_HEADS_REGISTRY.register()
class RedetectROIHeadsV1(RedetectROIHeads):
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

        self.num_classes          = cfg.MODEL.ROI_HEADS.NUM_CLASSES

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
        
    def _forward_box(self, features, proposals):
        # 增加 CPE loss # features[0] list(Tensor(B, C, H, W)) BFP_SHAPE features list(5, T(B == 16, C == 256, H == 200, W == 200))
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals]) # BFP_SHAPE box_features (2048, 256, 7, 7)
        novel_box_features = self.novel_box_head(box_features)  # BFP_SHAPE T(128 batch_size_per_image * B, 1024)
        
        box_features = self.box_head(box_features)
        base_class_logits = self.box_predictor(box_features, True)
        # novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features)

        novel_class_logits, novel_proposal_deltas = self.redetector(novel_box_features)
        box_features_contrast = self.encoder(novel_box_features)  # BFP_SHAPE T(2048, 128)
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
            box_features_contrast,
            self.criterion,
            self.contrast_loss_weight,
            self.box_reg_weight,
            self.cl_head_only,
        )
        losses = outputs.losses()
        # 不使用 consistency loss ，将下面两行注释掉
        losses["loss_con_cls"] = self._cls_consistency_loss(base_class_logits, novel_class_logits)
        return outputs, losses
    # 只有 novel branch 进行推理
    def inference(self, images, features, proposals, targets=None):
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]

        box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        # box_features = self.box_head(box_features)
        # novel_class_logits, novel_proposal_deltas = self.box_predictor(box_features) # redetector 和 box_head 共用 box_pooler， base class效果不好
        box_features = self.novel_box_head(box_features)
        novel_class_logits, novel_proposal_deltas = self.redetector(box_features) # redetector 和 box_head 共用 box_pooler， base class效果不好
 
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta,
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
        )
        return pred_instances, {}
    

    # 这个是 Without Forgetting 和 Contrastive 结合的推理
    def inference_mix(self, images, features, proposals, targets=None):
        batch = len(images)
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]
        # 增加 CPE loss
        old_box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(old_box_features)
        base_class_logits, base_proposal_deltas = self.box_predictor(box_features)

        box_features = self.novel_box_head(old_box_features)
        novel_class_logits, novel_proposal_deltas = self.redetector(box_features)

        # resize base results
        temp_logits = base_class_logits.new_zeros(novel_class_logits.size())
        base_mapping = self.class_split.to_all_classes_bool("base", device=temp_logits.device, include_bg=True)
        temp_logits[:, base_mapping] = base_class_logits[:, :-1]  # base class
        temp_logits[:, -1] = base_class_logits[:, -1] # bg
        base_class_logits = temp_logits

        temp_deltas = base_proposal_deltas.new_zeros(novel_proposal_deltas.size())
        temp_deltas = temp_deltas.view(temp_deltas.size(0), self.class_split.num_classes(), 4)
        base_proposal_deltas = base_proposal_deltas.view(base_proposal_deltas.size(0), -1, 4)
        temp_deltas[:, base_mapping[:-1]] = base_proposal_deltas
        base_proposal_deltas = temp_deltas.view(temp_deltas.size(0), -1)

        assert base_class_logits.size(0) == novel_class_logits.size(0)
        final_proposals = [Instances.cat([p, p]) for p in proposals]
        sizes = None
        final_logits = torch.cat([base_class_logits, novel_class_logits], dim=0)
        final_deltas = torch.cat([base_proposal_deltas, novel_proposal_deltas], dim=0)
        if batch > 1:
            #  拆分 logits
            sizes = []
            osizes = [0]
            for p in proposals:
                px = p.objectness_logits.shape[0]
                sizes.append(px)
                sizes.append(px)
                osizes.append(px)
            for i in range(1, batch * 2):
                sizes[i] += sizes[i - 1]
            for i in range(1, batch+1):
                osizes[i] += osizes[i - 1]
            bsizes = [0]
            bsizes.extend(sizes[:-1])
            
            for i in range(batch):
                bidx1 = osizes[i]
                bidx2 = osizes[i+1]
                nidx1 = bsizes[i*2]
                nidx2 = bsizes[i*2+1]
                final_logits[nidx1: nidx2] = base_class_logits[bidx1: bidx2]
                final_deltas[nidx1: nidx2] = base_proposal_deltas[bidx1: bidx2]
                nidx1 = sizes[i*2]
                nidx2 = sizes[i*2+1]
                final_logits[nidx1: nidx2] = novel_class_logits[bidx1: bidx2]
                final_deltas[nidx1: nidx2] = novel_proposal_deltas[bidx1: bidx2]
       
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            pred_class_logits = final_logits, 
            pred_proposal_deltas = final_deltas, 
            proposals = final_proposals, 
            smooth_l1_beta = self.smooth_l1_beta
        )
        pred_instances, _ = outputs.inference(
            self.test_score_thresh, self.test_nms_thresh, 
            self.test_detections_per_img,
            self.basedet_bonus, 
        )
        
        return pred_instances, {}
    
    # 这个是 Without Forgetting 和 Contrastive 结合的推理 for tsne
    def inference_tsne(self, images, features, proposals, targets=None):
        batch = len(images)
        del images
        assert not self.training, "re-detect only supports inference"
        del targets
        features_list = [features[f] for f in self.in_features]
        # 增加 CPE loss
        old_box_features = self.box_pooler(features_list, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(old_box_features)
        base_class_logits, base_proposal_deltas = self.box_predictor(box_features)

        box_features = self.novel_box_head(old_box_features)
        novel_class_logits, novel_proposal_deltas = self.redetector(box_features)

        # resize base results
        temp_logits = base_class_logits.new_zeros(novel_class_logits.size())
        base_mapping = self.class_split.to_all_classes_bool("base", device=temp_logits.device, include_bg=True)
        temp_logits[:, base_mapping] = base_class_logits[:, :-1]  # base class
        temp_logits[:, -1] = base_class_logits[:, -1] # bg
        base_class_logits = temp_logits

        temp_deltas = base_proposal_deltas.new_zeros(novel_proposal_deltas.size())
        temp_deltas = temp_deltas.view(temp_deltas.size(0), self.class_split.num_classes(), 4)
        base_proposal_deltas = base_proposal_deltas.view(base_proposal_deltas.size(0), -1, 4)
        temp_deltas[:, base_mapping[:-1]] = base_proposal_deltas
        base_proposal_deltas = temp_deltas.view(temp_deltas.size(0), -1)

        assert base_class_logits.size(0) == novel_class_logits.size(0)
       
        outputs = CustomFastRCNNOutputs(
            self.box2box_transform, 
            novel_class_logits, novel_proposal_deltas, proposals, 
            self.smooth_l1_beta
        )

        # pred_instances, _ = outputs.inference(
        #     self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
        #     self.basedet_bonus
        # )
        pred_instances, _ = outputs.inference_tsne(
            self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img,
            box_features
        )
        
        return pred_instances, {}