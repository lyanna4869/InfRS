
import logging
import numpy as np
import torch
from torch import nn

from fsdet.structures import ImageList
from fsdet.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads  # 修改
from .build import META_ARCH_REGISTRY

from fsdet.utils.events import get_event_storage
from fsdet.utils.visualizer import Visualizer
from fsdet.data.detection_utils import convert_image_to_rgb
from fsdet.modeling.utils import concat_all_gathered
from fsdet.config import CfgNode
from fsdet.config import globalvar as gv
# from ..proposal_generator.rpn import GFSDHeadRPN
from fsdet.structures import Boxes, Instances
__all__ = ["GeneralizedRCNN", "ProposalNetwork"]
from fsdet.structures.anno import VocAnnotation
from fsdet.utils import img as cutil
import copy

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, backbone: "nn.Module", proposal_generator: "nn.Module", roi_heads: "nn.Module", cfg: "CfgNode"):
        super().__init__()
        # fmt on #
        self.input_format = cfg.INPUT.FORMAT
        self.vis_period = cfg.INPUT.VIS_PERIOD
        # fmt off #

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        ## bsf.c 230824 以下代码是原 GRCNN 中通过配置文件初始化模型的代码
        # self.backbone = build_backbone(cfg)
        # self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape()) #RPN
        # self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)
        self.init_by_cfg(cfg)

    def init_by_cfg(self, cfg):
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.BACKBONE.FREEZE_P5:
            for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
                for p in connection.parameters():
                    p.requires_grad = False
            print('frozen P5 in FPN')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
                for p in self.roi_heads.box_head.fc2.parameters():
                    p.requires_grad = True
                print('unfreeze fc2 in roi head')

            # we do not ever need to use this in our works.
            if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
                for p in self.roi_heads.box_head.fc1.parameters():
                    p.requires_grad = True
                print('unfreeze fc1 in roi head')
        print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))

    def forward(self, batched_inputs: "list[VocAnnotation]"):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """

        if not self.training:
            results = self.inference(batched_inputs)
            return results

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features

        # RPN
        if batched_inputs[0].has_annotation():# 取一个batch的第一张图片
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N #把一个batch里所有的gt instances抽取出来
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        # RoI
        # ROI inputs are post_nms_top_k proposals.1000
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def visualize_training(self, batched_inputs: "list[VocAnnotation]", proposals: "list[Instances]"):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        storage = get_event_storage()
        max_vis_prop = 10

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def inference(self, batched_inputs: "list[VocAnnotation]", detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:  # bool(self.proposal_generator)
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # proposals在inference阶段是1000个
            
            results, _ = self.roi_heads(images, features, proposals, None)
            results: "list[Instances]"
            # cutil.save_bimg_with_predict(batched_inputs, results)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        # plot_period = 500
        # storage = get_event_storage()
        # save_result(batched_inputs,results,storage.iter)
        # if abs(storage.iter - plot_period) < 99:
        #     plot_result(batched_inputs, results)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(copy.deepcopy(results_per_image), height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    # def save_results(self, batched_input, results, num_iteration):
    #     return save_result_txt(batched_input, results, num_iteration)
class IncrementalRCNN(GeneralizedRCNN):
    pass


from fsdet.structures.boxes import pairwise_iou
@META_ARCH_REGISTRY.register()
class PrototypeRCNN(GeneralizedRCNN):
    def inference(self, batched_inputs: "list[VocAnnotation]", detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:  # bool(self.proposal_generator)
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # proposals在inference阶段是1000个
            ## bsf.c assign gt to proposal
            # for bi, binp in enumerate(batched_inputs):
            #     proposals[bi].gt_boxes = binp.instances.gt_boxes
            #     proposals[bi].gt_classes = binp.instances.gt_classes
            ## bsf.c use max iou to get anchor
            results, _ = self.roi_heads(images, features, proposals, None)
            # cutil.save_bimg_with_predict(batched_inputs, results)
            results: "list[Instances]"
            new_results = []
            for bindx in range(len(results)):
                res = results[bindx]
                batch_input = batched_inputs[bindx]
                if len(res.pred_boxes) == 0:
                    print("ProtoGRCNN Error", bindx, batch_input.instances)
                    continue
                binp = batch_input.instances.gt_boxes.to(self.device)
                iou = pairwise_iou(res.pred_boxes, binp)
                max_ind = torch.argmax(iou).item()
                ### regenerate Instance
                new_result = Instances(images[bindx].shape)
                new_result.pred_boxes = Boxes(res.pred_boxes.tensor[max_ind:max_ind+1])
                new_result.box_feature = res.box_feature[max_ind:max_ind+1]
                new_result.scores = res.scores[max_ind:max_ind+1]
                new_result.pred_classes = res.pred_classes[max_ind:max_ind+1]
                new_results.append(new_result)
            results = new_results
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image in results:
                processed_results.append({"instances": results_per_image})
            return processed_results
        else:
            return results

## bsf.c 230824.15 将不常用的moco 单独划分为一个 GRCNN
class MocoGeneralizedRCNN(GeneralizedRCNN):
    def __init__(self, backbone: nn.Module, proposal_generator: nn.Module, roi_heads: nn.Module, cfg: CfgNode):
        super().__init__(backbone, proposal_generator, roi_heads, cfg)
        self.moco = cfg.MODEL.MOCO.ENABLED
        
        if self.moco:
            if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
                self.roi_heads._moco_encoder_init(cfg)

            elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
                self.backbone_k = build_backbone(cfg)
                self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
                self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())

                self.roi_heads._moco_encoder_init(cfg,
                                                  self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
        else:
            assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """

        if not self.training:
            results = self.inference(batched_inputs)
            #storage = get_event_storage()
            return self.inference(batched_inputs)
            #return results, self.save_results(batched_inputs, results, storage.iter)

        # backbone FPN
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)  # List of L, FPN features

        # RPN
        if "instances" in batched_inputs[0]:# 取一个batch的第一张图片
             gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N #把一个batch里所有的gt instances抽取出来
        else:
            gt_instances = None

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
            self.roi_heads.gt_instances = gt_instances
        # RoI
        # ROI inputs are post_nms_top_k proposals.1000
        # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            # resize the raw outputs of an R-CNN detector to produce outputs according to the desired output resolution.
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
