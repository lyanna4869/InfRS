
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch.nn as nn
from PIL import Image
import os.path as osp, os
import torch

import numpy as np
import torch
import cv2

import logging
logger = logging.getLogger("fsdet.utils")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)
def sim_matrix(a: "torch.Tensor", b: "torch.Tensor", eps=1e-8):
    """a : T[N, d] b: T[M, d]
    added eps for numerical stability
    """
    a_n = a.norm(dim=-1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_n = b.norm(dim=1)[:, None]
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

## bsf.c 测试用
IMAGE_ROOT = "data/images/1003"
os.makedirs(IMAGE_ROOT, exist_ok=True)

def convert_tensor_as_npint8(xim: "torch.Tensor"):
    """xim: T[3, h, w]"""
    assert xim.dim() == 3
    arr = xim.detach().cpu().permute(1, 2, 0).numpy()
    m1 = arr.max()
    m2 = arr.min()
    diff = m1 - m2
    if diff > 1e-3:
        arr = (arr - m2) / (m1 - m2) * 255
    else:
        arr *= 255
    
    arr = arr.astype(np.uint8)
    return arr, m1, m2

def get_im_loc(shape: "tuple", im_idx : "int" = 0, hint="x"):
    H, W = shape
    loc = f"{IMAGE_ROOT}/{hint}{im_idx}_{H}_{W}.png"
    return loc

def save_tensor(xim : "torch.Tensor", im_idx : "int" =0):
    """xim: T[3, h, w]"""
    arr, m1, m2 = convert_tensor_as_npint8(xim)
    im = Image.fromarray(arr)

    H, W = xim.shape[-2:]
    loc = get_im_loc((H, W), im_idx)
    im.save(loc)
    print(f"{loc} max: {m1:.2f} {m2:.2f}")
    return im, loc

def save_np_as_img(xim : "np.ndarray", im_idx : "int" =0):
    """xim: T[h, w, 3]"""
    if xim.dtype != np.uint8:
        xim = mmcv.imdenormalize(xim, **img_norm_cfg)
        xim = xim.astype(np.uint8)
    im = Image.fromarray(xim)

    H, W = xim.shape[:2]
    loc = get_im_loc((H, W), im_idx)
    im.save(loc)
    print(f"{loc}")
    return im, loc

def save_bimg_tensor(xims: "torch.Tensor"):
    """xim: T[B, 3, h, w]"""
    for i, xim in enumerate(xims):
        save_tensor(xim, i)

def save_bimg(batched_inputs: "list[VocAnnotation]"):
    for imidx, binp in enumerate(batched_inputs):
        save_tensor(binp.image, im_idx=imidx)

def save_bimg_with_gtbbox(batched_inputs: "list[VocAnnotation]", offset: "int" = 0):
    """xim: T[B, 3, h, w]
    gt_bboxes  list(xywha)
    gt_bboxes: "list[torch.Tensor]", 
    gt_labels: "list[torch.Tensor]" = None,
    """
    for imidx, binp in enumerate(batched_inputs):
        xim = binp.image
        instance: "Instances" = binp.instances
        bbox: "torch.Tensor" = instance.gt_boxes.tensor
        bbox = bbox.detach().cpu().numpy()
        
        image, m1, m2 = convert_tensor_as_npint8(xim)
        H, W = xim.shape[-2:]
        loc = get_im_loc((H, W), imidx + offset)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for pi, poly in enumerate(bbox):
            color = (0, 0, 225)
            poly = np.array(poly, dtype=np.int32)
            image = cv2.rectangle(image, poly[:2], poly[2:], color, 1, lineType=cv2.LINE_AA)
            center = ((poly[0] + poly[2]) // 2, (poly[1] + poly[3]) // 2)
            image = cv2.putText(image, f"{imidx}-{pi}", center, cv2.FONT_HERSHEY_SIMPLEX, 
                                color=(0, 0, 0), fontScale=1)
        cv2.imwrite(loc, image)
    print(loc)
    # done

def split_bbox_result(bbox_result: "Instances", thr: "float" = 0., 
                      scale:"float" = 1) -> "tuple[np.ndarray, np.ndarray]":
    """将 bbox_result 转为 bbox, label  bbox_result 数组大小为 num_class
    bbox: dim + 1, last column is score
    并将 XYWHA 转为 X1Y1X2Y2X3Y3X4Y4
    """ 
    bboxes: "torch.Tensor" = bbox_result.pred_boxes.tensor
    bboxes = bboxes.detach().cpu().numpy() # BFP_SHAPE list(15)
    bboxes[:, :4] *= scale # img is resized and box need to adapt

    labels: "torch.Tensor" = bbox_result.pred_classes
    labels = labels.detach().cpu().numpy()

    scores: "torch.Tensor" = bbox_result.scores
    scores = scores.detach().cpu().numpy()

    keep = scores > thr
    bboxes = bboxes[keep]
    if bboxes.shape[0] == 0:
        return None
    scores = scores[keep]
    labels = labels[keep]
    bboxes = np.concatenate((bboxes, scores[:, None]), axis=1)
    return bboxes, labels

def split_rpn_bbox_result(bbox_result: "Instances", scale:"float" = 1):
    bboxes: "torch.Tensor" = bbox_result.proposal_boxes.tensor
    labels = np.zeros((bboxes.shape[0], ), dtype=np.int32)
    bboxes = bboxes.detach().cpu().numpy() # BFP_SHAPE list(15)
    bboxes[:, :4] *= scale # img is resized and box need to adapt
    scores: "torch.Tensor" = bbox_result.objectness_logits.sigmoid()
    scores = scores.detach().cpu().numpy()
    bboxes = np.concatenate((bboxes, scores[:, None]), axis=1)

    return bboxes, labels


from fsdet.structures.anno import *
COLORS = [(0, 116, 217), (128, 0, 128), (5, 5, 5), (127, 219, 255), (25, 25, 112), (127, 255, 0), (46, 139, 87), (112, 128, 144), (255, 220, 0), (255, 133, 27), (255, 65, 54), (133, 20, 75), (240, 18, 190), (177, 13, 201), (170, 170, 170), (16, 31, 63), (1, 255, 112)]
def save_bimg_with_predict(batched_inputs: "list[VocAnnotation]", bbox_results: "list[Instances]", 
            thr=0.3, scales: "list[float]"=None, proposal=False, images: "torch.Tensor" = None):
    """xim: T[B, 3, h, w]
    bbox_list  list( [np.ndarray] )
    """
    loc = ""
    for imidx, binp in enumerate(batched_inputs):
        if images is None:
            xim = binp.image
        else:
            xim = images[imidx]

        bbox_result: "Instances" = bbox_results[imidx]
        scale = 1 if scales is None else scales[imidx]
        if not proposal:
            ret = split_bbox_result(bbox_result, thr, scale=scale)
            if ret is None:
                continue
        else:
            ret = split_rpn_bbox_result(bbox_result, scale=scale)
        bboxes, labels = ret
        image, m1, m2 = convert_tensor_as_npint8(xim)
        H, W = xim.shape[-2:]
        loc = get_im_loc((H, W), imidx)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        pi = 0
        h, w, c = image.shape
        scale_x, scale_y = (w / binp.width, h / binp.height)

        for cls_id, polygon in zip(labels, bboxes):
            pi += 1
            color = COLORS[cls_id % len(COLORS)]
            score = polygon[-1]
            poly = np.array(polygon[:-1])
            # poly[::2]  *= scale_x
            # poly[1::2] *= scale_y
            poly = np.array(poly, dtype=np.int32)

            image = cv2.rectangle(image, poly[:2], poly[2:], color, 1, lineType=cv2.LINE_AA)
            center = ((poly[0] + poly[2]) // 2, (poly[1] + poly[3]) // 2)
            image = cv2.putText(image, f"{cls_id}-{pi}-{score*100:.1f}", center, cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
        ## gt box

        for instance in binp.annotations:
            instance: "VocInstance"
            bbox: "np.ndarray" = instance.bbox
            color = (0, 255, 0)
            poly = bbox.copy()
            # poly[::2]  *= scale_x
            # poly[1::2] *= scale_y
            poly = poly.astype(np.int32)

            image = cv2.rectangle(image, poly[:2], poly[2:], color, 1, lineType=cv2.LINE_AA)
            center = ((poly[0] + poly[2]) // 2, (poly[1] + poly[3]) // 2)
        cv2.imwrite(loc, image)
    print(loc)
    
    # done


def save_heatmap(features: "torch.Tensor", scale=8, near=False) -> "Image.Image":
    """ features CHW
    """
    # 1.1 获取feature maps
    # features = ...  # 尺度大小，如：torch.Size([1,80,45,45])
    # 1.2 每个通道对应元素求和
    assert features.ndim <= 3
    if features.ndim == 3:
        heatmap = torch.sum(features, dim=0)  # 尺度大小， 如torch.Size([1,45,45])
        size = features.shape[1:]  # 原图尺寸大小
    else:
        heatmap = features  # 尺度大小， 如torch.Size([1,45,45])
        size = features.shape  # 原图尺寸大小
    src_size = (size[1] * scale, size[0] * scale) # H W
    max_value = torch.max(heatmap)
    min_value = torch.min(heatmap)
    diff_value = (max_value - min_value)
    if diff_value > 1e-3 or abs(max_value - 0) > 1e-3:
        heatmap = (heatmap-min_value) / (max_value - min_value) * 255
    else:
        return None
    # heatmap = heatmap.unsqueeze(0)
    heatmap = heatmap.detach().cpu().numpy().astype(np.uint8)# .transpose(1,2,0)  # 尺寸大小，如：(45, 45, 1)
    if near:
        heatmap = cv2.resize(heatmap, src_size, interpolation=cv2.INTER_NEAREST)  # 重整图片到原尺寸
    else:
        heatmap = cv2.resize(heatmap, src_size, interpolation=cv2.INTER_LINEAR)  # 重整图片到原尺寸
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # H W C
    # 保存热力图
    im = Image.fromarray(heatmap)
    im.save(f"data/images/xh_{src_size[1]}_{src_size[0]}.png")
    print(f"{src_size[1]}_{src_size[0]} max: {max_value:.2f} {min_value:.2f}")
    return heatmap

def mix_heatmap(features: "torch.Tensor", scale=8, hmratio=0.5, indicator="mix", id=0, grid=False):
    """features: T[N, C, H, W] or T[C, H, W]
    使用 save_bimg_with_gtbbox 
    """
    if features.ndim == 4:
        feat = features[id]
    else:
        feat = features
    heatmap = save_heatmap(feat, scale, grid)
    if heatmap is None:
        print("heatmap invalid")
        return
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    
    src_size = heatmap.shape
    rawim = f"{IMAGE_ROOT}/x{id}_{src_size[0]}_{src_size[1]}.png"
    if not osp.exists(rawim):
        print("Not exist", rawim)
        return
    rawim = cv2.imread(rawim) # H W C
    superimposed_img = heatmap * hmratio + rawim * (1-hmratio)
    loc = f"{IMAGE_ROOT}/{indicator}{id}_{src_size[0]}_{src_size[1]}.png"
    print(loc)
    cv2.imwrite(loc, superimposed_img)

def mix_batched_heatmap(features: "list[torch.Tensor]", layer=0, hmratio=0.5, indicator="mix", grid=False):
    feature = features[layer]
    for id in range(feature.shape[0]):
        feat = feature[id]
        scale = 2 ** (2 + layer)
        heatmap = save_heatmap(feat, scale, grid)
        if heatmap is None:
            print("heatmap invalid")
            return
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        src_size = heatmap.shape
        
        rawim = f"{IMAGE_ROOT}/x{id}_{src_size[0]}_{src_size[1]}.png"
        if not osp.exists(rawim):
            print("Not exist", rawim)
            return
        rawim = cv2.imread(rawim) # H W C
        superimposed_img = heatmap * hmratio + rawim * (1-hmratio)
        
        os.makedirs(osp.join(IMAGE_ROOT, "hm"), exist_ok=True)

        loc = f"{IMAGE_ROOT}/hm/{indicator}{id}_{src_size[0]}_{src_size[1]}.png"
        print(loc)
        cv2.imwrite(loc, superimposed_img)
    

def mix_heatmap_separate(features: "torch.Tensor", scale=8, hmratio=0.5, indicator="mix"):
    """features: T[N, C, H, W] or T[C, H, W]
    使用 save_bimg_with_gtbbox 
    """
    if features.ndim == 4:
        feat = features[0]
    else:
        feat = features
    src_size = feat.shape[-2:]
    id = 0
    size = (src_size[0] * scale, src_size[1] * scale)
    
    
    rawim_loc = f"{IMAGE_ROOT}/x{id}_{size[0]}_{size[1]}.png"
    if not osp.exists(rawim_loc):
        print("Not exist", rawim_loc)
        return
    rawim = cv2.imread(rawim_loc) # H W C
    os.makedirs(f"{IMAGE_ROOT}/seperate", exist_ok=True)
    for fidx in range(feat.shape[0]):
        sf = feat[fidx]
        heatmap = save_heatmap(sf, scale)
        dst_img = f"{IMAGE_ROOT}/seperate/{indicator}-{fidx}_{size[0]}_{size[1]}.png"
        if heatmap is None:
            print("heatmap invalid")
            if osp.exists(dst_img):
                os.remove(dst_img)
            continue
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        # print(rawim.shape)
        superimposed_img = heatmap * hmratio + rawim * (1-hmratio)
        cv2.imwrite(dst_img, superimposed_img)

def tensor2img(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    """Convert tensor to 3-channel images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (C, H, W).
        mean (tuple[float], optional): Mean of images. Defaults to (0, 0, 0).
        std (tuple[float], optional): Standard deviation of images.
            Defaults to (1, 1, 1).
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    if torch is None:
        raise RuntimeError('pytorch is not installed')
    assert torch.is_tensor(tensor) and tensor.ndim == 3
    assert len(mean) == 3
    assert len(std) == 3

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if tensor.dtype == torch.float16:
        tensor = tensor.float()
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = mmcv.imdenormalize(
        img, mean, std, to_bgr=to_rgb).astype(np.uint8)
    
    return np.ascontiguousarray(img)