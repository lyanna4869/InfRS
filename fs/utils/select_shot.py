import numpy as np
import torch
from fsdet.structures.voc import VocAnnSet, VocObject, VocFile, AnnoDb
from fsdet.structures.anno import *
from argparse import ArgumentParser
import os, os.path as osp
def construct_parser():
    parser = ArgumentParser("select_shot")
    parser.add_argument("--cluster", type=int, default=10)
    parser.add_argument("--inherit", type=str, default=None)
    parser.add_argument("--np-seed", default=None, type=int)
    return parser

def construct_meta_parser():
    parser = ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--label", type=str, default="labelTxt")
    parser.add_argument("--seed", type=int, default=2024)
    return parser

def generate_meta(root: "str", labelDir: "str"):
    fileids = []
    for file in sorted(os.listdir(osp.join(root, labelDir))):
        fileid, ext = osp.splitext(file)
        fileids.append(fileid)
    dst_root = osp.join(root, "ImageSets", "Main")
    os.makedirs(dst_root, exist_ok=True)
    with open(osp.join(dst_root, "trainval.txt"), "w") as f:
        f.write("\n".join(fileids))

def load_class_feat_id(fileloc, CLASS_CONFIG):
    data1: "FeatureResultDict" = torch.load(fileloc)
    NOVEL_CLASSES = CLASS_CONFIG[1]['NOVEL_CLASSES']
    ALL_CLASSES = CLASS_CONFIG[1]['ALL_CLASSES']
    AnnoDb.ALL_CLASSES = ALL_CLASSES
    class_feat: "dict[str, list[np.ndarray]]" = {k: [] for k in ALL_CLASSES}
    class_id = {k: [] for k in ALL_CLASSES}

    raw_records1 : "list[FeatureInstance]" = data1['features']
    for record1 in raw_records1:
        feat1 = record1['feature']
        id1   = record1['oid']
        gt_label = record1['label']
        pred_label = record1['pred']
        gt_cat = ALL_CLASSES[gt_label.item()]
        # if gt_cat in NOVEL_CLASSES:
        ## feature n 维
        class_feat[gt_cat].append(feat1[0])
        class_id[gt_cat].append(id1)
    return class_feat, class_id
ESCAPE_IDSET = {}
CLUSTER = 10 # shots
def get_cluster_by_feat(class_feat, class_id: 'dict[str, list]', dataset: "VocAnnSet", cluster=-1):
    if cluster  == -1:
        cluster = CLUSTER
    class_cluster_idxes = {}
    for cat, feats in class_feat.items():
        ids: "list[int]" = class_id[cat]
        ids = np.stack(ids)
        escape_ids = ESCAPE_IDSET.get(cat, [])
        mask = torch.ones((len(ids), ), dtype=bool)
        ## 找出太小的 object，移除
        for i, id in enumerate(ids):
            obj: "VocObject" = dataset.unique_objects[id]
            if obj.width() < 50 and obj.height() < 50:
                mask[i] = False
                continue
            if id in escape_ids:
                mask[i] = False
            
        ids = ids[mask]
        cluster_idxes = np.random.choice(ids, cluster, replace=False)
        feats = np.stack(feats)
        feats: "list[torch.Tensor]" = torch.as_tensor(feats[mask])

        print(f"{cat:20}, {len(feats)}")
        cluster_idxes = list(sorted(cluster_idxes))
        if cluster < 100:
            assert len(cluster_idxes) >= cluster
        class_cluster_idxes[cat] = cluster_idxes[:cluster]

    return class_cluster_idxes