# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from copy import copy
import logging
from fvcore.common.file_io import PathManager
import os.path as osp, os
import json
import numpy as np
from collections import Counter
from fsdet.data import DatasetCatalog, MetadataCatalog
from fs.data.loader import VocAnnoLoader, DatasetMetaInfo, VocAnnotation
from fs.rotate import rotate_instances
from .category import *

default_dirname = "DIOR"

logger = logging.getLogger("fsdet.dior.meta_dior")
class DataLoader(VocAnnoLoader):
    def load_novel_from_all(self, classnames, split_dir):
        """从 全部数据集中，seed file 中加载 novel
        """
        #  xxx_shot2_seed1  xxx_shot2
        seed, shot = self.get_seed_shot()

        split_dir = osp.join(split_dir, f"seed{seed}")
        from . import COMMON_CONFG
        fileids = self._load_shot_files(split_dir, shot, classnames)
        
        logger.info(f"Using Novel Dir {self.AnnoDirName} {self.ImageDirName}")

        fileidset = set()
        for cls, fileids_ in fileids.items():
            for fi in fileids_:
                fileidset.add(fi)

        annos = self.load_instance_from_file_ex(fileidset, classnames, dirname=default_dirname) # 包含当前图片的所有 ins
        ins = sum([len(x["annotations"]) for x in annos])
        annos = sorted(annos, key= lambda x: len(x["annotations"]), reverse=False)
        ishot = int(shot)
        nannos = []
        # nannos.extend(annos)
        INS_COUNT = np.zeros(len(ALL_CATEGORIES[1]))
        for anno in annos:
            # this make sure that dataset_dicts has *exactly* K-shot
            try: 
                int(anno["image_id"])
            except : 
                nannos.append(anno) # enhanced
                continue
            instances = anno["annotations"]
            nins = []
            for instance in instances:
                cid = instance["category_id"]
                if INS_COUNT[cid] < ishot:
                    # anno = np.random.choice(anno, ishot, replace=False)
                    nins.append(instance)
                    INS_COUNT[cid] += 1
            if len(nins) > 0:
                anno["annotations"] = nins
                nannos.append(anno)
        # nannos = rotate_instances(nannos, img_dir=osp.join(dirname, ImageDirName))

        logger.info(f"annotations: {len(nannos)} / {len(annos)} ")
        return nannos

def check_annos_prototype(annos: "list[VocAnnotation]"):
    from fsdet.data import detection_utils as utils
    # for annotations in annos:
    #     anno = utils.transform_instance_annotations(
    #                 annotations[0], transforms, image_shape
    #             )
# 根据情况，加载 base  novel 数据
# 加载 novel 时，优先加载 图片中含 class 数量少，instance 数量少的图片
class DiorMetaInfo(DatasetMetaInfo):
  
  def load_filtered_instances(dsMeta):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"

    Args example:
        name: dior_all1shot
        dirname: VOC2007
        split: novel_10shot_split_3_trainval
    """
    dirname: str = dsMeta.dirname
    split: str = dsMeta.type
    classnames: list = dsMeta.classnames
    loader = DataLoader(dsMeta)
    
    if not loader.is_shots:
        annos = loader.load_from_meta()
        annos = loader.filter_empty_annotations(loader.annotation_list)

        if "prototype" in dsMeta.full_name:
            annos = loader.convert_anno_into_single(annos)
            # check_annos_prototype(annos)
            return annos
        # if "train" in split: 
            # annos = rotate_instances(annos, img_dir=osp.join(dirname, "JPEGImages"), step=10)
        return annos

    if loader.use_more_base:
        # split_dir = osp.join("datasets", default_dirname, f"split{databaseid}")
        return loader.load_more_base(classnames)
    seed, ishot = loader.get_seed_shot()
    # dirname  是 split 之后的路径
    txt_file = osp.join(dirname, f"ImageSets/Main/{split}.txt")
    with open(txt_file) as f:
        fileidset = set()
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            fileidset.add(line)

    annos = DataLoader.load_instance_from_file_ex(fileidset, dsMeta.full_classnames, 
                    dirname=dirname, design_classnames=classnames) # 包含当前图片的所有 ins
    # ins = sum([len(x["annotations"]) for x in annos])
    
    # load novel shots
    ins_counter = Counter()
    nannos = []
    for anno in annos:
        # this make sure that dataset_dicts has *exactly* K-shot
        instances = anno["annotations"]
        nins = []
        for instance in instances:
            cid = instance["category_id"]
            if ins_counter[cid] < ishot:
                # anno = np.random.choice(anno, ishot, replace=False)
                nins.append(instance)
                ins_counter[cid] += 1
        if len(nins) > 0:
            anno["annotations"] = nins
            nannos.append(anno)
            
    logger.info(f"Annotations: {len(nannos)} / {len(annos)} ")
    # nannos = rotate_instances(annos, step=1, img_dir = osp.join(dirname, loader.ImageDirName))
    # logger.info(f"Annotations rotated: {len(nannos)} / {len(annos)} ")
    def sort_anno(name):
        if "_" in name:
            return int(name[:name.find("_")])
        return int(name)
    files = sorted([anno["image_id"] for anno in nannos], key=sort_anno)
    logger.info(f"Ann files: {files}")
    return nannos

def _register_dataset_meta(name, metadata, dirname, ):
    meta = name.split("_")
    dataset_name = meta[0]
    split = meta[1] 
    if len(meta) >= 3:
        # bsf.example:  dior_train_base1
        keepclasses = meta[2]
        sid = int(keepclasses[-1])

        if keepclasses.startswith('base'):
            classnames = metadata["base_classes"][sid]
        elif keepclasses.startswith('novel'):
            classnames = metadata["novel_classes"][sid]
        else:
            classnames = metadata["thing_classes"][sid]
        dsMeta = DiorMetaInfo(name=dataset_name, classnames=classnames, dirname=dirname, type=split, split_id=sid)
        ## bsf.c added full classnames for prototype
        full_classnames = metadata["thing_classes"][sid]
        dsMeta.full_classnames = full_classnames
        dsMeta.full_name = name
        DatasetCatalog.register(
            name, dsMeta.load_filtered_instances
        )
        # print("REGIster meta", name, metadata, id(MetadataCatalog.get(name)))
        md = MetadataCatalog.get(name)
        md.set(
            thing_classes=full_classnames,
            dirname=dirname,
            split=split,
            base_classes=metadata["base_classes"][sid],
            novel_classes=metadata["novel_classes"][sid],
            year=2007
        )
        md.evaluator_type = dataset_name
    elif len(meta) == 2:
        keepclasses = meta[2]
        sid = int(keepclasses[-1])
        if keepclasses.startswith('base'):
            thing_classes = metadata["base_classes"][sid]
        elif keepclasses.startswith('novel'):
            thing_classes = metadata["novel_classes"][sid]
        else:
            thing_classes = metadata["thing_classes"][sid]
        dsMeta = DiorMetaInfo(name=dataset_name, classnames=classnames, dirname=dirname, type=split, split_id=sid)

        DatasetCatalog.register(
            name, 
            dsMeta.load_filtered_instances
        )
        md = MetadataCatalog.get(name)
        md.set(
            thing_classes=thing_classes, dirname=dirname,
            split=split,
            base_classes=metadata["base_classes"][sid],
            novel_classes=metadata["novel_classes"][sid],
            year=2007 
        )
        # md.evaluator_type = cls


def _register_dataset_normalmeta(name, dirname, split, sid=0):
    classnames = ALL_CATEGORIES[sid]
    DatasetCatalog.register(name, lambda: DataLoader.load_from_metafile(dirname, split, classnames=classnames))
    MetadataCatalog.get(name).set(
        thing_classes=classnames, dirname=dirname, split=split,
        evaluator_type = "dior"
    )
    
from fs.builtin import extend_metasplits_split, DatasetSplit
# 注册数据集
def register_all(root="datasets"):
    print("Register Dior ")
    SPLITS = [
        ("dior_trainval", default_dirname, "trainval"),
        ("dior_train", default_dirname, "train"),
        ("dior_val", default_dirname, "val"),
        ("dior_test", default_dirname, "test"),
    ]
    for name, dirname, split in SPLITS:
        _register_dataset_normalmeta(name, os.path.join(root, dirname), split, 0)
        
    # register meta datasets

    all_trainval_dirname = default_dirname # osp.join(default_dirname, "training")  # NWPU/training
    all_train_dirname = osp.join(default_dirname, "training")  # NWPU/training
    all_test_dirname = default_dirname # osp.join(default_dirname, "val")
    METASPLITS = [
        ("dior_trainval_all1", all_trainval_dirname, ),
        ("dior_trainval_all1_prototype", all_trainval_dirname,),  # NWPU/training

        ("dior_train_all1",    all_trainval_dirname, ),
        ("dior_val_all1",      all_test_dirname, ),
        ("dior_test_all1",   all_test_dirname, ),
        ("dior_train_all1_prototype", all_train_dirname,),  # NWPU/training

        ("dior_trainval_base1", all_trainval_dirname, ),
        ("dior_trainval_base1_prototype", all_trainval_dirname,),  # NWPU/training

        ("dior_train_base1",    all_trainval_dirname, ),
        ("dior_val_base1",      all_test_dirname, ),

        ("dior_trainval_all2",  all_trainval_dirname),
        ("dior_train_all2",     all_trainval_dirname),
        ("dior_val_all2",       all_test_dirname),

        ("dior_trainval_base2", all_trainval_dirname),
        ("dior_trainval_base2_prototype", all_trainval_dirname,),  # NWPU/training

        ("dior_train_base2",    all_trainval_dirname),
        ("dior_train_base2_prototype", all_trainval_dirname,),  # NWPU/training
        ("dior_val_base2",      all_test_dirname),

        ("dior_test_base2",  all_test_dirname, ),
        ("dior_test_novel2", all_test_dirname, ),
        ("dior_test_all2",   all_test_dirname, ),
        
        ("dior_test_tsne2", all_test_dirname),

        ("dior_trainval_all3", all_trainval_dirname, ),
        ("dior_train_all3",    all_trainval_dirname, ),
        ("dior_val_all3",      all_test_dirname, ),

        ("dior_trainval_base3", all_trainval_dirname, ),
        ("dior_train_base3",    all_trainval_dirname, ),
        ("dior_train_base3_prototype", all_trainval_dirname,),  # NWPU/training

        ("dior_val_base3",      all_test_dirname, ),

    ]

    # register small meta datasets for fine-tuning stage
    # for prefix in ["all", "novel"]:
        # for sid in range(1, 4):  # dataset idx
    
    prefix = "all"
    available_shots = [3, 5, 10, 20]
    for sid in range(1, 4):
      for shot in available_shots:
        for seed in range(50):
            ddir = osp.join(default_dirname, "train_part", f"seed{seed}_shot{shot}")
            v = extend_metasplits_split(prefix, sid, ddir, shot, seed)
            METASPLITS.append(v[:2])
    prefix = "novel"
    available_seeds = range(30)
    for sid in range(1, 4):
      for shot in available_shots:
        for seed in available_seeds:
            ddir = osp.join(default_dirname, "train_part", f"seed{seed}_shot{shot}")
            v = extend_metasplits_split(prefix, sid, ddir, shot, seed)
            METASPLITS.append(v[:2])
    available_seeds = range(30, 40)
    for sid in range(1, 4):
      for shot in available_shots:
        for seed in available_seeds:
            ddir = all_train_dirname
            v = extend_metasplits_split(prefix, sid, ddir, shot, seed)
            METASPLITS.append(v[:2])
    METASPLITS.append(("dior_train_novel3_50shot_seed1", all_train_dirname))
    METASPLITS.append(("dior_train_novel3_100shot_seed1", all_train_dirname))
    METASPLITS.append(("dior_train_novel3_1000shot_seed1", all_train_dirname))
    METASPLITS.append(("dior_train_novel3_20000shot_seed1", all_train_dirname))


    for name, dirname in METASPLITS:
        _register_dataset_meta(name,
                _get_fewshot_instances_meta(),
                os.path.join(root, dirname)
                )


    # register fine-tune dataset with more base (3ploidy)
    WITH_MORE_BASE = []
    # for prefix in ["all", "novel"]:
    #     extend_morebase(WITH_MORE_BASE, prefix)

    for name, dirname in WITH_MORE_BASE:
        _register_dataset_meta(name,
                _get_fewshot_instances_meta(),
                os.path.join(root, dirname), )
        MetadataCatalog.get(name).evaluator_type = "dior"

