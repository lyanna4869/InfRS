# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from fvcore.common.file_io import PathManager
import os.path as osp
import numpy as np
from collections import Counter

from fsdet.structures.anno import VocAnnotation
from fsdet.data import DatasetCatalog, MetadataCatalog
from fs.data.loader import VocAnnoLoader, DatasetMetaInfo
from fs.rotate import rotate_instances
from .category import *

default_dirname = "NWPU"
logger = logging.getLogger("fsdet.nwpu.meta")


class DataLoader(VocAnnoLoader):
    def load_novel_from_all(self, classnames, split_dir):
        """从 全部数据集中，seed file 中加载 novel
        """
        #  xxx_shot2_seed1  xxx_shot2
        seed, shot = self.get_seed_shot()

        split_dir = osp.join(split_dir, f"seed{seed}")

        fileids = self._load_shot_files(split_dir, shot, classnames)

        logger.info(f"Using Novel Dir {self.AnnoDirName} {self.ImageDirName}")

        fileidset = set()
        for cls, fileids_ in fileids.items():
            for fi in fileids_:
                fileidset.add(fi)

        annos = self.load_instance_from_file_ex(fileidset, classnames)  # 包含当前图片的所有 ins
        ins = sum([len(x["annotations"]) for x in annos])
        annos = sorted(annos, key=lambda x: len(x["annotations"]), reverse=False)
        ishot = int(shot)
        nannos = []
        # nannos.extend(annos)
        ins_counter = Counter()
        for anno in annos:
            # this make sure that dataset_dicts has *exactly* K-shot
            try:
                int(anno["image_id"])  # error if _ is used
            except:
                nannos.append(anno)  # enhanced
                continue
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
        # nannos = rotate_instances(nannos, img_dir=osp.join(dir_name, ImageDirName))

        logger.info(f"annotations: {len(nannos)} / {len(annos)} ")
        return nannos

    def load_filtered_instances(self,
                                name: str, dirname: str, split: str, classnames: str):
        """ 该方法暂时未使用
        # 根据情况，加载 base  novel 数据
        # 加载 novel 时，优先加载 图片中含 class 数量少，instance 数量少的图片
        Load Pascal VOC detection annotations to Detectron2 format.

        Args:
            dirname: Contain "Annotations", "ImageSets", "JPEGImages"
            split (str): one of "train", "test", "val", "trainval"

        Args example:
            name: nwpu_all1shot
            dirname: VOC2007
            split: novel_10shot_split_3_trainval
        """
        if not self.is_shots:
            return self.load_all_instances(classnames)  # tsne过滤器在这里

        databaseid = 1
        for n in name.split("_"):
            if n.startswith("all"):
                databaseid = int(n[-1])
                break
        split_dir = osp.join("datasets", default_dirname, f"split{databaseid}")

        if self.use_more_base:
            return self.load_more_base(classnames, split_dir)
        #  xxx_shot2_seed1  xxx_shot2
        annos = self.load_all_from_directory(classnames)  # load from trainpart
        before = len(annos)
        img_dir = osp.join(dirname, self.ImageDirName)
        # annos = rotate_instances(annos, step=1, img_dir=img_dir)
        logger.info(f"Annotations: {len(annos)} , before: {before}")
        return annos

    # 读取 shot 对应的文件
    def _load_shot_files(split_dir, shot, classnames):
        """根据 meta file 加载数据，现在使用 voc 格式。
        可根据id加载shot 返回 dict
        """
        fileids = {}
        for cls in classnames:
            loc = osp.join(split_dir, f"box_{shot}shot_{cls}_train.txt")
            with PathManager.open(loc) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [fid.split('/')[-1].split('.jpg')[0] for fid in fileids_]
                fileids[cls] = fileids_
        return fileids



# 根据情况，加载 base  novel 数据
# 加载 novel 时，优先加载 图片中含 class 数量少，instance 数量少的图片
class NwpuMetaInfo(DatasetMetaInfo):

  def check_annos_prototype(self, annos: "list[VocAnnotation]"):
    from fsdet.data import detection_utils as utils
    # for annotations in annos:
    #     anno = utils.transform_instance_annotations(
    #                 annotations[0], transforms, image_shape
    #             )

  def load_filtered_instances(dsMeta):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"

    Args example:
        name: nwpu_all1shot
        dirname: NWPU
        split: novel_10shot_split_3_trainval
    """

    dirname: str = dsMeta.dirname
    split: str = dsMeta.type
    classnames: list = dsMeta.classnames
    loader = DataLoader(dsMeta)

    if not loader.is_shots:
        # 读取文件，加载 base class 

        loader.load_from_meta()
        annos = loader.filter_empty_annotations(loader.annotation_list)
        if "prototype" in dsMeta.full_name:
            annos = loader.convert_anno_into_single(annos)
            dsMeta.check_annos_prototype(annos)
            if "num" in dsMeta.full_name:
                ishot = int(dsMeta.full_name[-2:])
                annos = dsMeta.filter_shots(annos, ishot)
            return annos
        # if "train" in split:
        #     annos = rotate_instances(annos, img_dir=osp.join(dirname, "JPEGImages"))
        return annos

    if loader.use_more_base:
        return loader.load_more_base(classnames)
    #  xxx_shot2_seed1  xxx_shot2
    seed, ishot = loader.get_seed_shot()
    # dirname  是 split 之后的路径
    file_loc = osp.join(dirname, "ImageSets/Main/trainval.txt")
    with open(file_loc) as f:
        fileidset = set()
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            fileidset.add(line)
    fileidset = list(sorted(fileidset))
    ## bsf.c 这里只用 novel object 训练的时候需要将 offset 设置为 base class 的数量，否则设为 0
    ## bsf.global_novel_offset
    annos = DataLoader.load_instance_from_file_ex(fileidset, dsMeta.full_classnames, 
                dirname=dirname, design_classnames=classnames)  # 包含当前图片的所有 ins
    annos = sorted(annos, key=lambda x: len(x["annotations"]), reverse=False)
    nannos = dsMeta.filter_shots(annos, ishot)
    
    logger.info(f"Annotations: {len(nannos)} / {len(annos)} ")
    # bsf.c 增加旋转增强
    # nannos = rotate_instances(nannos, img_dir=osp.join(dirname, "JPEGImages"))
    logger.info(f"Annotations rotated: {len(nannos)} / {len(annos)} ")

    def sort_anno(name):
        if "_" in name:
            return int(name[:name.find("_")])
        return int(name)

    files = sorted([anno["image_id"] for anno in nannos], key=sort_anno)
    logger.info(f"Ann files: {files}")
    return nannos
  
  def filter_shots(self, annos, ishot):
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
    return nannos

def _register_dataset_meta(name, dirname, metadata, ):
    meta = name.split("_")
    dataset_name = meta[0]
    split = meta[1]
    if len(meta) >= 3:
        ## 
        keepclasses = meta[2]
        sid = int(keepclasses[-1])

        if keepclasses.startswith('base'):
            classnames = metadata["base_classes"][sid]
        elif keepclasses.startswith('novel'):
            classnames = metadata["novel_classes"][sid]
        else:
            classnames = metadata["thing_classes"][sid]
        dsMeta = NwpuMetaInfo(name=dataset_name, classnames=classnames, dirname=dirname, type=split, split_id=sid)
        ## bsf.c added full classnames for prototype
        full_classnames = metadata["thing_classes"][sid]
        dsMeta.full_classnames = full_classnames
        dsMeta.full_name = name
        DatasetCatalog.register(
            name,
            dsMeta.load_filtered_instances
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
        classnames = metadata["thing_classes"][0]
        dsMeta = DatasetMetaInfo(name=dataset_name, classnames=classnames, dirname=dirname, type=split)
        dsMeta.full_name = name
        # DatasetCatalog.register(
        #     name, 
        #     lambda: DataLoader.load_from_metafile(dsMeta)
        # )
        md = MetadataCatalog.get(name)
        md.set(
            thing_classes=classnames, dirname=dirname, split=split
        )
        md.evaluator_type = dataset_name


from fs.builtin import extend_metasplits_split


# ==== Predefined splits for NWPU ===========
# 注册数据集


def register_all(root="datasets"):
    logger.info("Register nwpu")

    SPLITS = [
        ("nwpu_trainval", default_dirname,),
        ("nwpu_train", default_dirname,),
        ("nwpu_val", default_dirname,),
        ("nwpu_test", default_dirname,),
    ]
    # sid = 1 default_dirname=NWPU

    all_trainval_dirname = osp.join(default_dirname, "training")  # NWPU/training
    all_test_dirname = osp.join(default_dirname, "val")
    METASPLITS = [
        ("nwpu_trainval_base1", all_trainval_dirname,),  # NWPU/training
        ("nwpu_trainval_base1_prototype", all_trainval_dirname,),  # NWPU/training

        ("nwpu_trainval_all1_prototype", all_trainval_dirname,),  # NWPU/training
        ("nwpu_trainval_all1_prototype_num30", all_trainval_dirname,),  # NWPU/training
        ("nwpu_trainval_all1_prototype_num50", all_trainval_dirname,),
        ("nwpu_trainval_all1", all_trainval_dirname,),

        ("nwpu_test_base1", all_test_dirname,),  # NWPU/val
        ("nwpu_test_novel1", all_test_dirname,),
        ("nwpu_test_all1", all_test_dirname,),

        ("nwpu_trainval_base2", all_trainval_dirname,),
        ("nwpu_trainval_base2_prototype", all_trainval_dirname,),  # NWPU/training
        
        ("nwpu_trainval_all2_prototype", all_trainval_dirname,),  # NWPU/training
        ("nwpu_trainval_all2", all_trainval_dirname,),

        ("nwpu_test_base2", all_test_dirname,),
        ("nwpu_test_novel2", all_test_dirname,),
        ("nwpu_test_all2", all_test_dirname,),

        ("nwpu_trainval_base3", all_trainval_dirname,),
        ("nwpu_trainval_base3_prototype", all_trainval_dirname,),  # NWPU/training
        ("nwpu_trainval_all3_prototype", all_trainval_dirname,),  # NWPU/training
        ("nwpu_trainval_all3", all_trainval_dirname,),

        ("nwpu_test_novel3", all_test_dirname,),
        ("nwpu_test_base3", all_test_dirname,),
        ("nwpu_test_all3", all_test_dirname,),
    ]
    METASPLITS.extend(SPLITS)

    METASPLITS.append(("nwpu_test_tsne1", all_test_dirname,))
    available_shots = [1, 3, 5, 10, 20]
    availabel_seeds = [1, 3, 4, 5, 8, 10, 11, 20, 30]
    for sid in range(1, 4):
        prefix = "all"
        for shot in available_shots:
            for seed in availabel_seeds:
                ddir = osp.join(default_dirname, "train_part", f"seed{seed}_shot{shot}")
                v = extend_metasplits_split(prefix, sid, ddir, shot, seed, dataset="nwpu")
                METASPLITS.append(v[:2])
        prefix = "novel"
        for shot in available_shots:
            for seed in availabel_seeds:
                # ddir = default_dirname
                ddir = osp.join(default_dirname, "train_part", f"seed{seed}_shot{shot}")
                v = extend_metasplits_split(prefix, sid, ddir, shot, seed, dataset="nwpu")
                METASPLITS.append(v[:2])
    nwpu_meta_things = _get_nwpu_fewshot_instances_meta()

    for name, dirname, in METASPLITS:
        dst_dir = osp.join(root, dirname)
        _register_dataset_meta(name, dst_dir, nwpu_meta_things, )
