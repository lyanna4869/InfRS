__version__ = "1.5"
"""
2.0 使用 compare_imageset_{shot} ids.txt 进行生成
2.1_rsod

"""

import multiprocessing as mp
import os
import os.path as osp
import sys
from copy import copy, deepcopy
import argparse
import os, random
from tqdm import tqdm
from functools import partial

from fs.utils import set_random_seed
from fs.meta_path import *
from fs.utils import MyProcessPool

from fsdet.structures.voc import VocAnnSet, AnnoDb

from fs.utils.shot_generator import ShotsMerger, copy_raw_images

DEBUG_INFO = False

# MaskGenerator.save_anno = DotaDataset.save_anno


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=50, type=int)
    parser.add_argument("--shots", default=None, type=int)
    parser.add_argument("--gather-thresh", default=0.25, type=float)
    # parser.add_argument("--gather-novel-thresh", default=0.8, type=float)
    parser.add_argument("--gather-nshots", default=15, type=int)
    # parser.add_argument("--gather-nimgs", default=8, type=int)
    parser.add_argument("--blur-radius", default=100, type=int)
    parser.add_argument("--steps", default='123c', type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--single-thread", action="store_true")
    parser.add_argument("--ranges", type=str, help="For compatibility of previous scripts")
    return parser

from fs_rsod.core.category import CLASS_CONFIG

AnnoDb.ALL_CLASSES = CLASS_CONFIG[1]["ALL_CLASSES"]
AnnoDb.BASE_CLASSES = CLASS_CONFIG[1]["BASE_CLASSES"]
AnnoDb.NOVEL_CLASSES = CLASS_CONFIG[1]["NOVEL_CLASSES"]


def main(args=None):
    parser = get_parser()
    args = vars(parser.parse_args(args))
    print(args)

    seed = args["seeds"]
    shot = args["shots"]
    if shot is None: shot = seed
    global DEBUG_INFO
    DEBUG_INFO = args["debug"]

    novel_dst_dir = osp.join(DATASETS_PATH, f"RSOD/compare_imageset_{shot}")
    novel_dataset = VocAnnSet(novel_dst_dir, "novel_shot")
    merger = ShotsMerger(novel_dataset, shot)
    # s1
    if '1' in args["steps"] or '2' in args["steps"]:
        raw_anno_dir = osp.join(DATASETS_PATH, "RSOD/train")
        base_dataset = VocAnnSet(raw_anno_dir, "novel_base")

        merger.merge_novel_annotations(novel_dataset, base_dataset, skip_base=args['skip_base'])
        AnnoDb.Datasets.pop("novel_base")
        if args["debug"]:
            # novel_dataset.save_novel("NovelAnnotations{shot}_novel")
            novel_dataset.print()
    dst_dir = osp.join(novel_dst_dir, f"NovelAnnotations{shot}_novel")
    os.makedirs(dst_dir, exist_ok=True)
    if '2' in args["steps"]:
        raw_anno_dir = osp.join(DATASETS_PATH, "RSOD", "train")
        base_dataset = VocAnnSet(raw_anno_dir, "base_only")
        if not args['skip_base']:
            merger.pull_base_if_not_meet(novel_dataset, base_dataset)
        # step last
        sum_object = 0
        for fileid, anno in novel_dataset.unique_annotations.items():
            loc = osp.join(dst_dir, f"{AnnoDb.IMG_ID_PREFIX}{fileid}.xml")
            sum_object += len(anno.all_objects)
            # print(loc, sum_object)
            novel_dataset.save_anno(anno, loc)
            # print(loc)
        # novel_only_dataset.dumptxt()
        # novel_dataset.save_novel(f"NovelAnnotations{shot}_novel")
        novel_dataset.print()

    # 开始生成 train_part 目录下的文件
    dst_dir = osp.join(novel_dst_dir, f"NovelAnnotations{shot}_novel")
    seed_dir = f"seed{seed}_shot{shot}"
    root_path = osp.join(DATASETS_PATH, "RSOD", "train_part", seed_dir)
    os.makedirs(root_path, exist_ok=True)
    dst_root = osp.join(root_path, "Annotations")
    novel_dataset.saveAllAnnos(dst_root)
    # step 1
    AnnoDb.INSTANCE_SELECT_SOURCE = osp.join(DATASETS_PATH, "RSOD", "train", "JPEGImages")
    if '3' in args["steps"]:
        print("******* mask step1: select instances from original folder whose shots may be over  ")
        dataset = VocAnnSet(novel_dst_dir, "selected_data")
        dataset.load_from_root(f"trainval")
        novel_dataset = dataset

    ### generate trainval.txt
    from select_shot import generate_meta
    generate_meta(root_path, "Annotations")

    if 'c' in args['steps']:
        # dataMasker.copy_raw_images(dataset)
        copy_raw_images(root_path, novel_dataset, dst_img_dir_name="JPEGImages")


if __name__ == "__main__":
    sargs = sys.argv[1:]
    if len(sargs) == 0:
        sargs = [
            "--steps", "123c",
            "--seeds", "5", "--shots", "3",
            "--single-thread"
        ]
    set_random_seed(20220934)
    main(sargs)

    ## bsf.c please mannualy change images folder into JPEGImages
    #### last: filter_novel_shot
    #### next: python fs/dota/novel_scripts/fewshot_prepare_dota_split.py
