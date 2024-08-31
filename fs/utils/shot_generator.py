"""增加 skip_base 功能
"""


import os.path as osp
import os
import multiprocessing as mp
from fsdet.structures.voc import AnnSet, VocAnnSet, merge_annotations, AnnoDb
class ShotsMerger():
    def __init__(self, dataset, shot) :
        self.dataset = dataset
        self.shot = shot

    # step 1
    def merge_novel_annotations(self, dataset: "VocAnnSet", base_dataset: "VocAnnSet", 
                                skip_base=False):
        """basedir: nwpu/novel{shot}
        将 NovelAnnotations{shot}_{cls} 文件夹合并到 NovelAnnotations{shot}_novel 文件夹
        """
        ## NOVEL
        dirname = f"Annotations"
        data_dir = osp.join(dataset.root_path, dirname)
        assert osp.exists(data_dir), data_dir
        dataset.load_from_root("trainval", ann_dir=dirname, skip_cls=AnnoDb.BASE_CLASSES, 
                               allow_same_file=True)

        ##  可以用 labelTxt_base 增加选中的 base object
        ## 如果在 novel shot 中选择已有的 base object ，在 skip_base 设为 False
        if not skip_base:
            base_dataset.load_from_root("trainval", include_name=dataset.get_anno_filenames(), skip_cls=AnnoDb.NOVEL_CLASSES)
            base_dataset.print()
            # 由于 base 不存在于 merge 之后的 文件中，所以这一步不会受影响
            merge_annotations(dataset, base_dataset, self.shot)
        dataset.filter_empty_anno()
        dataset.print()

    def merge_novel_annotations_v1(self, dataset: "VocAnnSet", base_dataset: "VocAnnSet", skip_base=False):
        """basedir: nwpu/novel{shot}
        将 NovelAnnotations{shot}_{cls} 文件夹合并到 NovelAnnotations{shot}_novel 文件夹
        """
        classes = AnnoDb.ALL_CLASSES
        # classes.extend(NOVEL_CLASSES, BASE_CLASSES)
        for cls in classes:
            # dirname = f"NovelAnnotations{self.shot}_{cls}"
            dirname = f"labelTxt"
            data_dir = osp.join(dataset.root_path, dirname)
            # print(data_dir)
            if not osp.exists(data_dir): 
                continue
            # print("Loading from : ", cls)
            if cls in AnnoDb.NOVEL_CLASSES:
                dataset.load_from_root(ann_dir=dirname, skip_cls=AnnoDb.BASE_CLASSES, allow_same_file=True)
            elif not skip_base:
                dataset.load_from_root(ann_dir=dirname, skip_cls=AnnoDb.NOVEL_CLASSES, allow_same_file=True)
            # dataset.print()
        # return
        ## def populate_base(self, novel_dataset):
        ## 如果在 novel shot 中选择已有的 base object ，在 skip_base 设为 False
        if not skip_base:
            base_dataset.load_from_root(include_name=dataset.get_anno_filenames(), skip_cls=AnnoDb.NOVEL_CLASSES)
            base_dataset.print()
            # 由于 base 不存在于 merge 之后的 文件中，所以这一步不会受影响
            merge_annotations(dataset, base_dataset, self.shot)
        dataset.print()

    def get_include_names(self, dataset: "VocAnnSet", anno_dir):
        exclude_names = dataset.get_anno_filenames()
        include_names = []
        dst = osp.join(anno_dir, "ImageSets", "Main", "trainval.txt")
        with open(dst) as f:
            for file in f:
                if file in exclude_names:
                    continue
                include_names.append(file)
        return include_names
    # step 2
    def pull_base_if_not_meet(self, dataset: "VocAnnSet", base_dataset: "VocAnnSet"):
        include_names = self.get_include_names(dataset, base_dataset.root_path)
        base_dataset.load_from_root("trainval", include_name=include_names, skip_cls=AnnoDb.NOVEL_CLASSES)
        base_dataset.print()
        data_stats = dataset.class_count()
        
        def calc_remain_class(remain_data):
            rcs = {}
            for k, v in remain_data.items():
                if v < self.shot:
                    rcs[k] = self.shot - v 
            return rcs
        remains = calc_remain_class(data_stats)
        if len(remains) == 0: return

        meet_classes = set()
        meet_classes.update(AnnoDb.NOVEL_CLASSES)
        # os.makedirs(dst_dir)

        for fileid, anno in base_dataset.unique_annotations.items():
           
            contain_needed = False
            for rc, count in remains.items():
                if rc not in anno.clsset:
                    continue
                contain_needed = True
                size = len(anno.object_per_cls[rc])
                
                if size > count:
                    # 每幅图最多选两个对象
                    obj_size = min(2, count)
                    anno.object_per_cls[rc] = anno.object_per_cls[rc][:obj_size]
            if not contain_needed:
                continue
            for cls in anno.clsset:
                if cls not in remains:
                    anno.object_per_cls.pop(cls)
            anno._update_objects()

            dataset.add_annotations(anno, fileid)

            remains = calc_remain_class(dataset.class_count())
            if len(remains) == 0: 
                return

    def save_novel(self, novel_dataset, dst_dir):
        sum_object = 0
        for fileid, anno in novel_dataset.unique_annotations.items():
            loc = osp.join(dst_dir, f"{AnnoDb.IMG_ID_PREFIX}{fileid}.txt")
            sum_object += len(anno.all_objects)
            # print(loc, sum_object)
            novel_dataset.save_anno(anno, loc)
            # print(loc)
        # novel_only_dataset.dumptxt()
        # novel_dataset.save_novel(f"NovelAnnotations{shot}_novel")
        novel_dataset.print()

from PIL import Image, ImageDraw, ImageFilter
import shutil as sh
import numpy as np
from copy import deepcopy
import random
from fsdet.structures.voc import VocObject, VocFile
from fs.core.iou_calc import _bbox_iou_np_single


def copy_raw_images(root_path: "str", dataset: "VocAnnSet",  
        src_ext: "str" = None, link=False, dst_img_dir_name: "str"="images"):

        # dataset.save_novel("labelTxt")
        src_img_dir = AnnoDb.INSTANCE_SELECT_SOURCE
        dst_img_dir = osp.join(root_path, dst_img_dir_name) # TAG MASKRAW
        os.makedirs(dst_img_dir, exist_ok=True)
        if src_ext is None:
            src_ext = AnnoDb.IMG_EXT
        for fileid, anno in dataset.annos.items():
            
            src_img_loc = osp.join(src_img_dir, f"{fileid}.{src_ext}")
            dst_img_loc = osp.join(dst_img_dir, f"{fileid}.{AnnoDb.IMG_EXT}")
            # sh.copy(src_img_loc, dst_img_loc)
            if not osp.exists(dst_img_loc):
                if link:
                    if osp.islink(dst_img_loc):
                        os.unlink(dst_img_loc)
                    os.symlink(src_img_loc, dst_img_loc)
                elif src_ext == AnnoDb.IMG_EXT:
                    sh.copy(src_img_loc, dst_img_loc)
                else:
                    im = Image.open(src_img_loc)
                    im.save(dst_img_loc)
                    
        dataset.print()