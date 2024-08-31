"""
1.0 默认无 seed 字符时使用 seed 0 目录
1.1 提供选项，可将 seed 文件合并到一个label文件钟
"""

__version__ = "1.1"

import argparse
import copy
import os
import random
import numpy as np
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager
import os.path as osp

CUR_DIR = osp.abspath(osp.dirname(__file__))

BASE_DIR = osp.join(CUR_DIR, "DIOR")
from fs_dior.core.meta import ALL_CATEGORIES, NOVEL_CATEGORIES
split=2
VOC_CLASSES =  ALL_CATEGORIES[split]

def getAnnotations(data):
    annos_per_cat = {c: [] for c in VOC_CLASSES}  ## images
    for fileid in data:
        anno_file = osp.join(BASE_DIR, f"Annotations", fileid + ".xml")
        if not osp.exists(anno_file): 
            continue
        tree = ET.parse(anno_file)
        clses = set()
        ins_count = {}
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            cls = cls.replace(" ", "-")
            if cls not in clses:
                clses.add(cls)
                ins_count[cls] = 1
            else:
                ins_count[cls] += 1
        try:
            for cls in clses:
                annos_per_cat[cls].append( (fileid, ins_count) )
        except Exception as e:
            print(anno_file, cls, e)
    # print(annos_per_cat["storage-tank"])
    for c in annos_per_cat.keys():
        catins = annos_per_cat[c]
        annos_per_cat[c] = sorted(catins, key=lambda x: sum([v for v in x[1].values()]) )
    
    return annos_per_cat

INSTANCE_SWITCH = {c: 0 for c in VOC_CLASSES}
spliter = 1
def generate_cls_states(cls, fileinfos, dst_files):
    result = {}
    result_other = {}
    
    for fileinfo in fileinfos:
        fileid = fileinfo[0]
        anno_file = osp.join(BASE_DIR, f"Annotations", fileid + ".xml")
        if not osp.exists(anno_file): 
            print("Not found" , anno_file)
            continue
        result[fileid] = 0
        result_other[fileid] = 0
        tree = ET.parse(anno_file)
        for obj in tree.findall("object"):
            objcls = obj.find("name").text
            if objcls == cls:
                result[fileid] += 1
            else:
                result_other[fileid] += 1

    # exit(0)
    print("==== ", cls, sum(x for x in result.values()))
    count_info = []
    if dst_files is None:
        for fileid, count in result.items():
            if count == 0: continue
            count_info.append((fileid, count))
            # print(fileid, count)
        count_info = sorted(count_info, key=lambda x: x[1])
        for f, c in count_info:
            if c > 10: continue
            print(f, c)
    else:
        for fileid in dst_files[cls]:
            rof = result_other.get(fileid, 0)
            if rof > 0:
                count_info.append((fileid, result[fileid], "-", rof))
            else:
                count_info.append((fileid, result[fileid]))

        print(count_info)
    return count_info


def generate_stats(args):
    global spliter
    spliter = args["index"]
    seed = args["seed"]
    print(args)
    all = True
    if all:
        data_file = osp.join(BASE_DIR, 'ImageSets/Main/trainval.txt')
        with PathManager.open(data_file) as f:
            fileids = np.loadtxt(f, dtype=str).tolist()
    else:
        fileids = set()       
        root_dir = osp.join(BASE_DIR, f"split{spliter}/seed{seed}")
        for shot_files in os.listdir(root_dir):
            with open(osp.join(root_dir, shot_files)) as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0: continue
                    line = line[line.rfind("/")+1:]
                    line = line[:-4]
                    fileids.add(line)
        fileids = list(sorted(fileids))
    annos_per_cat = getAnnotations(fileids)
    
    # dst_cls = "storage-tank"
    dst_cls = args["cls"]

    dst_files = {C: [] for C in VOC_CLASSES}
    for cls in VOC_CLASSES:
        if dst_cls is not None and cls != dst_cls:
            continue
        data_file = osp.join(BASE_DIR, f'split{spliter}/seed{seed}/box_{args["shot"]}shot_{cls}_train.txt')
        print("DataFile: ", data_file)
        with PathManager.open(data_file) as f:
            dstfiles = np.loadtxt(f, dtype=str).tolist()
            dstfiles = [x[x.rfind("/")+1:-4] for x in dstfiles] # filenames
            dst_files[cls].extend(dstfiles)

    if args["no_compare"]:
        dst_files = None

    print("dst_file: ", sum(len(x) for x in dst_files), "\n\n")
    total_imgs = 0
    for cls, fileinfos in annos_per_cat.items():
        if dst_cls is not None and cls != dst_cls:
            continue
        count_info = generate_cls_states(cls, fileinfos, dst_files)
        total_imgs += len(count_info)
        print(cls, sum((x[1] for x in count_info)))
    print(total_imgs)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser("stats_dior")
    parser.add_argument("--cls", type=str, default=None)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", type=int, default=20)
    args = vars(parser.parse_args())
    generate_stats(args)
