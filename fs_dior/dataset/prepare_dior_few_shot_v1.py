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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20],
                        help="Range of seeds")
    parser.add_argument("--shots", type=str, default="10",
                        help="Range of shots")
    parser.add_argument("--base", type=str, default="DIOR", help="basic directory")
    parser.add_argument("--index", type=int, default=2, help="split id")
    parser.add_argument("--merge-label", action="store_true", help="Merge class shots file in one .txt file")
    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = osp.join(CUR_DIR, args.base)
    return args
INSTANCE_PER_IMG = {}
# INSTANCE_PER_IMG = {c: 0 for c in VOC_CLASSES}
INSTANCE_PER_IMG.update({nc : 5 for nc in NOVEL_CATEGORIES[split]})
def rank(fileinfo):
    return sum([v for v in fileinfo[1].values()])

INS_COUNT_RANGE = range(1, 15)


def getAnnotations(fileids, sortit = False):
    data_per_cat = {c: [] for c in VOC_CLASSES}  ## images
    for fileid in fileids:
        anno_file = os.path.join(BASE_DIR, "Annotations", fileid + ".xml")
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
                data_per_cat[cls].append( (fileid, ins_count) )
        except Exception as e:
            print(anno_file, cls, e)
    # print(data_per_cat["storage-tank"])
    for cls, fileinfos in data_per_cat.items():
        
        if sortit:
            if cls in INSTANCE_PER_IMG:
                fileinfos = sorted(fileinfos, key=lambda x: sum([x[1][cls]]) ) # 只考虑当前类
            else:
                fileinfos = sorted(fileinfos, key=rank)
        else:
            size = min(len(fileinfos), 50)
            fileinfos = random.sample(fileinfos, size)
        # newcats = fileinfos
        newcats = []
        min_value = 100 # random.randint(3, 5)
        # print(cls, min_value)
        INS_COUNT = {r: max(min_value, 6 - r) for r in INS_COUNT_RANGE}  # key: instance in image
        for ins in fileinfos:
            
            cur_ins_count = ins[1][cls]
            if cur_ins_count not in INS_COUNT_RANGE: 
                continue
            # print(cur_ins_count)
            if INS_COUNT[cur_ins_count] == 0: 
                continue
            INS_COUNT[cur_ins_count] = INS_COUNT[cur_ins_count] - 1

            newcats.append(ins)
        # print(cls, INS_COUNT)

        data_per_cat[cls] = newcats
    # raise ValueError("Test")
    return data_per_cat

RandomChoice = False
DatabaseId = 1
def generateSeedFiles(data_per_cat, shots, i):
    """
    data_per_cat  {"airplane" : [("322", 5) , ...] }
    """
    result = {cls: {} for cls in VOC_CLASSES}
    random.seed(i)
    for c, fileinfos in data_per_cat.items():
        last_name_data = []
        for j, shot in enumerate(shots):
            c_data = []
            nc_data = []
            if j == 0:
                diff_shot = shots[0]
            else:
                nc_data.extend(last_name_data)
                # print(c, c_data)
                diff_shot = shots[j] - shots[j-1] 

            if len(fileinfos) > diff_shot + 4 and RandomChoice:
                shots_c_files = random.sample(fileinfos, diff_shot + 4)
                shots_c_files = sorted(shots_c_files, key=rank)
            else:
                shots_c_files = sorted(fileinfos, key=rank)
            # print(c, shots_c_files, diff_shot, "\n")
            INSTANCE_SWITCH = {c: 0 for c in VOC_CLASSES}
            for s_fileid, ins_count in shots_c_files:
                name = os.path.join(BASE_DIR, f"JPEGImages/{s_fileid}.jpg")
                if name  in nc_data:
                    continue
                INSTANCE_SWITCH[c] += min(ins_count[c], 3)
                nc_data.append((name, ins_count[c]))
                if INSTANCE_SWITCH[c] >= diff_shot:
                    break
            # print("Shot ", shots_c_files, len(c_data), INSTANCE_SWITCH)
            result[c][shot] = nc_data
            last_name_data = copy.deepcopy(nc_data)
    save_path = osp.join(BASE_DIR, f'split{DatabaseId}/seed{i}')
    if False:
        for c, res in result.items():
            print(c, res, "\n")
        return
    # print("Save path: ", save_path)
    os.makedirs(save_path, exist_ok=True)
    if not args.merge_label:
        for c, res in result.items():
            for shot, data in res.items():
                filename = f'box_{shot}shot_{c}_train.txt'
                loc = osp.join(save_path, filename)
                # print(loc, filename)
                names = [x[0] for x in data]
                with open(loc, 'w') as fp:
                    fp.write('\n'.join(names)+'\n')
    else:
        for c in result.keys():
            for shot in result[c].keys():
                filename = f'label_{shot}shot_train.txt'
                loc = osp.join(save_path, filename)

def generate_seeds(args):
    global DatabaseId
    DatabaseId = args.index
    data_file = osp.join(BASE_DIR, 'ImageSets/Main/trainval.txt')
    

    with PathManager.open(data_file) as f:
        fileids = np.loadtxt(f, dtype=str).tolist()
    sort_dataset = getAnnotations(fileids, True)
    dataset = getAnnotations(fileids, False)
    if False:
        for cls, fileinfos in dataset.items():
            s = sum([fi[1][cls] for fi in fileinfos])
            print(cls, "---", s, len(fileinfos)
                , "\n", fileinfos
                )
        return
    shots = args.shots.split(",")
    shots = list(map(int, shots)) # [1, 2, 3, 5, 10]
    save_path = osp.join(BASE_DIR, f'split{DatabaseId}/seedx')
    print("Shots", shots, "Split", DatabaseId, save_path)

    
    generateSeedFiles(sort_dataset, shots, 0)
    start = max(1, args.seeds[0])
    for i in range(start, args.seeds[1]):
        generateSeedFiles(dataset, shots, i)


if __name__ == '__main__':
    args = parse_args()
    generate_seeds(args)
