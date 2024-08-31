__version__ = "1.4"
"""现在使用 fs.core.data.nwpu2voc
"""

import os, sys
import os.path as osp
from PIL import Image, ImageFilter
import numpy as np
import cv2
from tqdm import tqdm

import copy
from fs.utils import set_random_seed
from fs_nwpu.utils.annotation import *


def split_from_directory(meta_file, dst_dir):
    with open(osp.join(IMAGE_SETS_LOC, meta_file), "r") as f:
        filenames = map(lambda x: x.strip(), f.readlines())
        filenames = list(filter(lambda x: len(x) > 0, filenames))
    # print(filenames)
    dst_meta_dir = osp.join(dst_dir, IMAGE_SETS_LOC)
    os.makedirs(dst_meta_dir, exist_ok=True)
    
    nwpu_annotations = parse_nwpu_annotations()
    
    imgSpliter = NwpuImageSpliter(dst_dir, nwpu_annotations, subsize=1024, overlap=0.1, padding=True)
    names = []
    for origin_filename in tqdm(filenames):
        subnames = imgSpliter.split(origin_filename)
        names.extend(subnames)

    dst_file = osp.join(dst_meta_dir, meta_file)
    with open(dst_file, "w") as f:
        f.write("\n".join(names))
        f.write("\n")

import argparse 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, required=True, help="Range of seeds")
    parser.add_argument("--shots", type=int, required=True, help="Range of shots (absolute/percent)")
    parser.add_argument("--random-seed", type=int, default=20230326, help="Range of shots (absolute/percent)")
    return parser

def split_from_directory_with_objects(meta_file, args: dict):
    with open(osp.join(IMAGE_SETS_LOC, meta_file), "r") as f:
        filenames = map(lambda x: x.strip(), f.readlines())
        filenames = list(filter(lambda x: len(x) > 0, filenames))
    # print(filenames)
    dst_dir = f"train_part/seed{args['seeds']}_shot{args['shots']}"
    dst_meta_dir = osp.join(dst_dir, IMAGE_SETS_LOC)
    os.makedirs(dst_meta_dir, exist_ok=True)

    objid_list_file = f"train_shot{args['shots']}_seed{args['seeds']}.txt"
    nwpu_annotations = parse_nwpu_annotations(filenames)
    object_sets_loc = "ImageSets/Split"
    with open(osp.join(object_sets_loc, objid_list_file), "r") as f:
        objects_id = []
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("#"):
                continue
            data = line.split("-")
            objects_id.append(int(data[0]))

    dropped_annotations = {} # key is image_id, value is array of objects
    for k in list(nwpu_annotations.keys()):
        arr = nwpu_annotations[k]
        dropped_annotations[k] = list(filter(lambda x: x['oid'] not in objects_id, arr))
        
        arr = list(filter(lambda x: x['oid'] in objects_id, arr))
        if len(arr) > 0:
            nwpu_annotations[k] = arr
        else:
            nwpu_annotations.pop(k)
    
    # print(nwpu_annotations)
    # return
    imgSpliter = NwpuImageSpliter(dst_dir, nwpu_annotations)
    names = []
    for origin_filename in tqdm(filenames, desc="split image into patchs"):
        if origin_filename not in nwpu_annotations:
            continue
        subnames = imgSpliter.split(origin_filename)
        if subnames is not None:
            names.extend(subnames)
    # mask unused objects
    for origin_filename in tqdm(filenames, desc="split image into patchs"):
        drop_img_inses = imgSpliter.translate_objects(origin_filename, dropped_annotations)
        keep_img_inses = imgSpliter.translate_objects(origin_filename, nwpu_annotations)
        for sub_img_name, drop_objects in drop_img_inses:
            if sub_img_name not in names:
                continue
            im_loc = osp.join(dst_dir, "JPEGImages", f"{sub_img_name}.jpg")
            if not osp.exists(im_loc):
                # no objects
                continue
            keep_objects = list(filter(lambda x: x[0] == sub_img_name, keep_img_inses))[0][1]
            keep_objects_poly = [a['bbox'] for a in keep_objects]
            drop_objects_poly = [a['bbox'] for a in drop_objects]
            im = Image.open(im_loc)
            for drop_obj in drop_objects_poly:
                max_iou = 0
                for keep_obj in keep_objects_poly:
                    max_iou = max(max_iou, max(hbb_iou(keep_obj, drop_obj, )))
                if max_iou > 0.15:
                    continue
                drop_obj_region = im.crop(drop_obj)
                blur_radius = 20
                blurred_region = drop_obj_region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                im.paste(blurred_region, drop_obj)
            im.save(im_loc)

    dst_file = osp.join(dst_meta_dir, "trainval.txt")
    with open(dst_file, "w") as f:
        f.write("\n".join(names))
        f.write("\n")

def novel_split_by_shots(args=None):
    parser = parse_args()
    args = vars(parser.parse_args(args))
    set_random_seed(args['random_seed'])
    split_from_directory_with_objects("trainval.txt", args)

def select_class_instances():
    """从不同的类别中选择指定数量的 instance
    """
    dst_root_path = "sim_class"
    nwpu_annotations = parse_nwpu_annotations()
    for cls in NWPU_CLASSES:
        os.makedirs(osp.join(dst_root_path, cls), exist_ok=True)
    for image_id, annotations in nwpu_annotations.items():
        # print(image_id, len(annotations))
        file_loc = osp.join(POSITIVE_IMG_DIRECTORY, f"{image_id}.jpg")
        im = Image.open(file_loc)
        for anno in annotations:
            root_path = osp.join(dst_root_path, anno['name'])
            
            bbox = anno['bbox'] 
            oid = anno['oid']
            cls_im = im.crop(bbox)
            save_im_name = f"{image_id}_{oid}.jpg"
            save_im_name = osp.join(root_path, save_im_name)
            cls_im.save(save_im_name)
            print(save_im_name)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = ['--seeds', "2", "--shots", "10"]
    novel_split_by_shots(args)