import torch
import torch.nn.functional as F
import numpy as np
import os, os.path as osp, sys
import shutil as sh
from fsdet.structures.voc import VocAnnSet, VocObject, VocFile, AnnoDb
from fsdet.structures.anno import *
from fs.meta_path import *
# data2 = torch.load("work_dirs/tsne/dota/box_feature_bg_base_1.pth")
from fs_dior.core.category import CLASS_CONFIG

from fs.utils.select_shot import get_cluster_by_feat, load_class_feat_id, construct_parser, construct_meta_parser, generate_meta
from fs.utils import set_random_seed
from collections import Counter

def test():
    set_random_seed(2309142004)
    fileloc = "checkpoints/dior/prototype/select_shot/results.feature"
    class_feat, class_id = load_class_feat_id(fileloc, CLASS_CONFIG)
    for cat, feats in class_feat.items():
        feats = torch.as_tensor(np.stack(feats))

        min_sim_list = []
        for f in feats:
            sim = F.cosine_similarity(f, feats, -1)
            min_sim = torch.min(sim)
            min_sim_list.append(min_sim)
        min_sim_tensor = torch.stack(min_sim_list)
        min_sim = torch.min(min_sim_tensor)
        print(cat, min_sim)

BASE_DIR = osp.join(DATASETS_PATH, "DIOR")
DATA_DIR = osp.join(BASE_DIR, 'training')
def main(args=None):
    """将指定 id 的 object crop 出来，方便查看
    """
    parser = construct_parser()
    args = vars(parser.parse_args(args))
    global CLUSTER
    CLUSTER = args['cluster']
    if args['np_seed'] is None:
        args['np_seed'] = CLUSTER
    set_random_seed(args['np_seed'])
    

    fileloc = "checkpoints/dior/prototype/select_shot/results.feature"
    class_feat, class_id = load_class_feat_id(fileloc)
    dataset = VocAnnSet(DATA_DIR, "select_shot")
    dataset.load_from_root("trainval")
    dataset.print()

    dst_dir = osp.join(BASE_DIR, f"compare_imageset_{CLUSTER}/")
    os.makedirs(dst_dir, exist_ok=True)

    ## bsf.c 将 object 保存至图片，方便查看
    unique_id: "list[str]" = []
    unique_obj: "list[VocObject]" = []
    instance_counter = Counter()
    ### bfp: add inherit here
    dst_cluster = CLUSTER
    if args['inherit'] is not None:
        inh_id_file = osp.join(args['inherit'], "ids.txt")
        with open(inh_id_file) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                unique_id.append(line)
        inh_dataset = VocAnnSet(args['inherit'], "inherit_dataset")
        inh_dataset.load_from_root("trainval")
        # print(inh_dataset)
        inh_objects = list(inh_dataset.unique_objects.values())
        for obj in inh_objects:
            instance_counter[obj.cls] += 1
        unique_obj.extend(inh_objects)
        # dst_cluster += 1
    skiped_instance_counter = Counter()

    class_cluster_idxes = get_cluster_by_feat(class_feat, class_id, dataset, cluster=dst_cluster)

    for cat, cluster_idxes in class_cluster_idxes.items():
        dst_cls_dir = osp.join(dst_dir, "images", cat)
        try:
            sh.rmtree(dst_cls_dir)
        except: pass
        os.makedirs(dst_cls_dir, exist_ok=True)
        for id in cluster_idxes:
            obj: "VocObject" = dataset.unique_objects[id]
            obj.save(dst_cls_dir)
            if instance_counter[obj.cls] < CLUSTER:
                uni_obj_id = obj.get_unique_loc()
                if uni_obj_id not in unique_id:
                    instance_counter[obj.cls] += 1
                    unique_id.append(uni_obj_id)
                    unique_obj.append(obj)
            else:
                skiped_instance_counter[obj.cls] += 1
    if sum(v for v in skiped_instance_counter.values()):
        print("****Skip Instance because of INHERIT: ", skiped_instance_counter)
        print(len(unique_id), unique_id)
        check_obj_counter = {}
        for obj in unique_obj:
            if obj.cls not in check_obj_counter:
                check_obj_counter[obj.cls] = [obj.id]
            else:
                check_obj_counter[obj.cls].append(obj.id)
        for k, v in check_obj_counter.items():
            assert len(set(v)) >= CLUSTER, f"{k} => {v}"
    ## bsf.c 创建新的 AnnSet 用来保存文件
    mds = VocAnnSet("", "merged")
    for uobj in unique_obj:
        dfile: "VocFile" = uobj.parent
        if dfile.fileid not in mds.unique_annotations:
            nfile: "VocFile" = VocFile(dfile.fileid, dataset=mds,)
            prev_file: "VocFile" = dataset.unique_annfile[dfile.fileid]
            nfile.width = prev_file.width
            nfile.height = prev_file.height
            mds.add_annotations(nfile, dfile.fileid)
        else:
            nfile = mds.unique_annotations[dfile.fileid]

        nfile.add_object(uobj)
    for _, ann in mds.unique_annotations.items():
        ann.sort_object()    
    dst_label_dir = osp.join(dst_dir, "Annotations")
    try:
        sh.rmtree(dst_label_dir)
    except: pass
    mds.saveAllAnnos(dst_label_dir)

    ## 写入 id 文件
    with open(osp.join(dst_dir, "ids.txt"), "w") as f:
        f.write("\n".join(unique_id))
    generate_meta(dst_dir, "Annotations")
    

def generate_meta_main():
    sargs = sys.argv[1:]
    parser = construct_meta_parser()
    # set_random_seed
    args = vars(parser.parse_args(sargs))
    generate_meta(args["dir"], args['label'])

if __name__ == "__main__":    
    # test()
    args = sys.argv[1:]
    if len(args) == 0:
        args = ["--cluster", "3", "--inherit", "datasets/DIOR/compare_imageset_1"]
        args = ["--cluster", "200",]
    main(args)