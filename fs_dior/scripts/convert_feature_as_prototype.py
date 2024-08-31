import os, os.path as osp
import torch
from fsdet.structures.anno import *
def convert(src_dir: "str"):
    src = osp.join(src_dir, "results.feature")
    dst = osp.join(src_dir, "prototype.feature")
    prototype_dict: "FeatureResultDict" = torch.load(src)  # [num_classes+1, 1024]
    
    feat_by_cat = feature_by_category(prototype_dict)
    prototype_by_cat = {}
    for cat_id, feat_pred_list in feat_by_cat.items():
        prototype_features = []
        for feat, pred in feat_pred_list:
            prototype_features.append(feat)
        prototype_features_np = np.concatenate(prototype_features)
        prototype = np.mean(prototype_features_np, axis=0, keepdims=True)
        prototype_by_cat[cat_id] = prototype
        # if label.item != pred.item:
        #     print("")
    ## bsf.todo 将所有 object 的 feature 转为 list[torc.Tensor]
    ## convert dict into list
    prototype_list = []
    for idx in range(len(prototype_by_cat)):
        prototype_list.append(prototype_by_cat[idx])
    prototype_np = np.concatenate(prototype_list)
    prototype_te = torch.as_tensor(prototype_np)
    torch.save(prototype_te, dst)
    print("Done", dst)
import argparse    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=1)
    args = vars(parser.parse_args())

    split_id = args['split']
    convert(f"checkpoints/dior/prototype/split{split_id}")