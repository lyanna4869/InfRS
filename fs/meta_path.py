import os, os.path as osp

CUR_PATH = osp.abspath(osp.dirname(__file__))
FS_PATH = CUR_PATH
PROJECT_ROOT = osp.abspath(osp.join(CUR_PATH, ".."))

ROOT_PATH = osp.abspath(osp.join(CUR_PATH, ".."))
DATASETS_PATH = osp.join(ROOT_PATH, "datasets")

DATASET_ROOT_PATH = osp.join(ROOT_PATH, "datasets")
