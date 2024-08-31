import os.path as osp
from fs.core.data.voc import VocDataset
from fs_dior.core.meta import ALL_CATEGORIES

if __name__ == "__main__":
    CUR_DIR = osp.abspath(osp.dirname(__file__))
    PROJECT_ROOT_PATH = osp.abspath(osp.join(CUR_DIR, "..", "..", ".."))
    DATA_DIR = osp.join(PROJECT_ROOT_PATH, "datasets", "DIOR")
    VocDataset.ALL_CLASSES = ALL_CATEGORIES[2]
    vocdataset = VocDataset(DATA_DIR, "test")
    vocdataset.id_objects()
    # vocdataset.load_from("test")
    print(vocdataset)