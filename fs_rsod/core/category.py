__all__ = ["ALL_CATEGORIES", "NOVEL_CATEGORIES", "BASE_CATEGORIES", "_get_fewshot_instances_meta"]

### 这里 classnames 必须严格按照数据集中原有的顺序，因为 nwpu 格式 分类是按照索引，
### 使用 voc 格式可随意
ORIGINAL_CLASS_NAMES = [
    "aircraft", "oiltank", "overpass", "playground", 
]
# NWPU categories
ALL_CATEGORIES = {
    0: ORIGINAL_CLASS_NAMES,
    1: ["oiltank", "overpass", "playground", "aircraft",],
    2: ["aircraft", "overpass", "playground", "oiltank",],
    3: ["aircraft", "oiltank", "playground", "overpass"],
}

NOVEL_CATEGORIES = {
    1: ["aircraft",],
    2: ["oiltank",],
    3: ["overpass"],
}

BASE_CATEGORIES = {
    1: ["oiltank", "overpass", "playground",],
    2: ["aircraft", "overpass", "playground",],
    3: ["aircraft", "oiltank", "playground",],
}

def get_base_by_remove_novel(i):
    base = ORIGINAL_CLASS_NAMES[:]
    novel = NOVEL_CATEGORIES[i]
    for n in novel:
        base.remove(n)
    return base

## 与 s2anet 中现有代码兼容，方便移植辅助类脚本
CLASS_CONFIG = {
    1: {
        "ALL_CLASSES": ALL_CATEGORIES[1],
        "BASE_CLASSES": BASE_CATEGORIES[1],
        "NOVEL_CLASSES": NOVEL_CATEGORIES[1],
    },
    2: {
        "ALL_CLASSES": ALL_CATEGORIES[2],
        "BASE_CLASSES": BASE_CATEGORIES[2],
        "NOVEL_CLASSES": NOVEL_CATEGORIES[2],
    }
}

# 验证 base+novel = all，确保class名称和序号对应
for i in range(1, 4):
    a = ALL_CATEGORIES[i]
    b = BASE_CATEGORIES[i]
    n = NOVEL_CATEGORIES[i]
    # print(set(b + n))
    assert len(set(a)) == len(set(b + n)), f"Err: split {i}, {len(set(a))} != {len(set(b + n))}"
    assert tuple(a[:len(b)]) == tuple(b)
    assert tuple(a[len(b):]) == tuple(n)


def _get_fewshot_instances_meta():
    ret = {
        "thing_classes": ALL_CATEGORIES,
        "novel_classes": NOVEL_CATEGORIES,
        "base_classes": BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "nwpu_fewshot":
        return _get_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
