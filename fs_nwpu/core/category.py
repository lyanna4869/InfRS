__all__ = ["ALL_CATEGORIES", "NOVEL_CATEGORIES", "BASE_CATEGORIES", "_get_nwpu_fewshot_instances_meta"]

### 这里 classnames 必须严格按照数据集中原有的顺序，因为 nwpu 格式 分类是按照索引，
### 使用 voc 格式可随意
ORIGINAL_CLASS_NAMES = [
    'airplane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'vehicle',
]
# NWPU categories
ALL_CATEGORIES = {
    0: ORIGINAL_CLASS_NAMES,
    1: ['ship', 'storage-tank', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'vehicle',
        'airplane', 'baseball-diamond', 'tennis-court',
        ],
    2: ['airplane', "baseball-diamond", "tennis-court", "ground-track-field", "basketball-court", "harbor", "bridge", 
        'ship', 'storage-tank', 'vehicle',
        ],
    3: ["airplane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "bridge", "vehicle",
        'basketball-court', 'ground-track-field', 'harbor', 
        ],
}

NOVEL_CATEGORIES = {
    1: ['airplane', 'baseball-diamond', 'tennis-court'],
    2: ['ship', 'storage-tank', 'vehicle'],
    3: ['basketball-court', 'ground-track-field', 'harbor', ],
}

BASE_CATEGORIES = {
    1: ['ship', 'storage-tank', 'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'vehicle'],
    2: ['airplane', "baseball-diamond", "tennis-court", "ground-track-field", "basketball-court", "harbor", "bridge", ],
    3: ["airplane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "bridge", "vehicle",],
}

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
CLASS_COLOR = {
    "airplane":  (54,  67,  244), 'ship':  ( 99,  30, 233), 
    'storage-tank':  (176,  39, 156), 'baseball-diamond': (183,  58, 103), 
    'basketball-court':  (181,  81,  63), 'ground-track-field':  (243, 150,  33),  
    'harbor':  (212, 188,   0), 'bridge':  (136, 150,   0), 
    'vehicle':  ( 80, 175,  76), 'ST':  ( 74, 195, 139), 
    'SBF': ( 57, 220, 205), 'RA':  ( 59, 235, 255), 
    'HA':  (  0, 152, 255), 'SP':  ( 34,  87, 255), 
    'HC':  ( 72,  85, 121)
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


def _get_nwpu_fewshot_instances_meta():
    ret = {
        "thing_classes": ALL_CATEGORIES,
        "novel_classes": NOVEL_CATEGORIES,
        "base_classes": BASE_CATEGORIES,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "nwpu_fewshot":
        return _get_nwpu_fewshot_instances_meta()
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
