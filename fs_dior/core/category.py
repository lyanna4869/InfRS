__all__ = ["ALL_CATEGORIES", "NOVEL_CATEGORIES", "BASE_CATEGORIES", "_get_fewshot_instances_meta"]

# PASCAL VOC categories
ALL_CATEGORIES = {
    0: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
        'chimney', "dam", "Expressway-Service-area", "Expressway-toll-station", "golffield", 
        "groundtrackfield", 'harbor', "overpass", "ship",  "stadium", 
        "storagetank", "tenniscourt", "trainstation", "vehicle",   'windmill', 
    ],
    1: ['airplane', 'baseballfield', 'basketballcourt',  'chimney', "dam", 
        "Expressway-Service-area", "Expressway-toll-station", "golffield", 'harbor', "overpass", 
        "ship",  "stadium", "tenniscourt", "trainstation",  'windmill', 
        'airport', 'bridge', 'groundtrackfield', 'vehicle', 'storagetank'
     ],
    2: ['airport', 'basketballcourt',  'bridge', "chimney", 'dam', 
    "Expressway-Service-area", "Expressway-toll-station", "golffield", 'groundtrackfield', 'harbor', 
    "overpass", "ship", "stadium", "storagetank", 'vehicle', 
    "airplane", 'baseballfield', 'tenniscourt', "trainstation", "windmill",
    ],
    3: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
        'chimney', "dam", "Expressway-toll-station", "golffield", "groundtrackfield", 
        "storagetank", "tenniscourt", "trainstation", "vehicle",   'windmill',
        "Expressway-Service-area", 'harbor', "overpass", "ship",  "stadium",
    ],
}

NOVEL_CATEGORIES = {
    1: ['airport', 'bridge', 'groundtrackfield', 'vehicle', 'storagetank'],
    2: ["airplane", 'baseballfield', 'tenniscourt', "trainstation", "windmill",], ### use
    3: ["Expressway-Service-area", 'harbor', "overpass", "ship",  "stadium", ],
}

BASE_CATEGORIES = {
    1: ['airplane', 'baseballfield', 'basketballcourt',  'chimney', "dam", 
        "Expressway-Service-area", "Expressway-toll-station", "golffield", 'harbor', "overpass", 
        "ship",  "stadium", "tenniscourt", "trainstation",  'windmill', 
    ],
    2: ['airport', 'basketballcourt',  'bridge', "chimney", 'dam', 
    "Expressway-Service-area", "Expressway-toll-station", "golffield", 'groundtrackfield', 'harbor', 
    "overpass", "ship", "stadium", "storagetank", 'vehicle', ],
    3: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 
        'chimney', "dam", "Expressway-toll-station", "golffield", "groundtrackfield", 
        "storagetank", "tenniscourt", "trainstation", "vehicle",   'windmill', 
        ],
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

for i in range(1, 4):
    a = ALL_CATEGORIES[i]
    b = BASE_CATEGORIES[i]
    n = NOVEL_CATEGORIES[i]
    assert len(set(a)) == len(set(b + n))
    for x, y in zip(a[:len(b)], b):
        assert x == y
    for x, y in zip(a[len(b):], n):
        assert x == y


def _get_fewshot_instances_meta():
    ret = {
        "thing_classes": ALL_CATEGORIES,
        "novel_classes": NOVEL_CATEGORIES,
        "base_classes": BASE_CATEGORIES,
    }
    return ret

