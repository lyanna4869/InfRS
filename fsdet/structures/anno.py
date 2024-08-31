import dataclasses 
from fsdet.structures import BoxMode
from torch import Tensor
from fsdet.structures.instances import Instances
import numpy as np

@dataclasses.dataclass
class DatasetMetaInfo:
    # dataset name 如 nwpu
    name: str
    # dataset 中包含的类别名称，如 [airplane, ship, ...]
    classnames: "list[str]"
    # 数据集的路径
    dirname: str
    # 数据集类别：trainval, train, val, test (注意，之前的变量名可能命名为 split)
    type: str
    #  split_id: 如 1， 2， 3
    split_id: int = dataclasses.field(default=None)
    # 完整的 meta 名称，如 nwpu_trainval_base1
    full_name: int = dataclasses.field(default=None)
    full_classnames: "list[str]" = dataclasses.field(default=None)

"""
这是一个继承自 `UserDict` 的子类 `VocInstance`，在 `__init__` 方法中调用了其父类 `UserDict` 的 `__init__` 方法。
该子类如果没有重写其他方法，则继承了其父类的所有方法，并且可以用来创建一个可变字典的实例。
该类可以被扩展以添加额外的功能，或者用于处理不同类型的数据。
"""
@dataclasses.dataclass()
class VocInstance():
    """包含instance 的一些必要信息
    """
    # 类别索引号: 0, 1, 2...
    category_id: int
    # 类别名称: "airplane"
    category: str
    # boudning box  x1, y1, x2, y2
    bbox: list
    # 框模式
    bbox_mode: BoxMode
    # object id
    oid: int
    # 该 instance 在整个数据集中的序号
    instance_idx: int
    def get(self, key, dv = None):
        return dv
    def __setitem__(self, key, v):
        if key == "bbox":
            self.bbox = v
        elif key == "bbox_mode":
            self.bbox_mode = v

    def __getitem__(self, key):
        if key == "category_id":
            return self.category_id
        elif key == "bbox":
            return self.bbox
        elif key == "bbox_mode":
            return self.bbox_mode
            
    def pop(self, k, dv = None):
        pass

"""
UserDict: 这是一个 Python 类 `UserDict` 的实现，它继承了 `collections.abc.MutableMapping` 类。
该类实现了一个可变字典，并且在实现中使用了另一个 Python 类 `dict`。 该类通过实现 `__init__`，`__len__`，`__getitem__`，
`__setitem__`，`__delitem__`，`__iter__` 和 `__contains__` 方法，实现了一个可变字典的所有必须实现的方法。
在此基础上，该类还添加了其他方法，例如 `__repr__`，`__or__`，`__ror__`，`__ior__`，`__copy__` 和 `copy`。
这个类的实例可以像普通的字典一样被操作，但是它提供了一些额外的方法，例如 `copy` 和 `fromkeys`。
'copy` 返回字典的一份浅拷贝，而 `fromkeys` 返回一个将键映射到指定值的字典。
"""

@dataclasses.dataclass    
class VocAnnotation():
    """VocAnnotation 表示一个 Voc数据文件对 (.jpg 和 .xml)
    """
    ## 数据对应的 jpeg 文件路径, 如 'datasets/NWPU/training/JPEGImages/001.jpg'，与 FSDET 框架保持兼容
    file_name: str
    ## 数据对应的 xml 文件路径, 如 'datasets/NWPU/training/Annotations/001.xml'
    ## 图片 id: 如 '001'
    image_id: str
    # 图片高度
    height: int = dataclasses.field(default=None)
    # 图片宽度
    width: int = dataclasses.field(default=None)
    # anno name : "001.xml"
    anno_name: str = dataclasses.field(default=None)
    depth: int = dataclasses.field(default=3)
    # 该 Anno 文件中包含的所有 instance，为原始的 标注信息
    annotations: "list[VocInstance]" = dataclasses.field(default=None)
    # 该 instance 是用于网络训练或输出的 对象
    instances: "list[Instances]" = dataclasses.field(default=None)
    # 经过 datasetmapper 读取的 tensor 图片数据
    image: "Tensor" = dataclasses.field(default=None)
    def __post_init__(self):
        pass

    def set_annotations(self, inses: "list[VocInstance]"):
        self.annotations = inses
        # if self.instances is None:
        #     self.instances = inses

    def set_instances(self, inses: "list[Instances]"):
        self.instances = inses
    
    def clone(self):
        return VocAnnotation(**self.__dict__)

    def has_annotation(self):
        return self.annotations is not None and len(self.annotations) > 0
    
    def pop(self, k, dv = None):
        v = getattr(self, k, dv)
        
        return v

    def __getitem__(self, key):
        if key == "annotations":
            return self.annotations
        return getattr(self, key)
    def __contains__(self, key):
        return key in ["file_name", "image_id", "height", "width", 
            "annotations", "instances", "anno_name", "depth", "image",
        ]

    def __setitem__(self, key, v):
        if key in ["file_name", "image_id", "height", "width", 
            "annotations", "instances", "anno_name", "depth", "image",
        ]:
            setattr(self, key, v)
    
    def get(self, key, dv=None):
        if key in ["file_name", "image_id", "height", "width", 
            "annotations", "instances", "anno_name", "depth","image",
        ]:
            return getattr(self, key, dv)
            
    def to_dict(self):
        """bsf.c 为兼容 FSDET 框架
        """
        data = {
            "file_name": self.file_name,
            "image_id": self.image_id,
            "height": self.height,
            "width": self.width,
            "annotations": self.instances,
            "anno_name": self.anno_name,
            "depth": self.depth,
        }
        return data
from typing import TypedDict
    
class FeatureInstance(TypedDict):
    feature: "np.ndarray"
    label: "np.ndarray"
    pred: "np.ndarray"
    file: "str"
    oid: "int"

class FeatureResultDict(TypedDict):
    features: "list[FeatureInstance]"
    version: "float"

def feature_by_category(self: "FeatureResultDict") -> "dict[int, tuple]":
    feat_by_cat = {}
    for feat_ins in self["features"]:
        feat  = feat_ins["feature"]
        label = feat_ins["label"]
        pred  = feat_ins["pred"]
        if label.item() not in feat_by_cat:
            feat_by_cat[label.item()] = []
        feat_by_cat[label.item()].append((feat, pred.item()))
    return feat_by_cat
