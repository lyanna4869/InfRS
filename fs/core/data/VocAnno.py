import xml.etree.ElementTree as ET
from PIL import Image
from fsdet.structures import BoxMode
import os.path as osp

class VocInstance:
    def __init__(self) -> None:
        self.cls = None
        self.id = None
        self.bbox = None
        self.bbox_mode = None
        self.cat_id = 0

class VocImage:
    def __init__(self) -> None:
        self.file_name = "" # full name
        self.image_id = "" # 无 ext
        self.width = 0
        self.height = 0
        self.instances = []

    @staticmethod
    def createFromAnn(anno_file, img_file, fileid):
        tree = ET.parse(anno_file)
        vi = VocImage()
        vi.file_name = img_file
        vi.image_id = fileid
        vi.height = int(tree.findall("./size/height")[0].text)
        vi.width  = int(tree.findall("./size/width")[0].text)
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            vins = VocInstance()
            vins.cls = cls
            vins.bbox = bbox
            vins.bbox_mode = BoxMode.XYXY_ABS
            vi.instances.append(vins)

    def dump(self, anno_file, img_file):
        pass