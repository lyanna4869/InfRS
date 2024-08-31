
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
from PIL import Image

import os.path as osp
import os


class AnnObject():
    # xy : x1 y1 x2 y2 对角顶点
    # oxy: x1 y1 x2 y2 x3 y3 x4 y4
    # xywha: 
    def __init__(self) -> None:
        self.cls = None  # class name: "airplane"
        self.xy = None   # (x1, y1, x2, y2)  or (x1, y1, x2, y2, x3, y3, x4, y4)
        self.im = None   # cropeed image 
        self.difficulty = 0

    def set(self, cls, xy):
        self.cls = cls
        self.xy = xy
        self.hbb_width  = xy[2] - xy[0]
        self.hbb_height = xy[3] - xy[1]

    def __getitem__(self, k):
        return self.xy[k]
    def __gt__(self, other):
        return self.id > other.id
    def __eq__(self, other):
        return self.id == other.id
    def __hash__(self) -> int:
        return self.id
import numpy as np
         
class VocObject(AnnObject):
    """每一个 object 代表一个 instance
    """
    def __init__(self, parent, id) -> None:
        super().__init__()
        assert id is not None
        self.id = id
        self.parent = parent
        
    def crop(self):
        if self.im is not None:
            return self.im
        im = Image.open(self.parent.img_file)
        im = im.crop(self.xy)
        self.im = im
        return im
    def get_img_name(self):
        w = self.xy[2] - self.xy[0]
        h = self.xy[3] - self.xy[1]
        return f"{self.cls}_{self.parent.fileid}_{self.id}_{w}_{h}_{self.xy[0]}.jpg"
    
    def crop_without_mask(self, margin = 0):
        if self.im is not None:
            return self.im
        pim = self.parent.im
        m = self.xy
        if margin == 0:
            self.im = pim.crop(m)
        else:
            self.im = pim.crop([m[0] - margin, m[1] - margin, m[2] + margin, m[3] + margin])
        return self.im

    def save(self, dst_dir):
        self.crop()
        self.im.save(osp.join(dst_dir, self.get_img_name()))

    def calc_similarity(self, other):
        # 0 ~ 1
        im = self.crop()
        im2 = other.crop()
        w1 = im.size[0]
        h1 = im.size[1]

        w2 = im2.size[0]
        h2 = im2.size[1]
        if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0: 
            return 1
        if h1 > 4 * w1: # im 1 是 竖着的
            if w2 > 4 * h2:  # im2 是横着的
                
                im2 = im2.rotate(90)
                h2, w2 = w2, h2
        elif w1 > 4 * h1:
            if h2 > 4 * w2:  

                im2 = im2.rotate(90)
                h2, w2 = w2, h2

        if w1 > 10 * w2 and h1 > 10 * w1:
            return 0.2
        w = max(w1, w2)
        h = h1 if w == w1 else h2
        s = classify_hist_with_split(im, im2, (w, h))
        return s
    @property
    def oxy(self):
        return [self.xy[0], self.xy[1], self.xy[0], self.xy[3], 
                self.xy[2], self.xy[3], self.xy[2], self.xy[1]]
    def translate(self, x, y):
        # bbox 左上角平移到 (x, y)
        dx = self.xy[0] - x
        dy = self.xy[1] - y
        self.xy[0] = x
        self.xy[1] = y
        self.xy[2] -= dx
        self.xy[3] -= dy
    def trytranslate(self, x, y):
        # translate into (x, y)
        xy = self.xy
        dx = xy[0] - x
        dy = xy[1] - y
        return (xy[2] - dx, xy[3] - dy)    
    def resize(self, scale_ratio):
        """中心位置不变
        scale_ratio 0.5 - 2
        """
        # if self._resized:
        #     return
        # self._resized = True
        assert self.im
        # 先以 obj 中心为原点
        # ox, oy = self.xy[:2]
        # if self.id == 3707: 
        #     print("Found")
        rw = self.hbb_width // 2 # bbox w h  
        rh = self.hbb_height // 2
        self.translate(-rw, -rh)

        nw = (self.xy[2] - self.xy[0]) * scale_ratio # new bbox w h
        nh = (self.xy[3] - self.xy[1]) * scale_ratio
        self.im = self.im.resize((int(nw), int(nh)))
        size = self.im.size
        left = size[0] // 2
        top = size[1] // 2
        right = size[0] - left
        bottom = size[1] - top

        # nw2 = nw / 2; nh2 = nh / 2
        # self.oxy *= scale_ratio # 直接 乘会导致精度丢失
        self.xy = np.asarray([-left, -top, right, bottom])
    def __repr__(self) -> str:
        return f'<Instance: {self.cls} {self.id} ({self.parent.fileid})>'

class VocAnnotation():
    """每一个 Annotation 代表一个 文件
    """
    def __init__(self, fileid, dataset, **kwargs):
        # fileid: "00001"
        self.fileid = fileid
        self._all_object = []
        self.removed_count = 0
        self.dataset = dataset
        self.object_per_cls = None
        self.clsset = set()
        self.img_file = None # set by dataset
        self.anno_file = None
        self._im = None

    def add_object(self, vocobject):
        if vocobject not in self._all_object:
            self._all_object.append(vocobject)
    
 
    def sort_object(self):
        """根据  _all_object 更新其他内容
        """
        self.object_per_cls = {}
        self.clsset = set()
        for obj in self._all_object:
            cls = obj.cls
            if cls not in self.object_per_cls:
                self.object_per_cls[cls] = []
            self.object_per_cls[cls].append(obj)
            self.clsset.add(cls)

    def _update_objects(self):
        """根据 object_per_cls 更新其他内容
        """
        objects = []
        for _, nobjects in self.object_per_cls.items():
            objects.extend(nobjects)
        self._all_object = sorted(objects, key= lambda x: x.id)
        self.clsset = set(self.object_per_cls.keys())

    def object_count(self, cls):
        if cls in self.object_per_cls:
            return len(self.object_per_cls[cls])
        return 0


    @property
    def all_objects(self):
        return self._all_object
    @property
    def im(self):
        if self._im is None:
            self._im = Image.open(self.img_file)
        return self._im

    def __len__(self):
        return len(self.clsset)

    def remove_by_ids(self, ids):
        # 移除 ids 中含有 object id 
        object_per_cls = {}
        for cls, objects in self.object_per_cls.items():
            nobjs = []
            for object in objects:
                if object.id in ids:
                    continue
                nobjs.append(object)
            if len(nobjs) > 0:
                object_per_cls[cls] = nobjs
        count = {}
        for cls, object in self.object_per_cls.items():
            if cls in object_per_cls:
                count[cls] = len(object) - len(object_per_cls[cls])
            else:
                count[cls] = len(object)
        self.object_per_cls = object_per_cls
        self._update_objects()
        return count

    def reserve_by_ids(self, ids):
        # 保留 ids 中含有 object id 
        object_per_cls = {}
        for cls, objects in self.object_per_cls.items():
            nobjs = []
            for object in objects:
                if object.id in ids:
                    nobjs.append(object)
            if len(nobjs) > 0:
                object_per_cls[cls] = nobjs
        count = {}
        for cls, object in self.object_per_cls.items():
            if cls in object_per_cls:
                count[cls] = len(object) - len(object_per_cls[cls])
            else:
                count[cls] = len(object)
        self.object_per_cls = object_per_cls
        self._update_objects()
        
        return count

    def print_dist(self):
        print(self.object_per_cls)

    def __repr__(self) -> str:
        return f'<{self.fileid} cls: {len(self.clsset)} - obj: {len(self._all_object)}>'

    def __gt__(self, other):
        return int(self.fileid) > int(other.fileid)

import tqdm
class VocDataset():
    Datasets = {}
    ALL_CLASSES = None

    # 加载时必须完全加载数据集，保证 id 一致
    def __init__(self, root_path: str, unique_name: str) -> None:
        self._objects_id = 0
        assert unique_name not in VocDataset.Datasets
        VocDataset.Datasets[unique_name] = self
        self.name = unique_name
        self.root_path = root_path
        self.unique_objects = {} # id 和 object 对应
        self.unique_annotations = {} # file id 和 annotation 对应
        ## 给外界使用，可随意更改 这里仅初始化
        self.annos_per_cat = {c: [] for c in VocDataset.ALL_CLASSES}  ## images

    def class_count(self):
        """以 class 分开，统计 obj 数量
        """
        obj_count = {cls: 0 for cls in self.ALL_CLASSES}
        for fileid, anno in self.unique_annotations.items():
            for cls in self.ALL_CLASSES:
                obj_count[cls] += anno.object_count(cls)
        return obj_count

    def obj_count_by_class(self, objcls):
        obj_count = 0
        for _, anno in self.unique_annotations.items():
            obj_count += anno.object_count(objcls)
        return obj_count

    def load_from(self, dtype, ann_dir = "Annotations", img_dir="JPEGImages"):
        """dtype : train  val  trainval  test
        """
        fileids = []
        with open(osp.join(self.root_path, f"ImageSets/Main/{dtype}.txt")) as f:
            for line in f:
                fileids.append(line.strip())
        for fileid in fileids:
            anno = self.parse_xml(fileid, osp.join(self.root_path, ann_dir, f"{fileid}.xml"))
            img_file = osp.join(self.root_path, img_dir, f"{fileid}.jpg")
            anno.img_file = img_file
            anno.sort_object()
            try:    
                for cls in anno.clsset:
                    self.annos_per_cat[cls].append(anno)
            except Exception as e:
                print(anno.fileid, e)
                break

    def load_from_directory(self, ann_dir = "Annotations", img_dir="JPEGImages", include_name=None, skip_cls=None):
        """dtype : train  val  trainval  test
        """
        root = osp.join(self.root_path, ann_dir)
        if not osp.exists(root):
            return
        for xmlfile in sorted(os.listdir(root)):
            if include_name is not None and xmlfile not in include_name:
                continue
            fileid, _ = osp.splitext(xmlfile)
            xmlfile = osp.join(root, xmlfile)
            anno = self.parse_xml(fileid, xmlfile, skip_cls=skip_cls)
            img_file = osp.join(self.root_path, img_dir, f"{fileid}.jpg")
            anno.img_file = img_file
            anno.sort_object()
            # if len(anno.object_per_cls) == 0 or len(anno._all_object) == 0:
            #     self.unique_annotations.pop(fileid)
            #     continue
            try:    
                for cls in anno.clsset:
                    self.annos_per_cat[cls].append(anno)
            except Exception as e:
                print(anno.fileid, e)
                break
    
    def id_objects(self, ann_dir = "Annotations", img_dir="JPEGImages"):
        """dtype : train  val  trainval  test
        """
        ann_dir = osp.join(self.root_path, ann_dir)
        for file in tqdm.tqdm(sorted(os.listdir(ann_dir))):
            fileid, ext = osp.splitext(file)
            anno_file = osp.join(ann_dir, f"{fileid}.xml")
            anno = self.parse_xml(fileid, anno_file, False)
            img_file = osp.join(self.root_path, img_dir, f"{fileid}.jpg")
            anno.img_file = img_file
            anno.sort_object()
            self.save_anno(anno, anno_file)            

    def parse_xml(self, fileid, ann_file, need_oid = True, skip_cls=None):
        """
        fileid: 00001
        ann_file : /some/path/to/xml/file
        """
        if fileid not in self.unique_annotations:
            anno = VocAnnotation(fileid, self)
            self.unique_annotations[fileid] = anno
        else:
            anno = self.unique_annotations[fileid]

        anno.anno_file = ann_file
        tree = ET.parse(ann_file)
        size = tree.find("size")
        anno.width  = int(size.find("width").text)
        anno.height = int(size.find("height").text)
        try:
            anno.depth  = int(size.find("depth").text)
        except:
            anno.depth = 3
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            if need_oid:
                oid = obj.find("oid")
                assert oid is not None
                oid = int(oid.text)
            else:
                self._objects_id += 1
                oid = self._objects_id
            cls = cls.replace(" ", "-")
            if skip_cls is not None and cls in skip_cls:
                continue
            vocobject = VocObject(anno, oid)            
            vocobject.set(cls, [xmin, ymin, xmax, ymax])
            self.unique_objects[vocobject.id] = vocobject
            anno.add_object(vocobject)
        # assert len(object_per_cls) > 0, anno_file
        return anno

    @staticmethod
    def save_anno(anno: VocAnnotation, dst: str):
        """
        dst: abs file location
        """
        if len(anno.object_per_cls) == 0 or len(anno._all_object) == 0:
            # raise ValueError(f"Empty Annos {anno.fileid} {anno}" )
            return

        # dst = osp.join(dst, f"{anno.fileid}.xml")

        root = ET.Element('annotation')
        filename = ET.SubElement(root, "filename")
        filename.text = f"{anno.fileid}.jpg"
        size = ET.SubElement(root, "size")
        width  = ET.SubElement(size, "width");   width.text  = str(anno.width) 
        height = ET.SubElement(size, "height");  height.text = str(anno.height)
        depth  = ET.SubElement(size, "depth");   depth.text  = str(anno.depth)
        
        for obj in anno._all_object:
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, "name")
            name.text = f"{obj.cls}"
            oid = ET.SubElement(object, "oid"); oid.text = f"{obj.id}"
            bndbox = ET.SubElement(object, "bndbox")
            
            xmin = ET.SubElement(bndbox, "xmin"); xmin.text = f"{obj.xy[0]}"
            ymin = ET.SubElement(bndbox, "ymin"); ymin.text = f"{obj.xy[1]}"
            xmax = ET.SubElement(bndbox, "xmax"); xmax.text = f"{obj.xy[2]}"
            ymax = ET.SubElement(bndbox, "ymax"); ymax.text = f"{obj.xy[3]}"

        # ET.dump
        # tree = ET.ElementTree(root)
        # tree.writexml(dst, short_empty_elements=False)
        rough_string = ET.tostring(root, 'utf-8')
        dom = md.parseString(rough_string)
        try:
            with open(dst, 'w', encoding='UTF-8') as fh:
                # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
                # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
                dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
        except Exception as err:
            print(f'错误信息：{err}')

    def __repr__(self):
        return f"<VocDataset {self.name} Anno: {len(self.unique_annotations)} Obj: {len(self.unique_objects)}>"

    def print(self):
        print(f"============ {self.name} Anno: {len(self.unique_annotations)}" )
        print(self.class_count())
        print("============")

    def dumptxt(self, dtype):
        dst_dir = osp.join(self.root_path, "ImageSets/Main")
        os.makedirs(dst_dir, exist_ok=True)
        with open(osp.join(dst_dir, f"{dtype}.txt"), "w") as f:
            for fileid in self.unique_annotations.keys():
                f.write(fileid + "\n")
    
    def add_annotations(self, anno: VocAnnotation, fileid):
        for cls, objects in anno.object_per_cls.items():
            # if cls not in anno.object_per_cls:
            #     anno.object_per_cls[cls] = []    
            # anno.object_per_cls[cls].extend(objects)
            for obj in objects:
                self.unique_objects[obj.id] = obj
        if fileid in self.unique_annotations:
            print("*****", fileid)
            self.unique_annotations[fileid]._all_object.extend(anno._all_object)
            self.unique_annotations[fileid].sort_object()
        else:
            self.unique_annotations[fileid] = anno

        anno.dataset = self
    def get_anno_filenames(self):
        return [f"{fileid}.xml" for fileid in self.unique_annotations.keys()]

    def check_anno_valid(self):
        for fileid, anno in self.unique_annotations.items():
            if len(anno.object_per_cls) == 0 or len(anno._all_object) == 0:
                raise ValueError(f"Empty Annos {fileid} {anno} : {anno.object_per_cls} {len(anno._all_object)}" )


def rank(anno: VocAnnotation):
    return len(anno.all_objects())

def rank_by_cls(cls):
    def rank(anno):
        return (anno.object_count(cls), len(anno.all_objects))
    return rank

def merge_annotations(datasetTo: VocDataset, datasetFrom: VocDataset, shot):
    for fileid, anno in datasetTo.unique_annotations.items():

        f_anno = datasetFrom.unique_annotations[fileid]
        for cls, objects in f_anno.object_per_cls.items():
            if cls not in anno.object_per_cls:
                anno.object_per_cls[cls] = []
            
            remain = shot - datasetTo.obj_count_by_class(cls)
            if remain == 0: continue
            if len(objects) > remain:
                objects = objects[:remain]
            anno.object_per_cls[cls].extend(objects)
            for obj in objects:
                datasetTo.unique_objects[obj.id] = obj
        anno._update_objects()


if __name__ == "__main__":
    CUR_DIR = osp.abspath(osp.dirname(__file__))
    PROJECT_ROOT_PATH = osp.abspath(osp.join(CUR_DIR, "..", "..", ".."))
    DATA_DIR = osp.join(PROJECT_ROOT_PATH, "datasets", "NWPU")
    vocdataset = VocDataset(DATA_DIR, "test")
    vocdataset.id_objects()
    # vocdataset.load_from("test")
    print(vocdataset)