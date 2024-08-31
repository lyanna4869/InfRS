"""
1.0 默认无 seed 字符时使用 seed 0 目录
1.1 提供选项，可将 seed 文件合并到一个label文件钟
"""

__version__ = "2.3"

import argparse
import copy
import multiprocessing
from multiprocessing.connection import wait
import os
import random
import shutil
import threading
from time import sleep
import numpy as np
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager
import os.path as osp
from PIL import Image
import cv2
from tqdm import tqdm
CUR_DIR = osp.abspath(osp.dirname(__file__))
BASE_DIR = osp.join(CUR_DIR, "DIOR")
from fs_dior.core.meta import ALL_CATEGORIES, NOVEL_CATEGORIES as ALL_NOVEL_CAT
split=2
VOC_CLASSES =  ALL_CATEGORIES[split]
NOVEL_CATEGORIES = ALL_NOVEL_CAT[split]
# print(VOC_CLASSES, "\n", NOVEL_CATEGORIES)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 20],
                        help="Range of seeds")
    parser.add_argument("--shots", type=str, default="10",
                        help="Range of shots")
    parser.add_argument("--base", type=str, default="DIOR", help="basic directory")
    parser.add_argument("--index", type=int, default=1, help="split id")
    parser.add_argument("--merge-label", action="store_true", help="Merge class shots file in one .txt file")
    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = osp.join(CUR_DIR, args.base)
    return args
INSTANCE_PER_IMG = {}
# INSTANCE_PER_IMG = {c: 0 for c in VOC_CLASSES}
INSTANCE_PER_IMG.update({nc : 5 for nc in NOVEL_CATEGORIES})
def rank(anno):
    return anno.all_cls_count()

def rank_by_cls(cls):
    def rank(anno):
        return (anno.cls_count(cls), anno.all_cls_count())
    return rank


def calculate(image1, image2):
    # 灰度直方图算法
    # 计算单通道的直方图的相似值
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree
 
 
def classify_hist_with_split(image1, image2, size=(256, 256)):
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)
    # RGB每个通道的直方图相似度
    # 将图像resize后，分离为RGB三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

class VocObject():
    ObjectId = 0
    def __init__(self, parent, id = None) -> None:
        if id is None:
            self.id = VocObject.ObjectId
            VocObject.ObjectId += 1
        else:
            self.id = id
        self.parent = parent
        self.im = None
    def set(self, cls, xy):
        self.cls = cls
        self.xy = xy
    def crop(self):
        if self.im is not None:
            return self.im
        im = Image.open(self.parent.img_file)
        im = im.crop(self.xy)
        self.im = im
        return im
    def save(self):
        self.crop()
        w = self.xy[2] - self.xy[0]
        h = self.xy[3] - self.xy[1]
        self.im.save(osp.join(tmp_dir, f"{self.cls}_{self.parent.fileid}_{self.id}_{w}_{h}_{self.xy[0]}.jpg"))
        
    def calc_similarity(self, other):
        # 0 ~ 1
        im = self.crop()
        im2 = other.crop()
        w1 = im.size[0]
        h1 = im.size[1]

        w2 = im2.size[0]
        h2 = im2.size[1]
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
        # s = random.random()
        # print(s)
        return s

    def get_img(self):
        pass

    def __getitem__(self, k):
        return self.xy[k]

    def __repr__(self) -> str:
        return f'<{self.cls} {self.id} ({self.parent.fileid})>'


class VocAnnotation():
    Objects = {}
    def __init__(self, fileid, anno_dir="Annotations", img_dir="JPEGImages") -> None:
        # fileid 0001
        self.fileid = fileid
        # self.object_per_cls = set()
        self.parse(fileid, anno_dir="Annotations", img_dir="JPEGImages")

        self.removed_count = 0

    def parse(self, fileid, anno_dir, img_dir):
        print(BASE_DIR)
        anno_file = osp.join(BASE_DIR, anno_dir, fileid + ".xml")
        img_file = osp.join(BASE_DIR, img_dir, fileid + ".jpg")
        self.img_file = img_file

        tree = ET.parse(anno_file)
        clses = set()
        object_per_cls = {}
        size = tree.find("size")
        self.width = size.find("width").text
        self.height = size.find("height").text
        self.depth = size.find("depth").text
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            oid = bbox.find("ymax")
            if oid is not None:
                oid = int(oid.text)
            cls = cls.replace(" ", "-")
            w = xmax - xmin
            y = ymax - ymin
            if w < 6 or y < 6: continue
            vocobject = VocObject(self, oid)            
            vocobject.set(cls, (xmin, ymin, xmax, ymax))
            VocAnnotation.Objects[vocobject.id] = vocobject
            if cls not in clses:
                clses.add(cls)
                object_per_cls[cls] = [vocobject]
            else:
                object_per_cls[cls].append(vocobject)
        self.object_per_cls = object_per_cls
        self.clsset = clses

    def cls_count(self, cls):
        if cls in self.object_per_cls:
            return len(self.object_per_cls[cls])
        return 0

    def all_cls_count(self):
        s = 0
        for _, items in self.object_per_cls.items():
            s += len(items) 
        return s

    def all_objects(self):
        objects = []
        for _, items in self.object_per_cls.items():
            objects.extend(items) 
        return objects

    def reserve(self, cls):
        # 优先保留某个 cls
        return
        if cls not in self.object_per_cls:
            return
        
        obj = self.object_per_cls
        # cls_count = self.all_cls_count()
        self.object_per_cls = {cls: obj[cls]}

        for ocls, items in obj.items():
            if ocls == cls: continue
            if ocls in NOVEL_CATEGORIES:
                c = min(6, len(items))
            else:
                c = min(5, len(items))
            self.object_per_cls[ocls] = items[ :c]
        # ncls_count = self.all_cls_count()
        # self.removed_count = cls_count - ncls_count
        self.clsset = set(self.object_per_cls.keys())

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
        self.clsset = set(self.object_per_cls.keys())
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
        self.clsset = set(self.object_per_cls.keys())
        return count

    def save(self):
        """<annotation>
	<filename>00001.jpg</filename>
	<source>
		<database>DIOR</database>
	</source>
	<size>
		<width>800</width>
		<height>800</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>golffield</name>
		<pose>Unspecified</pose>
		<bndbox>
			<xmin>133</xmin>
			<ymin>237</ymin>
			<xmax>684</xmax>
			<ymax>672</ymax>
		</bndbox>
	</object>
    <object>
		<name>storagetank</name>
		<pose>Unspecified</pose>
		<bndbox>
			<xmin>39</xmin>
			<ymin>281</ymin>
			<xmax>87</xmax>
			<ymax>324</ymax>
		</bndbox>
	</object>
</annotation>
        """
        # save as VOC XML
        dst = osp.join(BASE_DIR, f"NovelAnnotations{split}", self.fileid + ".xml")
        root = ET.Element('annotation')
        filename = ET.SubElement(root, "filename")
        filename.text = f"{self.fileid}.jpg"
        filename.tail = "\n"
        size = ET.SubElement(root, "size")
        size.tail = "\n"
        width = ET.SubElement(size, "width");   width.text = self.width; width.tail="\n"
        height = ET.SubElement(size, "height"); height.text = self.height; height.tail="\n"
        depth = ET.SubElement(size, "depth");   depth.text = self.depth; depth.tail="\n"
        for cls, objects in self.object_per_cls.items():
            for obj in objects:
                object = ET.SubElement(root, 'object')
                name = ET.SubElement(object, "name")
                name.tail = "\n"
                name.text = obj.cls
                bndbox = ET.SubElement(object, "bndbox")
                oid = ET.SubElement(bndbox, "oid"); oid.text = f"{obj.id}"; oid.tail = "\n"
                xmin = ET.SubElement(bndbox, "xmin"); xmin.text = f"{obj.xy[0]}"; xmin.tail = "\n"
                ymin = ET.SubElement(bndbox, "ymin"); ymin.text = f"{obj.xy[1]}"; ymin.tail = "\n"
                xmax = ET.SubElement(bndbox, "xmax"); xmax.text = f"{obj.xy[2]}"; xmax.tail = "\n"
                ymax = ET.SubElement(bndbox, "ymax"); ymax.text = f"{obj.xy[3]}"; ymax.tail = "\n"
                object.tail = "\n"
        # data = ET.save(root)
        tree = ET.ElementTree(root)
        tree.write(dst, short_empty_elements=False)

    def print_dist(self):
        print(self.object_per_cls)

    def __repr__(self) -> str:
        return f'<{self.fileid} {len(self.clsset)} - {self.all_cls_count()}>'
    def __gt__(self, other):
        return int(self.fileid) > int(other.fileid)

SPECIAL_INS_COUNT_RANGE = {
    "airport": range(1, 10),
}
BASE_INS_COUNT_RANGE = range(1, 8)

# 10 shot
TEST_RANGE = slice(0, 0)
PARALLEL = 10
ObjSimThreshDict = {
    "airport": (0.083, 0.49),
    "basketballcourt": (0.074, 0.488),
    "bridge": (0.105, 0.45),
    "chimney": (0.22, 0.463),
    "dam": (0.077, 0.475),

    "Expressway-Service-area": (0.1244715, 0.46),
    "Expressway-toll-station": (0.095, 0.455),
    "golffield": (0.19016636, 0.46),
    "groundtrackfield": (0.07, 0.452),
    "harbor": (0.170, 0.46),

    "overpass": (0.1, 0.44),
    "ship": (0.149, 0.42),
    "stadium": (0.14, 0.42),
    "storagetank": (0.142, 0.43),
    "vehicle": (0.145, 0.43),

    "airplane": (0.16, 0.46),
    "baseballfield": (0.061, 0.46),
    "tenniscourt": (0.063, 0.55),
    "trainstation": (0.16, 0.46),
    "windmill": (0.1, 0.47),
    }
class DatasetsByCat():
    ObjSimThresh = (0.3, 0.5)
    # cat file objects
    def __init__(self):
        self.annos_per_cat = {c: [] for c in VOC_CLASSES}  ## images
        self.vocannos = {}
        self.similarity = {c: {} for c in VOC_CLASSES}

    def append(self, fileid):
        anno = VocAnnotation(fileid)
        self.vocannos[fileid] = anno
        try:    
            for cls in anno.clsset:
                self.annos_per_cat[cls].append(anno)
        except Exception as e:
            print(anno.fileid, cls, e)

    def check(self):
        for cls in tqdm(VOC_CLASSES):
            annos = self.annos_per_cat[cls]
            for anno in annos:
                assert cls in anno.clsset

    def reserve_anno(self, sortit):
        # 保留 anno anno 的 instance 数量需要满足要求，第一步过滤
        dpc = {}
        new_anno_set = set()
        INS_COUNT = {}  # key: instance in image
        TOTAL_INS_COUNT = {cls: 450 for cls in VOC_CLASSES}
        TOTAL_INS_COUNT.update({cls: 700 for cls in NOVEL_CATEGORIES})
        default_ins_count = {
            # "airplane": 200,
            "baseballfield": 1300,
            "tenniscourt": 670,
            # "trainstation": 300,
            # "windmill": 600,
        }
        TOTAL_INS_COUNT.update(default_ins_count)
        for cls in VOC_CLASSES:
            for r in SPECIAL_INS_COUNT_RANGE.get(cls, BASE_INS_COUNT_RANGE):
                INS_COUNT[(cls, r)] = default_ins_count.get(cls, 150)
  
        for cls in reversed(VOC_CLASSES):
            annos = self.annos_per_cat[cls]
            # fileinfos (fileid, [objects xy])
            if sortit:
                annos = sorted(annos, key=rank_by_cls(cls)) # 优先考虑当前类
            else:
                size = min(len(annos), 100)
                annos = random.sample(annos, size)
            # newcats = annos
  
            newcats = []
            # 保留 voc anno
            for anno in annos:
                anno.reserve(cls)
            
            INS_COUNT_RANGE = SPECIAL_INS_COUNT_RANGE.get(cls, BASE_INS_COUNT_RANGE)
            # self.print(cls)
            for anno in annos:   
                # for acls in anno.clsset:
                cur_ins_count = anno.cls_count(cls)
                id = (cls, cur_ins_count)
                # print(id)
                if TOTAL_INS_COUNT[cls] <= 0:
                    break
                if cur_ins_count not in INS_COUNT_RANGE: 
                    continue
                # if INS_COUNT[id] == 0: 
                #     continue
                # INS_COUNT[id] = INS_COUNT[id] - 1
                # anno.print_dist()
                for acls, objects in anno.object_per_cls.items():
                    TOTAL_INS_COUNT[acls] -= len(objects)
                newcats.append(anno)
                new_anno_set.add(anno)
            if len(newcats) > 0:
                dpc[cls] = newcats

        self.annos_per_cat = {c: [] for c in VOC_CLASSES}
        for anno in sorted(new_anno_set, key=rank_by_cls(cls)):
            for cls in anno.clsset:
                self.annos_per_cat[cls].append(anno)
        annos = self.unique_annos()
        for anno in annos:
            for cls in VOC_CLASSES:
                if anno.cls_count(anno) > 3:
                    raise ValueError("Unacceptable anno")
        # self.print(hinthist=True)
        # exit()
        
    def print(self, dstcls = None, hinthist=False):
        if dstcls is None:
            annosset = set()
            for _, annos in self.annos_per_cat.items():
                for anno in annos:
                    annosset.add(anno)
            ins = 0
            for anno in annosset:
                    ins += anno.all_cls_count()
            print("Annos: ", len(annosset))
            print("Instance: ", ins)
            for cls in VOC_CLASSES:
                annos = self.annos_per_cat[cls]
                ins_count = [x.cls_count(cls) for x in annos]
                all_ins_count = [x.all_cls_count() for x in annos]
                s = sum(ins_count)
                hint =  ""
                if s < 10 or s > 20: hint = " *"
                print(cls, s, hint) #, sum(all_ins_count))
                if hinthist:
                    hist, bins = np.histogram(ins_count, bins=BASE_INS_COUNT_RANGE)
                    print(hist, bins)
            print("---------------------")
        else:
            annos = self.annos_per_cat[dstcls]
            ins_count = [x.all_cls_count() for x in annos]
            dst_ins_count = [x.cls_count(dstcls) for x in annos]
            hist, bins = np.histogram(dst_ins_count)
            print(dstcls, sum(ins_count), hist, bins)
    ALL_REMOVE_IDS = {}
    INFO = {}
    def _remove_sim_objects(self, annos, cls, context, bar=False):
        
        objects = []
        for anno in annos:
            if cls in anno.clsset:
                objects.extend(anno.object_per_cls[cls])
 
        for obj in objects:
            assert obj.cls == cls
        len_object = len(objects)
        remove_ids = set()
        reserve_ids = set()
        objmap = {}
        sims = []
        thresh = ObjSimThreshDict.get(cls, DatasetsByCat.ObjSimThresh)
        simmap = np.zeros((len_object, len_object))
        # print("Start", cls)
        iter = range(1, len_object) if TEST_RANGE.stop - TEST_RANGE.start > 1 or bar else tqdm(range(1, len_object))
        for i in iter:
            obj1 = objects[i]
            obj1.sim_index = i
            if obj1.id in remove_ids: 
                continue
            objmap[obj1.id] = obj1.id
            for j in range(i+1, len_object):
                obj2 = objects[j]
                sim = obj1.calc_similarity(obj2)[0]
                sims.append(sim)
                simmap[i, j] = sim
        
                if sim > thresh[1]:
                    remove_ids.add(obj2.id)
                    objmap[obj2.id] = obj1.id
                elif sim < thresh[0]:
                    reserve_ids.add(obj1.id)
                    reserve_ids.add(obj2.id)
        unique_reserve_ids = set()
        for rid in sorted(reserve_ids):
            unique_reserve_ids.add(objmap[rid])
        uri = list(sorted(unique_reserve_ids))
        def calc_sim(i, j):
            o1 = VocAnnotation.Objects[uri[i]]
            o2 = VocAnnotation.Objects[uri[j]]
            x = min(o1.sim_index, o2.sim_index)
            y = max(o1.sim_index, o2.sim_index)
            return simmap[x, y]
        sims = []
        lenuri = len(uri)
        nsimmap = np.ones((lenuri, lenuri))
        for i in range(lenuri):
            for j in range(i + 1, lenuri):
                sim = calc_sim(i, j)
                sim = np.round(sim, decimals=8)
                if sim > 2 * thresh[0]:
                    continue
                nsimmap[i, j] = sim
                sims.append(sim)
        sims = sorted(sims)

        hist, bins = np.histogram(sims)
        print("Done", cls, sims[5:8])
        # print(remove_count)
        context.put(
            {
                "class": cls,
                "INFO": f"{hist}\n {np.round(bins, decimals=3)} {len(remove_ids)}/{len(uri)}\n  {sims[5: 10]}",
                "ALL_REMOVE_IDS": uri
            }
        )
        
    def get_objects_by_cat(self, cat):
        annos = self.annos_per_cat[cat]
        objects = []
        for anno in annos:
            objs = anno.object_per_cls[cat]
            objects.extend(objs)
        return objects

    def remove_sim(self):
        dpc = self.annos_per_cat
        ## remove similarity objects
        pool = MyProcessPool(PARALLEL)
        annos = self.unique_annos()
        q = multiprocessing.Queue() # multiprocessing.Queue()

        for cls in reversed(VOC_CLASSES):
            # print("Start", cls)
            dst = VOC_CLASSES[TEST_RANGE]
            if cls not in dst and len(dst) > 0: continue
            t = multiprocessing.Process(target=self._remove_sim_objects, args=(annos, cls, q))
            pool.append(t)
            # t.start()
            # pool.apply_async(self._remove_sim_objects, args=(annos, cls, q))
        # pool.map(self._remove_sim_objects, arguments)
        pool.close()
        pool.start()
        pool.join()

        while not q.empty():
            data = q.get()
            cls = data["class"]
            DatasetsByCat.INFO[cls] = data["INFO"]
            DatasetsByCat.ALL_REMOVE_IDS[cls] = data["ALL_REMOVE_IDS"]
        reserve_ids = set()
        for cls in VOC_CLASSES:
            if cls in DatasetsByCat.INFO:
                print(cls, DatasetsByCat.INFO[cls])
            for ids in DatasetsByCat.ALL_REMOVE_IDS.values():
                for objid in ids:
                    reserve_ids.add(objid)
        remove_count = {cls : 0 for cls in VOC_CLASSES}
        for annocls in VOC_CLASSES:
            nannos = []
            for anno in dpc[annocls]:
                count = anno.reserve_by_ids(reserve_ids)
                for cls, count in count.items():
                    remove_count[cls] += count
                if anno.all_cls_count() > 0:
                    nannos.append(anno)
            dpc[annocls] = nannos

        # self.print()
        self.reassign()
        for anno in self.unique_annos():
            anno.save()
            for obj in anno.all_objects():
                obj.save()
        self.print()

    def unique_annos(self):
        annos = set()
        for clsannos in self.annos_per_cat.values():
            for anno in clsannos:
                annos.add(anno)
        return sorted(annos)
    def reassign(self):
        annos = self.unique_annos()
        self.annos_per_cat = {c: [] for c in VOC_CLASSES}
        for anno in annos:
            for cls in anno.clsset:
                self.annos_per_cat[cls].append(anno)

class MyProcessPool():
    def __init__(self, size = 4) -> None:
        self.size = size
        assert size > 0
        self._processes = []
        self._close = False
        self.idx = 0
        self._running = 0

    def append(self, process):
        if self._close:
            return
        self._processes.append(process)
    def _start(self):
        if self.idx >= len(self._processes):
            return
        p = self._processes[self.idx]
        self.idx += 1
        self._running += 1
        p.start()
        return p
    def start(self):
        waitting = []
        # bar = tqdm(total=len(self._processes))
        while self.idx < len(self._processes):
            for i in range(self._running, self.size):
                if self.idx >= len(self._processes):
                    continue
                p = self._start()
                waitting.append(p)
            
            while len(waitting) == self.size:
                alives = []
                for w in waitting:
                    w.join(1)
                    if w.is_alive():
                        alives.append(w)
                waitting = alives
                self._running = len(waitting)
            sleep(1)
            # print(self._running)

    def close(self):
        self._close = True
    def join(self):
        for p in self._processes:
            p.join()

def getAnnotations(data, sortit = False):
    dataset = DatasetsByCat()
    for fileid in data:
        dataset.append(fileid)
    # print(sum(len(v) for k, v in dataset.annos_per_cat.items()))
    # exit(0)
    dataset.check()
    dataset.reserve_anno(sortit)
    dataset.remove_sim()
    return dataset

RandomChoice = True
DatabaseId = 1
def generateSeedFiles(dataset, shots, i):
    """
    dataset:
    annos_per_cat  {"airplane" : [("322", 5) , ...] }
    """
    result = {cls: {} for cls in VOC_CLASSES}
    random.seed(i)
    for cls, fileinfos in dataset.annos_per_cat.items():
        last_name_data = []
        for j, shot in enumerate(shots):
            nc_data = []
            if j == 0:
                diff_shot = shots[0]
            else:
                nc_data.extend(last_name_data)
                # print(cls, c_data)
                diff_shot = shots[j] - shots[j-1] 

            if len(fileinfos) > diff_shot + 1 and RandomChoice:
                shots_c_files = random.sample(fileinfos, diff_shot + 1)
                shots_c_files = sorted(shots_c_files, key=rank_by_cls(cls))
            else:
                shots_c_files = sorted(fileinfos, key=rank_by_cls(cls))
      
            INSTANCE_SWITCH = 0
            for anno in shots_c_files:
                s_fileid = anno.fileid
                if cls not in anno.clsset: 
                    continue
                objects = anno.object_per_cls[cls]
                if anno in nc_data:
                    continue
                INSTANCE_SWITCH += len(objects)
                nc_data.append(anno)
                if INSTANCE_SWITCH >= diff_shot:
                    break
            # print("Shot ", shots_c_files, len(c_data), INSTANCE_SWITCH)
            result[cls][shot] = nc_data
            last_name_data = copy.deepcopy(nc_data)
    save_path = osp.join(BASE_DIR, f'split{DatabaseId}/seed{i}')
    if False:
        for cls, res in result.items():
            print(cls, res, "\n")
        return
    # print("Save path: ", save_path)
    os.makedirs(save_path, exist_ok=True)
    if not args.merge_label:
        for cls, res in result.items():
            for shot, data in res.items():
                filename = f'box_{shot}shot_{cls}_train.txt'
                loc = osp.join(save_path, filename)
                # print(loc, filename)
                names = [x.img_file for x in data]
                with open(loc, 'w') as fp:
                    fp.write('\n'.join(names))
    else:
        for cls in result.keys():
            for shot in result[cls].keys():
                filename = f'label_{shot}shot_train.txt'
                loc = osp.join(save_path, filename)

def generate_seeds(args):
    global DatabaseId
    DatabaseId = args.index
    data_file = osp.join(BASE_DIR, 'ImageSets/Main/trainval.txt')


    with PathManager.open(data_file) as f:
        fileids = np.loadtxt(f, dtype=str).tolist()
    random.seed(20220327)
    sort_dataset = getAnnotations(sorted(fileids), True)
    # dataset = getAnnotations(fileids, False)
    if False:
        for cls, fileinfos in dataset.items():
            s = sum([fi[1][cls] for fi in fileinfos])
            print(cls, "---", s, len(fileinfos)
                , "\n", fileinfos
                )
        return
    shots = args.shots.split(",")
    shots = list(map(int, shots)) # [1, 2, 3, 5, 10]
    save_path = osp.join(BASE_DIR, f'split{DatabaseId}/seedx')
    print("Shots", shots, "Split", DatabaseId, save_path)

    generateSeedFiles(sort_dataset, shots, 0)
    
    # start = max(1, args.seeds[0])
    # for i in range(start, args.seeds[1]):
    #     generateSeedFiles(dataset, shots, i)


if __name__ == '__main__':
    args = parse_args()
    split = args.index
    anno_dir = osp.join(BASE_DIR, f"NovelAnnotations{split}")
    tmp_dir = osp.join(BASE_DIR, f"tmp{split}")
    if osp.exists(anno_dir):
        shutil.rmtree(anno_dir)
    if osp.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(osp.join(BASE_DIR, "cache"), exist_ok=True)
    generate_seeds(args)
