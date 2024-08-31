"""
v1.2
"""
import multiprocessing as mp
import sys
import cv2
import random
import numpy as np
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt
try:
    import fs.dota.dota_utils as util
    from fs.core.bbox.transforms_rotated import *

except:
    pass
import os.path as osp
import math
CUR_PATH = osp.abspath(osp.dirname(__file__))
ROOT = osp.abspath(osp.join(CUR_PATH, "..", ".."))
import tqdm


raw = False

if raw:
    LABEL_PATH = osp.join(ROOT, "dota/labelTxt/")
    IMAGE_PATH = osp.join(ROOT, "dota/images/")
    JSON_PATH  = osp.join(ROOT, "dota/train_dota_xywha_polygen.json")
else:
    LABEL_PATH = osp.join(ROOT, "dota/train_1024/train_split/labelTxt/")
    IMAGE_PATH = osp.join(ROOT, "dota/train_1024/train_split/images/")
    JSON_PATH  = osp.join(ROOT, "dota/train_dota_xywha_polygen.json")

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


class Visualizer():
    ImgExt=".png"
    
    def __init__(self, basedir, labelDir="labelTxt", imageDir="images") -> None:
        self.basedir = basedir
        self._imgdir = imageDir
        self._labeldir = labelDir

    @property
    def imgdir(self):
        return osp.join(self.basedir, self._imgdir) + "/"

    @property
    def labeldir(self):
        if self._labeldir.startswith("/"):
            return self._labeldir
        return osp.join(self.basedir, self._labeldir) + "/"

    def vis(self, name): pass


class CocoVisualizer(Visualizer):
    
    def __init__(self, basedir, json_path) -> None:
        super().__init__(basedir)
        self.coco = COCO(json_path)

    def vis(self, num_image):
        coco = self.coco
        catIds = coco.getCatIds()
        # cats = coco.loadCats(catIds)
        # coco.info()
        # print(catIds, coco.catToImgs)
        list_imgIds = coco.getImgIds(catIds=catIds) # 获取含有该给定类别的所有图片的id
        img = coco.loadImgs(list_imgIds[num_image-1])[0]  # 获取满足上述要求，并给定显示第num幅image对应的dict
        print("imageids", img)
        image = io.imread(self.imgdir + img['file_name'])  # 读取图像
        image_name =  img['file_name'] # 读取图像名字
        image_id = img['id'] # 读取图像id
        img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
        img_anns = coco.loadAnns(img_annIds)
        # print(img_annIds)
        for i in range(len(img_annIds)):
            ann = img_anns[i]
            cid = ann["category_id"]
            # x, y, w, h, a = ann['bbox']  # 读取边框
            # x1 = x - w / 2
            # y1 = y - h / 2
            # x2 = x + w / 2
            # y2 = y + h / 2
            # image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            print("box: ", i, ann['bbox']) 
            box = ann['bbox']
            box[-1] = box[-1] / 180 * math.pi
            box = rotated_box_to_poly_single(ann['bbox'])
            poly = np.array(box, dtype=np.int32)
            poly = poly.reshape((-1, 2))
            print(poly)
            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1],(255,0,0), 2, lineType=cv2.LINE_AA)
            image = cv2.line(image, poly[3], poly[0],(255,0,0), 2, lineType=cv2.LINE_AA)
            print(" === \n", poly, cid)
            # break
        # plt.rcParams['figure.figsize'] = (10.0, 10.0)
        # 此处的20.0是由于我的图片是2000*2000，目前还没去研究怎么利用plt自动分辨率。
        plt.imshow(image)
        plt.show()

class DotaVisualizer(Visualizer):
    
    def vis(self, name):
        polygons = util.parse_dota_poly(self.labeldir + name + ".txt")
        image = io.imread(self.imgdir + name + self.ImgExt)  # 读取图像
        # polygons.append({
        #     "poly": np.asarray([(75.0, 626.0), (75.0, 1.0), (1024.0, 1.0), (1024.0, 626.0)], dtype=np.int32)
        # })

        for pi, polyobj in enumerate(polygons):
            poly = polyobj["poly"]
            poly = np.array(poly, dtype=np.int32)
            # poly = poly.reshape((-1, 1, 2))
            print(poly.reshape(-1))
            color = (255, (pi * 30) % 255, 0)
            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1],color, 1, lineType=cv2.LINE_AA)

            image = cv2.line(image, poly[3], poly[0],color, 3, lineType=cv2.LINE_AA)
            # image = cv2.polylines(image, poly, True, (0, 255, 255), 5)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(image)
        try:
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state('zoomed')    #最大化
        except : pass
        plt.show()
    def _output(self, imgfile):
        name, ext = osp.splitext(imgfile)
        # if "1446" not in name: return
        if "hf" in name: return
        if "vf" in name: return
        if "df" in name: return
        if "_unblur"  in name: return
        # if name.startswith("P"): return
        # if name.startswith("vf"): return
        # if name.startswith("hf"): return

        labelloc = self.labeldir + name + ".txt"
        if not osp.exists(labelloc): return

        polygons = util.parse_dota_poly(labelloc)
        if len(polygons) == 0: return
        image = cv2.imread(osp.join(self.imgdir, imgfile))  # 读取图像

        for pi, polyobj in enumerate(polygons):
            color = (0, 0, 0)
            if int(polyobj["difficult"]) == 0: 
                color = (255, (pi * 30) % 255, 0)
            elif int(polyobj["difficult"]) == 1: 
                color = (30, 30, 225)
            elif int(polyobj["difficult"]) == 2:
                color = (0, 0, 225)
            poly = polyobj["poly"]
            poly = np.array(poly, dtype=np.int32)
            # poly = poly.reshape((-1, 1, 2))
            for i in range(3):
                image = cv2.line(image, poly[i], poly[i+1], color, 1, lineType=cv2.LINE_AA)

            image = cv2.line(image, poly[3], poly[0], color, 3, lineType=cv2.LINE_AA)

            image = cv2.putText(image, polyobj["id"], poly[3], cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=0.5)
        cv2.imwrite(osp.join(self._outdir, imgfile), image)
    def output(self, outdir):
        outdir = osp.join(self.basedir, outdir)
        self._outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        print("Filtering raw, unmask", self.imgdir)
        pool = mp.Pool(8)
        files = os.listdir(self.imgdir)
        r = list(tqdm.tqdm(pool.imap(self._output, files), 
            total=len(files), desc=f"Output bbox: {outdir}") )
        # for imgfile in tqdm.tqdm(os.listdir(self.imgdir)):
        #     self._output(imgfile)
            # break

import xml.etree.ElementTree as ET
class VocRotateVisualizer(Visualizer) :

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def parse(self, filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            obj_struct["pose"] = obj.find("pose").text
            obj_struct["truncated"] = int(obj.find("truncated").text)
            obj_struct["difficult"] = int(obj.find("difficult").text)
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                float(bbox.find("xmin").text),  #BFP_MARK
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
                float(bbox.find("angle").text),
            ]
            objects.append(obj_struct)

        return objects

    def vis(self, name):
        """
        name: str without suffix
        """
        # print(self.imgdir, name, self.ImgExt)
        file = self.imgdir + name + self.ImgExt
        image = io.imread(file)  # 读取图像
        instances = self.parse(self.labeldir + name + ".xml")
        # print(instances)
        for polyobj in instances:
            poly = polyobj["bbox"]
            # poly[-1] = 0
            poly = util.rotRecToPolygon(poly)
            poly = np.array(poly, dtype=np.int32)
            print(polyobj["bbox"], poly)
            poly = poly.reshape((-1, 2))
            s = len(poly)
            for i in range(s):
                image = cv2.line(image, tuple(poly[i]), tuple(poly[(i+1)%s]), (255,0,0), 2, lineType=cv2.LINE_AA)
            # image = cv2.polylines(image, poly, True, (0, 255, 255), 5)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(image)
        try:
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state('zoomed')    #最大化
        except : pass
        plt.show()
import xml.etree.ElementTree as ET
class VocVisualizer(Visualizer) :

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def parse(self, filename):
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall("object"):
            obj_struct = {}
            obj_struct["name"] = obj.find("name").text
            if obj_struct['name'] != 'airport':
                continue
            # obj_struct["pose"] = obj.find("pose").text
            # obj_struct["truncated"] = int(obj.find("truncated").text)
            diff = obj.find("difficult")
            if diff:
                obj_struct["difficult"] = int(diff.text)
            else:
                obj_struct["difficult"] = 0
            oid = obj.find("oid").text
            bbox = obj.find("bndbox")
            obj_struct["oid"] = oid
            obj_struct["bbox"] = [
                float(bbox.find("xmin").text),  #BFP_MARK
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
            ]
            objects.append(obj_struct)

        return objects

    def vis(self, name):
        """
        name: str without suffix
        """
        # print(self.imgdir, name, self.ImgExt)
        file = self.imgdir + name + self.ImgExt
        image = io.imread(file)  # 读取图像
        instances = self.parse(self.labeldir + name + ".xml")
        # print(instances)
        for polyobj in instances:
            poly = polyobj["bbox"]
            # poly[-1] = 0
            poly = util.rotRecToPolygon(poly)
            poly = np.array(poly, dtype=np.int32)
            print(polyobj["bbox"], poly)
            poly = poly.reshape((-1, 2))
            s = len(poly)
            for i in range(s):
                image = cv2.line(image, tuple(poly[i]), tuple(poly[(i+1)%s]), (255,0,0), 2, lineType=cv2.LINE_AA)
            # image = cv2.polylines(image, poly, True, (0, 255, 255), 5)
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(image)
        try:
            figmanager = plt.get_current_fig_manager()
            figmanager.window.state('zoomed')    #最大化
        except : pass
        plt.show()
    def _output(self, imgfile):
        name, ext = osp.splitext(imgfile)
        dst_im_loc = osp.join(self._outdir, imgfile)
        if osp.exists(dst_im_loc):
            return
        # if "P11054__1.0__27648___21504" not in name: return
        if "vf" in name: return
        if "df" in name: return
        if "_unblur"  in name: return
        if "_cutmix"  in name: return
        if "r"  in name: return
        # if name.startswith("P"): return
        # if name.startswith("vf"): return
        # if name.startswith("hf"): return

        labelloc = self.labeldir + name + ".xml"
        if not osp.exists(labelloc): 
            return

        objects = self.parse(labelloc)
        if len(objects) == 0: 
            return
        src_im_loc = osp.join(self.imgdir, imgfile)
        image: "np.ndarray" = cv2.imread(src_im_loc)  # 读取图像
        
        assert image is not None
        # return
        for idx, object in enumerate(objects):
            color = (0, 0, 0)
            if object["difficult"] == 0: 
                color = (255, (np.pi * 30) % 255, 0)
            elif object["difficult"] == 1: 
                color = (30, 30, 225)
            elif object["difficult"] == 2:
                color = (0, 0, 225)
            bbox = object["bbox"]
            poly = np.array(bbox, dtype=np.int32)
            # poly = poly.reshape((-1, 1, 2))
            image = cv2.rectangle(image, poly[:2], poly[2:], color, 2, lineType=cv2.LINE_AA)
            name = f"{object['name']}-{object['oid']}"
            image = cv2.putText(image, name, poly[:2], cv2.FONT_HERSHEY_SIMPLEX, 
                                color=(0, 0, 0), fontScale=0.5)
        # print(src_im_loc)
        # breakpoint()
        cv2.imwrite(dst_im_loc, image)

    def output_bytype(self, outdir, dtype = "trainval"):
        outdir = osp.join(self.basedir, outdir)
        self._outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        print("Filtering raw, unmask", self.imgdir)
        pool = mp.Pool(8)
        # files = os.listdir(self.imgdir)
        ## filter from train or test
        files = []
        with open(osp.join(self.basedir, f"ImageSets/Main/{dtype}.txt")) as f:
            for line in f:
                name = f"{line.strip()}{self.ImgExt}"
                files.append(name)

        for imgfile in tqdm.tqdm(files):
            self._output(imgfile)
            # break

    def output(self, outdir):
        outdir = osp.join(self.basedir, outdir)
        self._outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        print("Filtering raw, unmask", self.imgdir)
        files = os.listdir(self.imgdir)
        ## filter from train or test
        mp_mode = False
        if mp_mode:
            pool = mp.Pool(8)
            r = list(tqdm.tqdm(pool.imap(self._output, files), 
                total=len(files), desc=f"Output bbox: {outdir}") )
        else:
            for imgfile in tqdm.tqdm(files):
                self._output(imgfile)
            # break

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=3,
          font_thickness=1,
          color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    txtpos = (x, int(y + text_h + font_scale - 1))
    cv2.putText(img, text, txtpos, font, font_scale, color, font_thickness)

    return text_size