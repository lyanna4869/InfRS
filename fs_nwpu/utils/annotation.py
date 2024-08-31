import xml.etree.ElementTree as ET
import xml.dom.minidom as md
from collections import UserDict, Counter
import numpy as np
import os, os.path as osp
from PIL import Image, ImageFilter
from typing import TypedDict

from fsdet.structures.boxes import BoxMode
from fsdet.structures.anno import VocAnnotation, VocInstance
from fs.data.loader import VocDataset as NwpuDataset
from fs_nwpu.core.category import ORIGINAL_CLASS_NAMES as NWPU_CLASSES

IMAGE_SETS_LOC = "ImageSets/Main/"
POSITIVE_IMG_DIRECTORY = "positive image set"
ANNOTATION_DIRECTORY = "ground truth"
IMG_EXT = ".jpg"
def parse_nwpu_img_meta():
    nwpu_img_meta = {}
    for file in os.listdir(POSITIVE_IMG_DIRECTORY):
        # file: 001.jpg
        file_name = osp.splitext(file)[0]  
        file_loc = osp.join(POSITIVE_IMG_DIRECTORY, file)
        im = Image.open(file_loc)
        w, h = im.size
        nwpu_img_meta[file_name] = {
            "width": w, "height": h, "depth": 3,
            "image_id": file_name
        }
    return nwpu_img_meta


class NwpuInstance(UserDict):
    def __init__(self, d: dict):
        super().__init__(d)
    def __hash__(self):
        return self.data['oid']
    def category(self):
        return self.data['name']


classMap = {idx+1: name for idx, name in enumerate(NWPU_CLASSES)}

class NwpuSingleObject(TypedDict):
    xy: "np.ndarray"
    name: "str"

class NwpuTxtDataset(NwpuDataset):
    def __init__(self) -> None:
        super().__init__()
        self.trainValCount = Counter() # key: image id
        self.object_id_count = 0

    def load_from_txt_folder(self, root: "str", img_folder: "str"):
        
        for filename in sorted(os.listdir(root)):
            file = osp.join(root, filename)
            file_id, ext = osp.splitext(osp.basename(file))
            im_file = osp.join(img_folder, file_id + IMG_EXT)

            anno: "VocAnnotation" = self.parseObjectFromTxt(file, im_file)
            self.annotation_list.append(anno)

    def parseObjectFromTxt(self, file, im_file = None):
        """ 
        读取txt文件 
        """
        objects: "list[str]" = []

        with open(file, 'r') as f:
            # contents = f.read()
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                objects.append(line)
        ann = VocAnnotation(file_name=file, image_id=im_file)
        retrived = []
        for obj in objects:
            # print(obj)
            obj = obj.replace("(", "")
            obj = obj.replace(")", "")
            if "," in obj:
                obj = list(map(int, obj.split(",")))
            else:
                obj = list(map(lambda x: int(float(x)), obj.split(" ")))

            cls_id = obj[4]
            if cls_id in classMap:
                name = classMap[cls_id]
            else:
                print(file)
                continue
            if self.id_instance:
                oid = self.object_id_count
            else:
                oid = int(obj[5])
            self.object_id_count += 1
            ann_ins: "VocInstance" = VocInstance(category_id=cls_id, category=name, 
                                                 bbox=np.asarray(obj[:-1]), bbox_mode=BoxMode.XYXY_ABS,
                                                 oid=oid, instance_idx=self.object_id_count)

            retrived.append(ann_ins)

        ann.set_annotations(retrived)
        return ann

from fs.core.iou_calc import hbb_iou
    
def parse_nwpu_annotations(filenames=None):
    nwpu_annotations = {} # key is file name
    
    object_id = 0
    for file in sorted(os.listdir(ANNOTATION_DIRECTORY)):
        # file: 001.txt
        file_name = osp.splitext(file)[0]  
        
        file_loc = osp.join(ANNOTATION_DIRECTORY, file)
        nwpu_instances = []
        with open(file_loc, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                line = line.replace("(", "")
                line = line.replace(")", "")
                data = line.split(",")
                insdict = {
                    "bbox": np.asarray(list(map(int, data[:4])), dtype=np.int32),
                    "oid": object_id,
                    "name": NWPU_CLASSES[int(data[4])-1],
                    "image_id": file_name
                }
                object_id += 1
                nwpu_instances.append(NwpuInstance(insdict))
        if filenames is not None and file_name not in filenames:
            # print(file_name)
            continue
        nwpu_annotations[file_name] = nwpu_instances
    return nwpu_annotations


def save_nwpu_anno(annos: list, meta: dict, dst_loc: str):
    """annos: list of NwpuInstances
    dst: abs file location
    """
    # dst = osp.join(dst, f"{anno.image_id}.xml")

    root = ET.Element('annotation')
    filename = ET.SubElement(root, "filename")
    filename.text = f"{meta['image_id']}.jpg"
    size = ET.SubElement(root, "size")
    width  = ET.SubElement(size, "width");   width.text  = str(meta['width']) 
    height = ET.SubElement(size, "height");  height.text = str(meta['height'])
    depth  = ET.SubElement(size, "depth");   depth.text  = str(meta['depth'])
    
    for obj in annos:
        object = ET.SubElement(root, 'object')
        name = ET.SubElement(object, "name")
        name.text = f"{obj['name']}"
        oid = ET.SubElement(object, "oid"); 
        oid.text = f"{obj['oid']}"
        bndbox = ET.SubElement(object, "bndbox")
        tlrb = obj["bbox"]
        xmin = ET.SubElement(bndbox, "xmin"); xmin.text = f"{tlrb[0]}"
        ymin = ET.SubElement(bndbox, "ymin"); ymin.text = f"{tlrb[1]}"
        xmax = ET.SubElement(bndbox, "xmax"); xmax.text = f"{tlrb[2]}"
        ymax = ET.SubElement(bndbox, "ymax"); ymax.text = f"{tlrb[3]}"

    # ET.dump
    # tree = ET.ElementTree(root)
    # tree.writexml(dst, short_empty_elements=False)
    rough_string = ET.tostring(root, 'utf-8')
    dom = md.parseString(rough_string)
    try:
        with open(dst_loc, 'w', encoding='UTF-8') as fh:
            # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
            # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
            dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
    except Exception as err:
        print(f'错误信息：{err}')
        
def save_nwpu_all_as_voc():
    annos = parse_nwpu_annotations()
    metas = parse_nwpu_img_meta()
    
    os.makedirs("Annotations", exist_ok=True)
    for k, v in annos.items():
        dst_loc = osp.join("Annotations", f"{k}.xml")
        save_nwpu_anno(v, metas[k], dst_loc)
import os
import os.path as osp
from PIL import Image
import numpy as np
import copy


class NwpuImageSpliter(object):
    """将不规则的 图片，划分成大小为 1024x1024 的图片
    """
    POSITIVE_IMG_DIRECTORY = "positive image set"
    def __init__(self, root_path, nwpu_annotations, subsize = 1024, overlap=0.2, 
                padding=True, ext=".jpg") -> None:
        out_img_path = osp.join(root_path, "JPEGImages")
        os.makedirs(out_img_path, exist_ok=True)
        out_anno_path = osp.join(root_path, "Annotations")
        os.makedirs(out_anno_path, exist_ok=True)
        
        self.out_img_path = out_img_path
        self.out_label_path = out_anno_path
        self.nwpu_annotations = nwpu_annotations
        self.subsize = subsize
        self.slide = int (subsize * (1-overlap))
        self.padding = padding
        self.im_ext = ext
        self.skip_empty_gt = True

    def _save_image_patches(self, img: Image.Image, subimgname: str, rect: list):
        """
        img: 原始图片
        subimgname: 新图片名称
        left
        """
        outdir = osp.join(self.out_img_path, subimgname + self.im_ext)
        subimg = img.crop(rect)
        if self.padding:
            outimg = Image.new(subimg.mode, (self.subsize, self.subsize))
            outimg.paste(subimg)
            outimg.save(outdir)
        else:
            subimg.save(outdir)

    def poly_orig2sub(self, rect, poly):
        """返回 poly xy 在子图中的坐标"""
        left, up, right, down = rect
        # if self.padding:
        #     right = self.subsize
        #     down = self.subsize
        polyInsub = np.zeros(len(poly))
        for i in range(0, len(poly), 2):
            polyInsub[i] = int(poly[i] - left)
            polyInsub[i + 1] = int(poly[i + 1] - up)
            polyInsub[i] = max(polyInsub[i], 0)
            polyInsub[i] = min(polyInsub[i], self.subsize)
            polyInsub[i+1] = max(polyInsub[i+1], 0)
            polyInsub[i+1] = min(polyInsub[i+1], self.subsize)

        return polyInsub
    
    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou
    
    def translate_sub_instances(self, imgpoly, objects, rect):
        valid_objects = []
        for obj in objects:
            poly = obj["bbox"] # w1y1x2y2
            gtpoly = poly
            # gtpoly = shgeo.Polygon(gtpoly)

            inter_poly, half_iou_obj, half_iou_img = hbb_iou(gtpoly, imgpoly)
            # print('writing...')
            # outline = ""
            ## if the left part is too small, label as '2'
            # diffi = obj["difficult"] if (half_iou > self.thresh) else 2
            if (half_iou_obj > 0.3):
                # 目标完全在图片中
                polyInsub = self.poly_orig2sub(rect, poly)
                outline = ' '.join(list(map(str, polyInsub)))
            else:
                continue

            sobj = copy.deepcopy(obj)
            sobj['bbox'] = polyInsub.astype(np.int32)
            valid_objects.append(sobj)
        # if len(valid_objects) == 0:
        #     print("pass")
        return valid_objects
    
    def save_patches(self, img, objects, sub_img_name, rect):
        """保存裁剪后的图片并写入 txt 
        objects {"poly": [], "difficult": 0 }
        """
        left, up, right, down = rect
        
        imgpoly = np.asarray(rect)  # 图片在原图的坐标

        valid_objects = self.translate_sub_instances(imgpoly, objects, rect)
        if len(valid_objects) == 0 and self.skip_empty_gt:
            # self.skip_patchs_count += 1
            return False
        
        ## write as xml
        out_loc = osp.join(self.out_label_path, sub_img_name + '.xml')
        if self.padding:
            meta = {"width": self.subsize, "height": self.subsize, "depth": 3, "image_id": sub_img_name}
        else:
            meta = {"width": right - left, "height": down - up, "depth": 3, "image_id": sub_img_name}
        save_nwpu_anno(valid_objects, meta, out_loc)
        
        self._save_image_patches(img, sub_img_name, rect)
        return True

    def calc_image_rects(self, width, height):
        imageareas = []
        left, up = 0, 0
        cache_left, cache_up = 0, 0
        try_left, try_up = 0, 0
        last_left, last_up = -100, -100
        right, down = width, height

        while (left < width):
            if (left + self.subsize >= width):
                # align right side into image right
                left = max(width - self.subsize, 0)
                cache_left = left
            if left - last_left > 5:
                try_left = left
                if try_left <= last_left + 10:
                    try_left = left # degrades as normal Spliter
                else:
                    left = try_left
                while (up < height):
                    if (up + self.subsize >= height):
                        # align bottom side into image bottom
                        up = max(height - self.subsize, 0)
                        cache_up = up

                    if up - last_up > 5: 
                        right = min(left + self.subsize, width - 1)
                        down = min(up + self.subsize, height - 1)
                        try_up = up

                        if try_up <= last_up + 10:
                            try_up = up # degrades as normal Spliter
                        else:
                            up = try_up
                        area = (int(left), int(up), int(right), int(down))
                        imageareas.append(area)
                    
                    if (cache_up + self.subsize >= height):
                        break
                    else:
                        up = cache_up + self.slide
            last_left = left
            
            if (cache_left + self.subsize >= width):
                break
            else:
                left = cache_left + self.slide
        imageareas = set(imageareas)
        imageareas = list(sorted(imageareas))
        return imageareas
    
    def split(self, origin_filename):
        src_im_name = osp.join(self.POSITIVE_IMG_DIRECTORY, f"{origin_filename}.jpg")
        src_im = Image.open(src_im_name)
        # src_anno_name = osp.join()
        if origin_filename not in self.nwpu_annotations:
            return None
        objects = self.nwpu_annotations[origin_filename]
        im_w, im_h = src_im.size
        img_rects = self.calc_image_rects(im_w, im_h)
        # self.split_im(src_im, img_rects)
        out_base_name = f"{origin_filename}__{1.0:.1f}__"
        sub_img_names = []
        for rect in img_rects:
            sub_img_name = f"{out_base_name}{rect[0]}___{rect[1]}"
            # self.f_sub.write(name + ' ' + sub_img_name + ' ' + str(left) + ' ' + str(up) + '\n')
            if self.save_patches(src_im, objects, sub_img_name, rect):
                sub_img_names.append(sub_img_name)

        return sub_img_names
    
    def translate_objects(self, origin_filename, nwpu_annotations):
        """计算 目标在 子图上的坐标
        """
        src_im_name = osp.join(self.POSITIVE_IMG_DIRECTORY, f"{origin_filename}.jpg")
        src_im = Image.open(src_im_name)
        # src_anno_name = osp.join()
        if origin_filename not in nwpu_annotations:
            return None
        objects = nwpu_annotations[origin_filename]
        im_w, im_h = src_im.size
        img_rects = self.calc_image_rects(im_w, im_h)
        # self.split_im(src_im, img_rects)
        out_base_name = f"{origin_filename}__{1.0:.1f}__"
        sub_img_instances = []
        for rect in img_rects:
            sub_img_name = f"{out_base_name}{rect[0]}___{rect[1]}"
            # self.f_sub.write(name + ' ' + sub_img_name + ' ' + str(left) + ' ' + str(up) + '\n')
            # self.save_patches(src_im, objects, sub_img_name, rect)
            left, up, right, down = rect
        
            imgpoly = np.asarray(rect)  # 图片在原图的坐标
            valid_objects = self.translate_sub_instances(imgpoly, objects, rect)
            sub_img_instances.append((sub_img_name, valid_objects))

        return sub_img_instances

        