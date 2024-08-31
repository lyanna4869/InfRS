import logging
import os, json
import os.path as osp
from copy import copy
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
from fvcore.common.file_io import PathManager
from fsdet.structures import BoxMode

from fs.core.rotate import rotate_instances
logger = logging.getLogger("fsdet.data.loader")
from fsdet.config import globalvar as gv
from collections import UserDict

class VocAnnotation(UserDict):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def set_instances(self, ins):
        self.data["annotations"] = ins

    def instances(self):
        return self.data['annotations']

class VocInstance(UserDict):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)


def get_imagesets_from_file(dirname:str, split:str):
    with PathManager.open(osp.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    return fileids

class VocAnnoLoader():
    BASE_CATEGORIES = None
    DATA_DIR = None # 
    ImageDirName = "JPEGImages"
    AnnoDirName = "Annotations"
    def __init__(self, name, dirname, split) -> None:
        self.name = name
        self.use_more_base = 'ploidy' in name
        self.is_shots = "shot" in name
        self.dirname = dirname
        self.split = split

    def get_seed_shot(self):
        if "seed" in self.name:
            seed = int(self.name.split('_seed')[-1]) 
            shot = self.name.split('_')[-2].split('shot')[0]
        else:
            seed = 0
            shot = self.name.split('_')[-1].split('shot')[0]
        return int(seed), int(shot)    
        
    def filter_for_tsne(self, annos: list, classnames: list):
        filtered_annos = []
        ### NOTE select instance here 
        manaual_select = gv.tsne_manaual_select_instances
        instance_count = gv.tsne_select_instance_count
        if not manaual_select:
            desired_annos = []
            count = {x : instance_count for x in range(len(classnames))}
            for anno in annos:
                for instance in anno['annotations']:
                    cls = instance['category_id']
                    prob = np.random.rand()
                    if count[cls] > 0 :
                        file_name = osp.basename(anno['file_name'])
                        file_name = osp.splitext(file_name)[0]
                        desired_annos.append((file_name, instance['object_id']))
                        count[cls] -= 1
        else:
            desired_annos = gv.tsne_desired_annos
        
        desired_files = set(x[0] for x in desired_annos)
        desired_inses = set(x[1] for x in desired_annos)
        for anno in annos:
            file_name = osp.basename(anno['file_name'])
            file_name = osp.splitext(file_name)[0]
            if file_name in desired_files:
                for instance in anno['annotations']:
                    if instance['object_id'] in desired_inses:
                        anno = copy(anno)
                        anno['annotations'] = [instance]
                        filtered_annos.append(anno)
        return filtered_annos

    # 读取 shot 对应的文件
    def _load_shot_files(self, split_dir, shot, classnames):
        """
        返回 dict
        {
            "cls": ["10222"]
        }
        """
        fileids = {}
        for cls in classnames:
            loc = osp.join(split_dir, f"box_{shot}shot_{cls}_train.txt")
            with PathManager.open(loc) as f:
                fileids_ = np.loadtxt(f, dtype=np.str).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [osp.splitext(osp.basename(fid))[0] for fid in fileids_]
                fileids[cls] = fileids_
        return fileids
    
    # 读取文件，加载 base class 
    def load_all_from_directory(self, classnames: list):
        """split: trainval train val test
        """
        dirname = self.dirname
        split   = self.split
        fileids = get_imagesets_from_file(dirname, split)
        return BaseDataLoader.load_instance_from_file_ex(fileids, classnames, dirname)
    
    def load_all_instances(self, classnames, dirname="", imageDir="JPGEImages", rotate_step=5):
        """加载所有的 instance，返回 annotations
        并根据 rotate_step 对图片进行旋转
        """
        if 'novelall' in self.name:
            annos = self.load_novel_test_instances(classnames)
            return annos
            
        annos = self.load_all_from_directory(classnames)
        if "tsne" in self.name: 
            return self.filter_for_tsne(annos, classnames)
        if "test" in self.name: return annos
        before = len(annos)
        img_dir = osp.join(dirname, imageDir)
        annos = rotate_instances(annos, step=rotate_step, img_dir=img_dir)
        logger.info(f"Annotations: {len(annos)} , before: {before}")
        return annos
    
    def load_more_base(self, classnames, split_dir):
        raise ValueError("Unimplemented or tested")
        ploidy = self.name.split('_')[-1]
        split_id = self.name.split('_')[3][-1]
        shot = self.name.split('_')[-3].split('shot')[0]
        seed = int(self.name.split('_')[-2].replace('seed', ''))
        split_dir = osp.join(split_dir, ploidy, f'split{split_id}', f"seed{seed}")

        fileids = self._load_shot_files(split_dir, shot, classnames)
        dicts = []
        for cls, fileids_ in fileids.items():
            dicts_ = self.load_instance_from_file_ex(fileids_, classnames, split_dir)
            # cls
            # filter(lambda x: x['annotations'][], dicts_)
            # this make sure that dataset_dicts has *exactly* K-shot
            if self.use_more_base and cls in self.BASE_CATEGORIES[int(split_id)]:
                if len(dicts_) > int(shot) * int(ploidy[0]):
                    dicts_ = np.random.choice(dicts_, int(shot)*int(ploidy[0]), replace=False)
            else:
                if len(dicts_) > int(shot):
                    dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
        # dicts = rotate_instances(dicts)
        return dicts
    
    def load_novel_test_instances(self, classnames: list, novel_first_index: int = 15, base_percent :float = 0.3):
        """split: trainval train val test
        """
        dirname = self.dirname
        cls_count = 2000
        fileids = get_imagesets_from_file(dirname, self.split)
        all_annos = self.load_instance_from_file_ex(fileids, classnames, dirname)
        for aa in all_annos:
            new_annotations = []
            for ins in aa['annotations']:
                if ins['category_id'] >= novel_first_index:
                    new_annotations.append(ins)
                elif cls_count > 0:
                    new_annotations.append(ins)
                    cls_count -= 1
            aa['annotations'] = new_annotations

        return all_annos
    
    @staticmethod
    def load_instance_from_file_ex(fileids, classnames: list, dirname: str, 
            annoDir="Annotations", imgDir="JPEGImages"):
        """ fileids: only filename, no prefix, no suffix
        classnames:  加载的元素必定在 classnames 中
        dirname:  root directory contains both annotation sub dir and image sub dir

        返回 数组，每个数组元素是一个 annotation（对应file），将 instance 放在一起
        """
        assert classnames is not None, "API has changed, please check params passed in"
        total_annotations = []
        idx = 0
        for fileid in fileids:
            anno_file = osp.join(dirname, annoDir, fileid + ".xml")
            jpeg_file = osp.join(dirname, imgDir,  fileid + ".jpg")

            tree = ET.parse(anno_file)

            r = VocAnnotation({
                "file_name": jpeg_file,
                "image_id": fileid,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            })
            instances = []
            ins_count = 0
            for obj in tree.findall("object"):
                cls = obj.find("name").text
                if not (cls in classnames):
                    continue
                oid = int(obj.find("oid").text)
                bbox = obj.find("bndbox")
                bbox = np.asarray([float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]])
                bbox[0] -= 1.0
                bbox[1] -= 1.0
                ins = VocInstance({
                    "category_id": classnames.index(cls),
                    "category": cls,
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "oid": oid,
                    "instance_idx": idx
                })
                instances.append(ins)
                idx += 1
                ins_count += 1
                if ins_count >= gv.instances_per_annotations > 0:
                    break
            r.set_instances(instances)
            total_annotations.append(r)
        return total_annotations
    
    @staticmethod
    def save_instance(annotaion: VocAnnotation, dst_loc: str):
        assert isinstance(annotaion, VocAnnotation) 
        # print(annotaion)
        root = ET.Element('annotation')
        filename = ET.SubElement(root, "filename")
        filename.text = f"{annotaion['image_id']}.jpg"
        size = ET.SubElement(root, "size")
        width  = ET.SubElement(size, "width");   width.text  = str(annotaion['width']) 
        height = ET.SubElement(size, "height");  height.text = str(annotaion['height'])
        depth  = ET.SubElement(size, "depth");   depth.text  = str(annotaion.get('depth', 3))
        
        for obj in annotaion.instances():
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, "name")
            name.text = f"{obj['category']}"
            oid = ET.SubElement(object, "oid"); oid.text = f"{obj['oid']}"
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

    @staticmethod
    def load_from_metafile(dirname: str, split: str, classnames:list = None):
        """ read image file names from {dirname}/ImageSets/Main/{split}.txt
        """
        fileids = get_imagesets_from_file(dirname, split)
        return VocAnnoLoader.load_instance_from_file_ex(fileids, classnames, dirname)
    
BaseDataLoader = VocAnnoLoader