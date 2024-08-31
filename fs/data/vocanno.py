from fsdet.structures import BoxMode
import xml.etree.ElementTree as ET
import xml.dom.minidom as md
import os.path as osp
import numpy as np
from fsdet.structures.anno import VocAnnotation, VocInstance, DatasetMetaInfo
    
class VocDataset():
    CLASS_NAMES = []
    IMG_EXT = "jpg"
    def __init__(self) -> None:
        self._instance_count = 0
        self.id_instance = False
        self.annotation_list: "list[VocAnnotation]" = []

    @classmethod
    def set_class_names(cls, names):
        cls.CLASS_NAMES = names


    def save_voc_annotation(self, vocAnno: "VocAnnotation", dst = None):
        root = ET.Element('annotation')
        filename = ET.SubElement(root, "filename")
        filename.text = f"{vocAnno.image_id}.{self.IMG_EXT}"
        size = ET.SubElement(root, "size")
        width  = ET.SubElement(size, "width");   
        width.text  = str(vocAnno.width) 
        height = ET.SubElement(size, "height");  
        height.text = str(vocAnno.height)
        depth  = ET.SubElement(size, "depth");   
        depth.text  = str(vocAnno.depth)
        
        for obj in vocAnno.annotations:

            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, "name")
            name.text = f"{obj.category}"
            oid = ET.SubElement(object, "oid"); 
            oid.text = f"{obj.oid}"
            
            diff = ET.SubElement(object, "difficult"); 
            diff.text = f"{0}"
            bndbox = ET.SubElement(object, "bndbox")
            tlrb = obj.bbox
            xmin = ET.SubElement(bndbox, "xmin"); xmin.text = f"{tlrb[0]}"
            ymin = ET.SubElement(bndbox, "ymin"); ymin.text = f"{tlrb[1]}"
            xmax = ET.SubElement(bndbox, "xmax"); xmax.text = f"{tlrb[2]}"
            ymax = ET.SubElement(bndbox, "ymax"); ymax.text = f"{tlrb[3]}"

        rough_string = ET.tostring(root, 'utf-8')
        dom = md.parseString(rough_string)
        if dst is None:
            dst = vocAnno.anno_name
        try:
            with open(dst, 'w', encoding='UTF-8') as fh:
                dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
        except Exception as err:
            print(f'错误信息：{err}')

    def parse_voc_annotation(self, file_loc, image_root: "str" = "JPEGImages"):
        """
        file_loc: training/Annotations/001.xml
        image_root: training/JPEGImages

        """
        assert len(self.CLASS_NAMES)
        # dir_name = osp.dirname(file_loc)
        file_name = osp.basename(file_loc)
        fileid, ext = osp.splitext(file_name)
        tree = ET.parse(file_loc)
        va = {
            "file_name": osp.join(image_root, f"{fileid}.{self.IMG_EXT}"),
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        r = VocAnnotation(**va)
        r.anno_name = file_loc
        
        annos = []
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if not (cls in self.CLASS_NAMES):
                continue
            oid_node = obj.find("oid")
            if self.id_instance:
                oid = self._instance_count
            else:
                if oid_node is not None:
                    oid = int(oid_node.text)
                else:
                    raise ValueError("No Oid Found")


            bbox = obj.find("bndbox")
            bbox = np.asarray([float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]])
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            idata = {
                "category_id": self.CLASS_NAMES.index(cls),
                "category": cls,
                "bbox": bbox.astype(int),
                "bbox_mode": BoxMode.XYXY_ABS,
                "oid": oid,
                "instance_idx": self._instance_count
            }
            ins = VocInstance(**idata)
            annos.append(ins)
            self._instance_count += 1
        r.set_annotations(annos)
        self.annotation_list.append(r)
    
    def vis_annotation(self, idx):
        anno = self.annotation_list[idx]
        
    @staticmethod
    def filter_empty_annotations(annotations: "list[VocAnnotation]") -> "list[VocAnnotation]":
        annos = []
        for anno in annotations:
            if len(anno.annotations) > 0:
                annos.append(anno)

        return annos
    
    @staticmethod
    def convert_anno_into_single(annos: "list[VocAnnotation]"):
        split_anno = []
        for anno in annos:
            anno: VocAnnotation
            for ins in anno.annotations:
                n_anno = VocAnnotation(anno.file_name, anno.image_id,
                    height=anno.height, width=anno.width, anno_name=anno.anno_name,
                    depth = anno.depth)
                n_anno.set_annotations([ins])
                split_anno.append(n_anno)
        return split_anno
            