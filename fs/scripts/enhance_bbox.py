import os
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from fsdet.structures import BoxMode

CUR_PATH = osp.abspath(osp.dirname(__file__))
PROJECT_ROOT = osp.abspath(osp.join(CUR_PATH, "..", ".."))
ROOT_PATH = osp.abspath(osp.join(PROJECT_ROOT, "datasets"))
# from fs.core.data.VocAnno import VocImage, VocInstance
def load_datasets(src_dir):
    set_file = osp.join(src_dir, "ImageSets", "Main", "trainval_RAW.txt")
    fileids = []
    with open(set_file) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            fileids.append(line)
    img_dir = osp.join(src_dir, "JPEGImages")
    label_dir = osp.join(src_dir, "Annotations")
    annotations = []
    for fileid in fileids:
        anno_file = osp.join(label_dir, fileid + ".xml")
        img_file  = osp.join(img_dir, fileid + ".jpg")
    
        tree = ET.parse(anno_file)

        r = {
            "file_name": img_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0

            instances.append({
                "category_name": cls,
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
            })
        r["annotations"] = instances
        annotations.append(r)
    return annotations
from PIL import Image
from fs.core.rotate import rotate_bbox
from tqdm import tqdm
def rotate_annotations(annotations, base_dir, angle=90):
    ran = slice(0, len(annotations), 1)
    new_fileids = []
    label_dir = osp.join(base_dir, "Annotations")
    img_dir = osp.join(base_dir, "JPEGImages")
    for anno in tqdm(annotations[ran], desc=f"rotate images, angle: {angle}"):
        fileid = anno["image_id"]
        im = Image.open(anno["file_name"])
        nim = im.rotate(angle, expand=True)
        
        nfileid = f"{fileid}_r{angle}"
        new_fileids.append(nfileid)
        anno_file = os.path.join(label_dir, nfileid + ".xml")
        img_file = osp.join(img_dir, nfileid + ".jpg")
        
        old_instances = anno["annotations"]

        instances = []
        center = (anno["width"] // 2, anno["height"] // 2)
        hw = (anno["height"], anno["width"])
        # 按照角度旋转 bbox
        for instance in old_instances:
            bbox = instance["bbox"]
            bbox = rotate_bbox(bbox, angle, center, hw)
            # imdraw = ImageDraw.Draw(nim)
            # imdraw.rectangle(bbox.tolist(), outline=(255, 0, 0))
            iid = instance.get("oid", 0)
            instances.append({
                "category_name": instance["category_name"],
                "bbox": bbox.tolist(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "oid": iid
            })

        # 保存图片
        nim.save(img_file)
        root = ET.Element('annotation')
        root.text = "\n"
        filename = ET.SubElement(root, "filename")
        filename.text = f"{fileid}.jpg"
        filename.tail = "\n"
        size = ET.SubElement(root, "size")
        size.text = "\n"
        size.tail = "\n"
        width = ET.SubElement(size, "width");   width.text  = str(nim.width); width.tail="\n"
        height = ET.SubElement(size, "height"); height.text = str(nim.height); height.tail="\n"
        # depth = ET.SubElement(size, "depth");   depth.text  = "0"; depth.tail="\n"
        for ins in instances:
            object = ET.SubElement(root, 'object')
            object.text = "\n"
            name = ET.SubElement(object, "name")
            name.tail = "\n"
            name.text = f"{ins['category_name']}"
            diff = ET.SubElement(object, "difficult")
            diff.text = "0"; diff.tail = "\n"
            pose = ET.SubElement(object, "pose")
            pose.text = "unspecified"; pose.tail = "\n"
            truncatged = ET.SubElement(object, "truncated")
            truncatged.text = "0"; truncatged.tail = "\n"

            bndbox = ET.SubElement(object, "bndbox")
            bndbox.text = "\n"; bndbox.tail = "\n"
            oid = ET.SubElement(bndbox, "oid"); oid.text = f"{ins['oid']}"; oid.tail = "\n"
            bbox = ins['bbox']
            xmin = ET.SubElement(bndbox, "xmin"); xmin.text = f"{bbox[0]}"; xmin.tail = "\n"
            ymin = ET.SubElement(bndbox, "ymin"); ymin.text = f"{bbox[1]}"; ymin.tail = "\n"
            xmax = ET.SubElement(bndbox, "xmax"); xmax.text = f"{bbox[2]}"; xmax.tail = "\n"
            ymax = ET.SubElement(bndbox, "ymax"); ymax.text = f"{bbox[3]}"; ymax.tail = "\n"

            object.tail = "\n"
        # data = ET.save(root)
        tree = ET.ElementTree(root)
        tree.write(anno_file, short_empty_elements=False)

    set_file = osp.join(base_dir, "ImageSets", "Main", "trainval.txt")
    with open(set_file, "a") as f:
        for fileid in new_fileids:
            f.write(fileid + "\n")

def flip_annotations(annotations, base_dir):
    ran = slice(0, len(annotations), 1)
    new_fileids = []
    label_dir = osp.join(base_dir, "Annotations")
    img_dir = osp.join(base_dir, "JPEGImages")
    for anno in tqdm(annotations[ran], desc=f"flip images"):
        fileid = anno["image_id"]
        im = Image.open(anno["file_name"])
        nim = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        nfileid = f"{fileid}_hf"
        new_fileids.append(nfileid)
        anno_file = os.path.join(label_dir, nfileid + ".xml")
        img_file = osp.join(img_dir, nfileid + ".jpg")
        nim.save(img_file)
        
        old_instances = anno["annotations"]

        instances = []
        width = anno["width"]
        center = (width // 2, anno["height"] // 2)
        hw = (anno["height"], width)
        # 按照角度旋转 bbox
        for instance in old_instances:
            bbox = instance["bbox"]
            h_bbox = bbox.copy()
            h_bbox[0] = width - bbox[2] - 1
            h_bbox[2] = width - bbox[0] - 1
            # imdraw = ImageDraw.Draw(nim)
            # imdraw.rectangle(bbox.tolist(), outline=(255, 0, 0))
            iid = instance.get("oid", 0)
            instances.append({
                "category_name": instance["category_name"],
                "bbox": h_bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "oid": iid
            })


        root = ET.Element('annotation')
        root.text = "\n"
        filename = ET.SubElement(root, "filename")
        filename.text = f"{fileid}.jpg"
        filename.tail = "\n"
        size = ET.SubElement(root, "size")
        size.text = "\n"
        size.tail = "\n"
        width = ET.SubElement(size, "width");   width.text  = str(hw[1]); width.tail="\n"
        height = ET.SubElement(size, "height"); height.text = str(hw[0]); height.tail="\n"
        # depth = ET.SubElement(size, "depth");   depth.text  = "0"; depth.tail="\n"
        for ins in instances:
            object = ET.SubElement(root, 'object')
            object.text = "\n"
            name = ET.SubElement(object, "name")
            name.tail = "\n"
            name.text = f"{ins['category_name']}"
            diff = ET.SubElement(object, "difficult")
            diff.text = "0"; diff.tail = "\n"
            pose = ET.SubElement(object, "pose")
            pose.text = "unspecified"; pose.tail = "\n"
            truncatged = ET.SubElement(object, "truncated")
            truncatged.text = "0"; truncatged.tail = "\n"

            bndbox = ET.SubElement(object, "bndbox")
            bndbox.text = "\n"; bndbox.tail = "\n"
            oid = ET.SubElement(bndbox, "oid"); oid.text = f"{ins['oid']}"; oid.tail = "\n"
            bbox = ins['bbox']
            xmin = ET.SubElement(bndbox, "xmin"); xmin.text = f"{bbox[0]}"; xmin.tail = "\n"
            ymin = ET.SubElement(bndbox, "ymin"); ymin.text = f"{bbox[1]}"; ymin.tail = "\n"
            xmax = ET.SubElement(bndbox, "xmax"); xmax.text = f"{bbox[2]}"; xmax.tail = "\n"
            ymax = ET.SubElement(bndbox, "ymax"); ymax.text = f"{bbox[3]}"; ymax.tail = "\n"

            object.tail = "\n"
        # data = ET.save(root)
        tree = ET.ElementTree(root)
        tree.write(anno_file, short_empty_elements=False)

    set_file = osp.join(base_dir, "ImageSets", "Main", "trainval.txt")
    with open(set_file, "a") as f:
        for fileid in new_fileids:
            f.write(fileid + "\n")

def enhance_nwpu_all():
    base_dir = osp.join(ROOT_PATH, "NWPU")
    annotations = load_datasets(base_dir)
    print(len(annotations))
    set_file = osp.join(base_dir, "ImageSets", "Main", "trainval_RAW.txt")
    fileids = []
    with open(set_file) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            fileids.append(line)

    set_file = osp.join(base_dir, "ImageSets", "Main", "trainval.txt")
    with open(set_file, "w") as f:
        for fileid in fileids:
            f.write(fileid + "\n")

    for angle in [90, 180, 270]:
        rotate_annotations(annotations, base_dir = base_dir, angle=angle)
    flip_annotations(annotations,  base_dir = base_dir)

if __name__ == "__main__":
    base_dir = osp.join(ROOT_PATH, "NWPU", "train_part", "seed10")
    annotations = load_datasets(base_dir)
    print(len(annotations))
    set_file = osp.join(base_dir, "ImageSets", "Main", "trainval_RAW.txt")
    fileids = []
    with open(set_file) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            fileids.append(line)

    set_file = osp.join(base_dir, "ImageSets", "Main", "trainval.txt")
    with open(set_file, "w") as f:
        for fileid in fileids:
            f.write(fileid + "\n")

    for angle in [90, 180, 270]:
        rotate_annotations(annotations, base_dir = base_dir, angle=angle)
    flip_annotations(annotations,  base_dir = base_dir)

