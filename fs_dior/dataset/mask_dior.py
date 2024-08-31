__version__="0.6"
"""another mask version"""

import os
import os.path as osp
import random
from PIL import Image, ImageDraw, ImageFilter
import shutil as sh
import numpy as np

from tqdm import tqdm
from prepare_dior_few_shot import NOVEL_CATEGORIES, VOC_CLASSES, VocAnnotation, VocObject
CUR_DIR = osp.abspath(osp.dirname(__file__))
BASE_DIR = osp.join(CUR_DIR, "DIOR")
USE_SHOT = 20

NovelAnnotationDirPrefix = f"{USE_SHOT}shot/NovelAnnotations20_"
NovelAnnotationDirName = f"{NovelAnnotationDirPrefix}_all"
ReservedNovelAnnotationDirName = "ReserveNovelAnnotations"
ReservedNovelAnnotationDir = osp.join(CUR_DIR, "DIOR", ReservedNovelAnnotationDirName)
NovelAnnotationDir = osp.join(CUR_DIR, "DIOR", NovelAnnotationDirName)
NovelJPEGImagesDir = osp.join(CUR_DIR, "DIOR", "ReserveNovelJPEGImages")
AnnotationDir = osp.join(CUR_DIR, "DIOR", "Annotations")
JPEGImagesDir = osp.join(CUR_DIR, "DIOR", "JPEGImages")

os.makedirs(NovelJPEGImagesDir, exist_ok=True)
os.makedirs(ReservedNovelAnnotationDir, exist_ok=True)

DEBUG_INFO = False
split = 1
class Dataset():
    def __init__(self, dir, *args, **kwargs) -> None:
        annos = os.listdir(dir)
        fileids = []
        for anno in sorted(annos):
            fileids.append(anno[:-4])
        annos = {}
        for fileid in fileids:
            anno = VocAnnotation(fileid, *args, **kwargs)
            annos[fileid] = anno
        self.annos = annos

    def save_mask_image(self, fileid):
        pass
    def maskImage(fileid):
        pass

 
def compute_iou(xy1, xy2):
    """
    computing IoU
    :param xy1: (x0, y0, x1, y1), which reflects
    :param xy2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    area1 = (xy1[2] - xy1[0]) * (xy1[3] - xy1[1])
    area2 = (xy2[2] - xy2[0]) * (xy2[3] - xy2[1])
 
    # computing the sum_area
    sum_area = area1 + area2
 
    # find the each edge of intersect rectangle
    left_line = max(xy1[0], xy2[0])
    right_line = min(xy1[2], xy2[2])
    top_line = max(xy1[1], xy2[1]) # 
    bottom_line = min(xy1[3], xy2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0

    intersect = (right_line - left_line) * (bottom_line - top_line)
    if intersect == area1 or intersect == area2 :
        return 1
    union_area = sum_area - intersect
    return (intersect / area1) * 1.0

def gather_annos(annos, shot):
    """获取 train.txt 中的文件，并获取对应的 xml 文件
    """
    annos_by_cat = {c: [] for c in VOC_CLASSES}
    annos = annos.values()
    save_path = osp.join(BASE_DIR, f'split{split}/seed0')
    os.makedirs(save_path, exist_ok=True)
    # print(annos)
    for cls in VOC_CLASSES:
        dst_annos = []
        for anno in annos:
            if cls not in anno.clsset:
                continue
            dst_annos.append(anno)
        filename = f'box_{shot}shot_{cls}_train.txt'
        # print(cls, dst_annos)
        names = [x.img_file for x in dst_annos]
        loc = osp.join(save_path, filename)
        with open(loc, 'w') as fp:
            fp.write('\n'.join(names))
    tmp_dir = osp.join(BASE_DIR, f"tmp_filtered_{split}")
    os.makedirs(tmp_dir, exist_ok=True)
    for anno in annos:
        for obj in anno.all_objects():
            obj.save(tmp_dir)

def main(shot = 10):
    ds = Dataset(AnnotationDir, "Annotations", "JPEGImages")
    nds = Dataset(NovelAnnotationDir, NovelAnnotationDirName, "JPEGImages", autoid=False)
    print("Annos", len(ds.annos), "Novel", len(nds.annos))
    masked_instance_count = 0
    skip_img_count = 0
    # all_objects = set()
    # novelds_objects = set()
    # reserve 10 shot
    instance_per_cls = {c: shot for c in VOC_CLASSES}
    novelds_annos = {}
    # 从 Novel 中挑选出 object
    novelds_annotations = list(nds.annos.values())
    # random.shuffle(values)
    for novelanno in tqdm(novelds_annotations):
        novelanno.choice_objects(instance_per_cls)
        if novelanno.all_object_count() == 0:
            continue
        
        novelds_annos[novelanno.fileid] = novelanno

    shots_count = {c: 0 for c in VOC_CLASSES}
    for fileid, anno in novelds_annos.items():
        for cls, objects in anno.object_per_cls.items():
            shots_count[cls] += len(objects)
        anno.save(osp.join(ReservedNovelAnnotationDir, f"{anno.fileid}.xml"))
    print("remain novel annotations: ", len(novelds_annos), shots_count)
    gather_annos(novelds_annos, shot)
    # exit(0)
    iou_dist = []
    for fileid, novelanno in tqdm(novelds_annos.items()):
        novelds_objects = novelanno.all_objects()
        if len(novelds_objects) == 0:
            continue
        # if fileid != "00024": continue
        # for object in novelds_objects:
        #     print(object)
        allds_anno = ds.annos[fileid]
        allds_objects = allds_anno.all_objects()
        assert len(novelds_objects) < 50
        src_img = osp.join(JPEGImagesDir, f"{fileid}.jpg")
        sdst_img = osp.join(NovelJPEGImagesDir, f"{fileid}_unblur.jpg")
        dst_img = osp.join(NovelJPEGImagesDir, f"{fileid}.jpg")
        dst_nimg = osp.join(NovelJPEGImagesDir, f"{fileid}_mask.jpg")
        # 保证 instance 数量相等
        assert len(allds_objects) >= len(novelds_objects)
        if len(allds_objects) == len(novelds_objects): 
            # 直接 copy 图片
            skip_img_count += 1
            sh.copy(src_img, dst_img)
            continue

        novel_not_contain_objects = []
        novel_contain_objects = []
        nobjectids = [x.id for x in novelds_objects]

        all_idset = set( [x.id for x in allds_objects])
        novel_idset = set(nobjectids)
        if not novel_idset.issubset(all_idset):
            print(novelds_objects)
            print(allds_objects)
            print("All sets not contain all novel sets")
            exit(1)
 
        for object in allds_objects:
            if object.id not in nobjectids:
                novel_not_contain_objects.append(object)
            else:
                novel_contain_objects.append(object)
        if len(novel_not_contain_objects) == len(allds_objects):
            skip_img_count += 1
            sh.copy(src_img, dst_img)
            continue
        ## 加载原始图片，保留至 novel 路径
        im = Image.open(src_img)
        draw = ImageDraw.Draw(im)
        if DEBUG_INFO:
            # print(all_objects)
            for obj in allds_objects:
                xy = obj.xy
                # draw.rectangle(xy, outline=(255, 0, 0))
                draw.text(xy[:2], obj.cls, (255, 0, 0))
            for nobject in novelds_objects:
                draw.rectangle(nobject.xy, outline=(255, 0, 0), width=2)
                draw.text(nobject.xy[:2], nobject.cls, (255, 128, 0))
        blured = False
        for pend_rm_object in novel_not_contain_objects:
            # 给未选中的 ins 进行 blur 处理
            iou = 0
            if pend_rm_object.cls in ["ship"]:
                continue
            no_overlap = True
            for nobject in novelds_objects:
                if pend_rm_object.id == nobject.id: continue
            

                iou = compute_iou(nobject.xy, pend_rm_object.xy)
                iou_dist.append(iou)
                # print(nobject, pend_rm_object, iou)
                if iou > 0.01:
                    no_overlap = False
                    break
            # 遮挡 非 novel object ( remove object)
            if no_overlap: 
                xy = pend_rm_object.xy
                xy = (xy[0] + 1, xy[1] + 1, xy[2] - 1, xy[3] - 1)
                blur_im = im.crop(xy)
                w = int(min(xy[2] - xy[0], xy[3] - xy[1]) / 8)
                w = max(14, w)
                # print(w)
                gbF = blur_im.filter(ImageFilter.GaussianBlur(radius=w))
                im.paste(gbF, xy)
                if DEBUG_INFO:
                    draw.rectangle(xy, outline=(0, 125, 125), width=3)
                    draw.text(xy[:2], pend_rm_object.cls, (255, 0, 0))
                masked_instance_count += 1
                blured = True
        if blured:
            im.save(dst_img)
            sh.copy(src_img, sdst_img)
        else:
            sh.copy(src_img, dst_img)

        nim = Image.new(im.mode, im.size, 0)
        imw, imh = im.size
        offset = 40
        dxy = (imw, imh, 0, 0)
       
        for nobj in novel_contain_objects:
            xy = list(nobj.xy)
            xy = tuple([max(xy[0] - offset, 0), max(xy[1] - offset, 0), 
                min(xy[2] + offset, imw), min(xy[3] + offset, imh)])
            dxy = (min(xy[0], dxy[0]), min(xy[1], dxy[1]),
                max(xy[2], dxy[2]), max(xy[3], dxy[3])
            )
        objimgarea = im.crop(dxy)
        nim.paste(objimgarea, dxy)
        nim.save(dst_nimg)
        
    print("Masked instance: ", masked_instance_count, "skip image: ", skip_img_count)
    hist, bins = np.histogram(iou_dist)
    print("iou_dist", hist, "\n", bins)
    remove_dumplicate_files(ReservedNovelAnnotationDir, NovelJPEGImagesDir)


def remove_dumplicate_files(annodir, imgdir):
    ## remove anno if image is filtered
    annos = sorted(os.listdir(annodir))
    annos = set([anno[:-4] for anno in annos])  #fileid
    images = os.listdir(imgdir)
    images = set([image[:-4] for image in images]) # fileid 
    nimages = set()
    for im in images:
        if im.endswith("r90") or im.endswith("r180") or im.endswith("r270"):
            continue
        if im.endswith("_mask"):
            continue
        if im.endswith("_unblur"):
            nimages.add(im[:-7])
            continue
        nimages.add(im)

    removed = annos - nimages

    for anno in removed:
        os.remove(osp.join(annodir, anno + ".xml"))
    images = os.listdir(imgdir)
    images = set([image[:-4] for image in images]) # fileid 
    nimages = []
    for im in images:
        if im.endswith("r90") or im.endswith("r180") or im.endswith("r270"):
            continue
        if im.endswith("_mask"):
            continue
        nimages.append(im)
    images = list(sorted(os.listdir(imgdir)))
    # remove_images = []
    # print(nimages)
    # for im in images:
    #     imname = im[:-4]
    #     print(imname, imname in nimages)
    #     if imname not in nimages:
    #         remove_images.append(imname)
    print("Size: anno", len(list(os.listdir(annodir))), len(nimages))
    print("Please move seed files")

def generate_seeds(annodir, save_path, shot=10):
    """生成 Seed 文件
    """
    annotationDir = osp.join(BASE_DIR, annodir)
    ds = Dataset(annotationDir, annodir, "JPEGImages") # image dir 不重要
    annos = ds.annos.values()
    os.makedirs(save_path, exist_ok=True)
    # print(annos)
    for cls in VOC_CLASSES:
        dst_annos = []
        for anno in annos:
            if cls not in anno.clsset:
                continue
            dst_annos.append(anno)
        filename = f'box_{shot}shot_{cls}_train.txt'
        # print(cls, dst_annos)
        names = [x.img_file for x in dst_annos]
        loc = osp.join(save_path, filename)
        with open(loc, 'w') as fp:
            fp.write('\n'.join(names))

def merge_annotations():
    """合并 不同 cls 的相同文件
    """
    annodict = {}
    total_file_count = 0
    dst = osp.join(BASE_DIR, NovelAnnotationDirName)
    if osp.exists(dst):
        sh.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    if osp.exists(NovelAnnotationDir):
        sh.rmtree(NovelAnnotationDir)
    os.makedirs(NovelAnnotationDir, exist_ok=True)
    if osp.exists(NovelJPEGImagesDir):
        sh.rmtree(NovelJPEGImagesDir)
    os.makedirs(NovelJPEGImagesDir, exist_ok=True)

    for cls in VOC_CLASSES:
        clsdir = f"{NovelAnnotationDirPrefix}{cls}"
        annodir = osp.join(BASE_DIR, clsdir)
        if not osp.exists(annodir):
            print("Not exist", annodir)
            continue
        for file in sorted(os.listdir(annodir)):
            total_file_count = total_file_count + 1
            fileid = file[:-4]
            if fileid not in annodict:
                anno = VocAnnotation(fileid, clsdir, autoid=False)
                annodict[fileid] = anno
            else:
                anno = annodict[fileid]
                anno.parseInsert(fileid, clsdir)
    print(len(annodict), "/", total_file_count)
    
    
    for fileid, anno in annodict.items():
        name = osp.join(BASE_DIR, NovelAnnotationDirName, fileid + ".xml")
        anno.save(name)
    pass

if __name__ == "__main__":
    merge_annotations()
    main(USE_SHOT) # shot
    # 
    # remove_dumplicate_files(osp.join(BASE_DIR, "NovelAnnotations_0330"), osp.join(BASE_DIR, "NovelJPEGImages_0330"))
    generate_seeds(ReservedNovelAnnotationDirName, osp.join(BASE_DIR, "split1", "seed26"), USE_SHOT)
