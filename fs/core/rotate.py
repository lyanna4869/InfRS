
from PIL import Image, ImageDraw
import cv2
import os.path as osp
import numpy as np
import os
from fsdet.structures import BoxMode
def rotate_bbox(bbox, angle, cxcy, hw):
    """Rotate the bounding box.
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    angle : float
        angle by which the image is to be rotated
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
    h : int 
        height of the image
    w : int 
        width of the image
    Returns
    -------
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y1 x2 y2 x1 y2`
    """
    cx, cy = cxcy
    h, w = hw
    corners = np.asarray((bbox[0], bbox[1], bbox[2], bbox[1], 
                          bbox[2], bbox[3], bbox[0], bbox[3]))
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T
    
    calculated = calculated.reshape(8)
    
    r = np.asarray((max(calculated[0], 0), max(calculated[1], 0), 
        min(calculated[4], nW), min(calculated[5], nH) ))
    r = np.round(r, 2)
    assert np.sum(r >= 0) == 4, f"{r}, {bbox}, {angle}"
    return r

def rotate_instances(annotations, step=1, img_dir="JPEGImages"):
    """ 
    旋转图片，以及 bouding box
    img_dir: full path
    [ {"file_name": "", "annotations": [ {"category_id": 0, "bbox_mode": 0, "bbox": []} ] 
    } ]
    """
    new_annos = []
    
    new_annos.extend(annotations)
    # return new_annos
    def add_mask(t):
        for anno in annotations:
            fileid = anno["image_id"] + t
            jpeg_file = os.path.join(img_dir, fileid + ".jpg")
            if not osp.exists(jpeg_file):
                continue
            nanno = {}
            nanno.update(anno)
            nanno["image_id"] = fileid
            nanno["file_name"] = jpeg_file
            new_annos.append(nanno)
        # print("*******", len(new_annos))
    if True:
        add_mask("_mask")
        # add_mask("_unblur")
    ran = slice(0, len(annotations), step)
    for anno in annotations[ran]:
        fileid = anno["image_id"]
        im = Image.open(anno["file_name"])
        for angle in [ 90, 180, 270]:
            nim = im.rotate(angle, expand=True)
            nfileid = f"{fileid}_r{angle}"
            # anno_file = os.path.join(dirname, NovelAnnoDirName, nfileid + ".xml")
            jpeg_file = os.path.join(img_dir, nfileid + ".jpg")
            
            annos = anno["annotations"]
            instances = []
            center = (int(anno["width"] / 2), int(anno["height"]/2))
            hw = (anno["height"], anno["width"],)
            for instance in annos:
                bbox = instance["bbox"]
                bbox = rotate_bbox(bbox, angle, center, hw)
                # imdraw = ImageDraw.Draw(nim)
                # imdraw.rectangle(bbox.tolist(), outline=(255, 0, 0))
                iid = instance.get("oid", 0)
                instances.append({
                    "category_id": instance["category_id"],
                    "bbox": bbox.tolist(),
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "oid": iid
                })
            if not osp.exists(jpeg_file) or True:
                nim.save(jpeg_file)
            # nim.save(f"tmp/{nfileid}.jpg")
            if angle == 180:
                height, width = hw
            else:
                width, height = hw
            nanno = {
                "file_name": jpeg_file,
                "image_id": nfileid,
                "height": height,
                "width": width,
                "annotations": instances
            }
            new_annos.append(nanno)
    return new_annos
import math

def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    
    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy

def rotate_bboxv2(bbox, angle, center):
    tl = rotate_around_point_highperf((bbox[0], bbox[1]), angle * math.pi / 180, center )
    lb = rotate_around_point_highperf((bbox[0], bbox[3]), angle * math.pi / 180, center )
    rb = rotate_around_point_highperf((bbox[2], bbox[3]), angle * math.pi / 180, center )
    tr = rotate_around_point_highperf((bbox[2], bbox[1]), angle * math.pi / 180, center )
    xmin = min(tl[0], lb[0], rb[0], tr[0])
    ymin = min(tl[1], lb[1], rb[1], tr[1])
    xmax = max(tl[0], lb[0], rb[0], tr[0])
    ymax = max(tl[1], lb[1], rb[1], tr[1])
    return (xmin, ymin, xmax, ymax)
    
