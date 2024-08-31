import numpy as np


def _bbox_iou_np_single(box1: "np.ndarray", box2: "np.ndarray", x1y1x2y2=True, eps=1e-7):
    """
     Returns the IoU of box1 to box2. box1 is 4, box2 is 4
    """
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    it1 = min(b1_x2, b2_x2) - max(b1_x1, b2_x1)
    it2 = min(b1_y2, b2_y2) - max(b1_y1, b2_y1)
    if it1 <= 0 or it2 <= 0:
        return 0, 1
    inter = it1 * it2 + eps

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    return inter, union

def bbox_iou_np_single(box1, box2, x1y1x2y2=True, eps=1e-7):
    """
     Returns the IoU of box1 to box2. box1 is 4, box2 is 4
    """
    # Get the coordinates of bounding boxes
    inter, union =_bbox_iou_np_single(box1, box2, x1y1x2y2, eps)
    return inter / union  # IoU


def hbb_iou(box1, box2):
    """
    Calculates the IoU between two HBB bounding boxes.
    box1: array-like object of shape (4,). The coordinates of the first bounding box in the order (xmin, ymin, xmax, ymax).
    box2: array-like object of shape (4,). The coordinates of the second bounding box in the order (xmin, ymin, xmax, ymax).
    Returns: float. The IoU between the two bounding boxes.
    """
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of the HBB bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou1 = intersection_area / float(box1_area)
    iou2 = intersection_area / float(box2_area)
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou, iou1, iou2