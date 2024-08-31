def compute_iou_for_obj1(xy1, xy2):
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
    return (intersect / union_area) * 1.0