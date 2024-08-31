import os
import seaborn as sn
import numpy as np
import torch
import matplotlib.pyplot as plt
from .iou import box_iou, box_iou_np
import os.path as osp

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    """
    nc foreground class
    """
    def __init__(self, nc, confThresh=0.25, iouThres=0.45):
        self._matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.confThresh = confThresh
        self.iouThres = iouThres
        self._aparr = None

    def process_cls(self, classify, labels):
        """
        classify  np  , contains all dataset images
        T( [score, class], )
        """
        x = classify[:, 0] > self.confThresh # fg
        nclassify = classify[x]
        fg_classes = labels[x].astype(np.int32)
        detection_classes = nclassify[:, 1].astype(np.int32)
        
        for i, (dc, gc) in enumerate(zip(detection_classes, fg_classes)):
            self._matrix[dc, gc] += 1

        x = classify[:, 0] < self.confThresh # fg
        nclassify = classify[x]
        # bg_classes = labels[x].astype(np.int32)
        detection_classes = nclassify[:, 1].astype(np.int32)
        for dclsidx, dc in enumerate(detection_classes):
            self._matrix[dc, self.nc] += 1 # FP


    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, confThresh, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.confThresh]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4]) # T[M, N]

        x = torch.where(iou > self.iouThres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy() # x y v
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]] # score >>
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self._matrix[gc, detection_classes[m1[j]]] += 1  # correct
            else:
                self._matrix[gc, self.nc] += 1  # FN  object -> background 

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self._matrix[self.nc, dc] += 1  # background FN
    
    def matrix(self):
        return self._matrix

    def plot(self, save_dir='', names=()):
        if len(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        names = list(names)
        array = self._matrix / (self._matrix.sum(axis=0).reshape(1, self.nc + 1) + 1E-6)  # normalize
        array[array < 0.001] = np.nan  # don't annotate (would appear as 0.00)
        self._aparr = array
        fig = plt.figure(figsize=(12, 9), tight_layout=True)
        sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
        labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
        xlabels = names + ['bg(FN)'] if labels else "auto"
        ylabels = names + ['bg(FP)'] if labels else "auto"
        sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, \
                vmin=0, vmax=1, cmap='Blues', fmt='.2f', 
                square=True, xticklabels= xlabels, yticklabels= ylabels) \
            .set_facecolor((1, 1, 1))
        fig.axes[0].set_ylabel('GroundTruth')
        fig.axes[0].set_xlabel('Predicted')
        try:
            fig.savefig(osp.join(save_dir, 'confusion_matrix.png'), dpi=250)
        except Exception as e:
            print("Exception when saving", e)

    def debug(self):
        # print("\n******************" )
        # for i in range(self.nc + 1):
        #     for v in self._matrix[i]:
        #         print(f'{int(v):6d}', end='')
        #     print('\n')
        # print("\n******************" )
        if self._aparr is None:
            return
        mean_ap = 0
        for i in range(self.nc):
            a = self._aparr[i, i]
            mean_ap += a
            print(f"{a*100:-8.2f}", end='')
        print()
        print("Map: ", mean_ap / self.nc * 100, "%")


def calcHBBIou(obb_gt, obb_det):
    """
    obb_gt: A[M, 4]
    obb_det: A[N, 4]

    return A[M, N]
    """
    hbb_det = np.zeros((len(obb_det), 4))
    hbb_gt = np.zeros((len(obb_gt), 4))
    for idx, bb in enumerate(obb_det):
        hbb_det[idx, 0] = np.min(bb[0::2])
        hbb_det[idx, 1] = np.min(bb[1::2])
        hbb_det[idx, 2] = np.max(bb[0::2])
        hbb_det[idx, 3] = np.max(bb[1::2])
    for idx, bb in enumerate(obb_gt):
        hbb_gt[idx, 0] = np.min(bb[0::2])
        hbb_gt[idx, 1] = np.min(bb[1::2])
        hbb_gt[idx, 2] = np.max(bb[0::2])
        hbb_gt[idx, 3] = np.max(bb[1::2])
        
    iou = box_iou_np(hbb_gt, hbb_det)
    return iou
