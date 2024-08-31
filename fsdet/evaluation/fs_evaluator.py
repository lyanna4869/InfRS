from .pascal_voc_evaluation import PascalVOCDetectionEvaluator, logger, comm, voc_eval, create_small_table
from collections import defaultdict, OrderedDict
import os, os.path as osp
import numpy as np
from fsdet.config import globalvar as gv
from datetime import datetime

class FsDetectionEvaluator(PascalVOCDetectionEvaluator):
    """bsf.c 该 Detection 修改几处地方，方便进行 FS 调试
    """
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        now = datetime.now()

        self.set_output_dir_name(f"/dev/shm/checkpoints/fs/eval_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}")

    def process(self, inputs: "list", outputs: "list"):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            if gv.rcnn_inference_post_process:
                ## 默认输出
                instances = output['instances'].to(self._cpu_device) # output is dict
            else:
                instances = output.to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for box, score, cls in zip(boxes, scores, classes):
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )
    def set_output_dir_name(self, dirname):
        self.dirname = dirname
        os.makedirs(dirname, exist_ok=True)

    
    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                predictions[clsid].extend(lines)
        del all_predictions
        

        ## 不使用 tempfile
        # with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
        res_file_template = os.path.join(self.dirname, "{}.txt")

        aps = defaultdict(list)  # iou -> ap per class
        aps_base = defaultdict(list)
        aps_novel = defaultdict(list)
        exist_base, exist_novel = False, False
        for cls_id, cls_name in enumerate(self._class_names):
            lines = predictions.get(cls_id, [""])

            with open(res_file_template.format(cls_name), "w") as f:
                f.write("\n".join(lines))

            for thresh in range(50, 100, 5):
                rec, prec, ap = voc_eval(
                    res_file_template,
                    self._anno_file_template,
                    self._image_set_path,
                    cls_name,
                    ovthresh=thresh / 100.0,
                    use_07_metric=self._is_2007,
                )
                aps[thresh].append(ap * 100)

                if self._base_classes is not None and cls_name in self._base_classes:
                    aps_base[thresh].append(ap * 100)
                    exist_base = True

                if self._novel_classes is not None and cls_name in self._novel_classes:
                    aps_novel[thresh].append(ap * 100)
                    exist_novel = True

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            ret["bbox"].update(
                {"bAP": np.mean(list(mAP_base.values())), "bAP50": mAP_base[50],
                 "bAP75": mAP_base[75]}
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            ret["bbox"].update({
                "nAP": np.mean(list(mAP_novel.values())), "nAP50": mAP_novel[50],
                "nAP75": mAP_novel[75]
            })

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}

        logger.info("Evaluate per-class mAP50:\n"+create_small_table(per_class_res))
        logger.info("Evaluate overall bbox:\n"+create_small_table(ret["bbox"]))
        return ret
    