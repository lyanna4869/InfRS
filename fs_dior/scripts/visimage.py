__version__ = "1.0"

"""
1.0 moved from bff
1.1 add id for dota object
1.2 visiualize bbox voc
1.3 move class into fs.core.imgviser
"""
from fs.core.imgviser import *
from fs_dior.core.category import ALL_CATEGORIES
from fs_dior.core.category import CLASS_COLOR
VOC_CLASSES = ALL_CATEGORIES[1]


class ResultViser(Visualizer):
    # parse data from voc results
    # 003 0.981 580.3 143.0 671.5 235.0
    def __init__(self, labelDir, imageDir) -> None:
        self._labeldir = labelDir
        self._imgdir = imageDir
        self.predictions = {} # key is image id, value is array of object
        self.score_thresh = 0.5
        
    def get_result_from_labeldir(self):
        for cls in VOC_CLASSES:
            self.parse(cls)

    def parse(self, cls):
        with open(osp.join(self._labeldir, f"{cls}.txt")) as f:
            for line in f:
                line = line.strip()
                if len(line) == 0 : continue

                line = line.split(" ")
                imgid = line[0]
                if imgid not in self.predictions:
                    self.predictions[imgid] = []
                img = self.predictions[imgid]
                score_bonus = float(line[1])
                if score_bonus>=1:
                    score = score_bonus - 0.1
                else:
                    score = score_bonus
                if score < self.score_thresh:
                    continue
                pred = {
                    "score": score,
                    "bbox": list(map(float, line[2:])),
                    "cls": cls,
                    "name": f"{cls} {min(score, 1.0):.3f}",
                    # "name": f"{cls}",
                }
                img.append(pred)

    def _output(self, imgfile, objects):
        imgloc = osp.join(self._imgdir, imgfile)
        assert osp.exists(imgloc), imgloc
        image = cv2.imread(imgloc)  # 读取图像
        for idx, object in enumerate(objects):
            color = CLASS_COLOR.get(object["cls"])
            # color = (0, 0, 225)
            bbox = object["bbox"]
            poly = np.array(bbox, dtype=np.int32)
            # poly = poly.reshape((-1, 1, 2))
            cv2.rectangle(image, poly[:2], poly[2:], color, 2, lineType=cv2.LINE_AA)
            # cv2.putText(image, object["name"], poly[:2], cv2.FONT_HERSHEY_SIMPLEX, color=color, fontScale=0.5)
            draw_text(image, object["name"], poly[:2], color=color, font_scale=1, text_color_bg=(255, 255, 255))
        cv2.imwrite(osp.join(self._outdir, imgfile), image)

    def output(self, dst):
        os.makedirs(dst, exist_ok=True)
        self._outdir = dst
        self.get_result_from_labeldir()
        for imgid, objects in tqdm.tqdm(self.predictions.items()):

            self._output(f"{imgid}{self.ImgExt}", objects)

def main(args = None):
    from argparse import ArgumentParser
    parser = ArgumentParser("visdota")
    parser.add_argument("--ext", type=str, default=".jpg", help="image extension")
    parser.add_argument("--src", type=str, default="/home/lwz/prj/fsce/checkpoints/dior/split1_fpn/10shot_inc_clp", help="classification txt")
    parser.add_argument("--image", type=str, default="/home/lwz/prj/fsce/datasets/DIOR/JPEGImages", help="classification txt")
    parser.add_argument("--dst", type=str, default=None, help="classification txt")
    parser.add_argument("--thresh", type=float, default=0.05, help="score thresh")
    # single image only
    args = vars(parser.parse_args(args))
    Visualizer.ImgExt=args["ext"]

    visualizer = ResultViser(osp.join(args['src'], "results"),
                             args['image']
                             )
    visualizer.score_thresh = args['thresh']
    if args['dst'] is None:
        dst = osp.join(args['src'], "image_results")
    else:
        dst = args['dst']
    visualizer.output(dst)
    # visualizer = ResultViser("checkpoints/fs/pascal_voc_eval(cpe)", "datasets/DIOR/Images")
    # visualizer.output("checkpoints/results/dior/res101_10shot_cpe1")

if __name__ == "__main__":
    args = sys.argv[1:]
    # if len(args) == 0:
    #     args = ["--type", "voc", "--dir", "datasets/DIOR/", "--labeldir",
    #         "Annotations", "--imagedir", "JPEGImages", 
    #         "--output", "bboximages", ]
    main(args)