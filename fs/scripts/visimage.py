__version__ = "1.0"

"""
1.0 moved from bff
1.1 add id for dota object
1.2 visiualize bbox voc
1.3 move class into fs.core.imgviser
"""
from fs.core.imgviser import *

def main(args = None):
    from argparse import ArgumentParser
    parser = ArgumentParser("visdota")
    parser.add_argument("--dir", default="dota_data", type=str, help="Base Dir without images or labelTxt")
    parser.add_argument("--ext", type=str, default=".png", help="image extension")
    parser.add_argument("--type", type=str, default="dota", choices=["dota", "voc", "coco"], help="Dataset type")
    parser.add_argument("--labeldir", type=str, default=None)
    parser.add_argument("--imagedir", type=str, default=None)
    # single image only
    parser.add_argument("--name", type=str, help="If not output, it should be a image name, or it is None")
    # directory only
    parser.add_argument("--output", type=str, default=None, help="Not show image with pillow but save into disk")

    args = vars(parser.parse_args(args))
    Visualizer.ImgExt=args["ext"]

    directory = args["dir"]
    print(f"Working on {directory}")
    name = args["name"]
    t = args["type"].lower()

    print("Using Dataset Type", t)
    # if args["output"] is Non:
    ld = args["labeldir"]
    if t == "coco":
        visualizer = CocoVisualizer(directory, JSON_PATH)
        visualizer.vis(int(name))
    elif t == "voc":
        visualizer = VocVisualizer(directory)
        args["labeldir"] = "Annotations" if ld is None else ld
        visualizer._imgdir = args["imagedir"]
        visualizer._labeldir = args["labeldir"]

        if args["output"]:
            visualizer.output(args["output"])
        else:
            visualizer.vis(name)
    else:
        visualizer = DotaVisualizer(directory)
        args["labeldir"] = "labelTxt" if ld is None else ld
        if args["output"]:
            visualizer._labeldir = args["labeldir"]
            visualizer.output(args["output"])
        else:
            visualizer.vis(name)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = ["--type", "voc", "--dir", "datasets/NWPU/", "--labeldir",
            "Annotations", "--imagedir", "JPEGImages", 
            "--output", "bboximages", ]
    main(args)