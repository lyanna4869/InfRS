import os, os.path as osp, sys
from argparse import ArgumentParser
def main(root: "str", labelDir):
    fileids = []
    for file in sorted(os.listdir(osp.join(root, labelDir))):
        fileid, ext = osp.splitext(file)
        fileids.append(fileid)
    dst_root = osp.join(root, "ImageSets", "Main")
    os.makedirs(dst_root, exist_ok=True)
    with open(osp.join(dst_root, "trainval.txt"), "w") as f:
        f.write("\n".join(fileids))

if __name__ == "__main__":
    sargs = sys.argv[1:]
    parser = ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--label", type=str, default="labelTxt")

    args = vars(parser.parse_args(sargs))
    main(args["dir"], args['label'])