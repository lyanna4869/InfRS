import torch
import os
import argparse
import copy
TAR_SIZE = 4

def parse_args(args):
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--src2', type=str, default='',
                        help='Path to the secondary checkpoint (for combining)')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Save directory')
    # Surgery method
    parser.add_argument('--method', choices=['combine', 'remove', 'randinit', "compare"],
                        required=True,
                        help='Surgery method. combine = '
                             'combine checkpoints. remove = for fine-tuning on '
                             'novel dataset, remove the final layer of the '
                             'base detector. randinit = randomly initialize '
                             'novel weights.')
    # Targets
    parser.add_argument('--param-name', type=str, nargs='+',
                        default=[
                            'roi_heads.box_predictor.cls_score',
                            'roi_heads.box_predictor.cls_score',
                            'roi_heads.box_predictor.bbox_pred',
                            'roi_heads.box_predictor.bbox_pred',
                            # 'bbox_head.odm_reg'
                            ],
                        help='Target parameter names')
    parser.add_argument('--copy-param-name', type=str, nargs='+',
                        default=[
                            'roi_heads.box_head.fc1',
                            'roi_heads.novel_box_head.fc1',
                            'roi_heads.box_head.fc2',
                            'roi_heads.novel_box_head.fc2',
                            'roi_heads.box_predictor.cls_score',
                            'roi_heads.redetector.cls_score',
                            'roi_heads.box_predictor.bbox_pred',
                            'roi_heads.redetector.bbox_pred',
                        ])
    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')
  
    args = parser.parse_args(args)
    return args

def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.

    Note: The base detector for LVIS contains weights for all classes, but only
    the weights corresponding to base classes are updated during base training
    (this design choice has no particular reason). Thus, the random
    initialization step is not really necessary.
    """
    def surgery(param_names, tar_size, ckpt, ckpt2=None):
        print("dev: ",  param_names)
        old_param_name = param_names[0]
        new_param_name = param_names[1]
        for is_weight in (True, False):
            weight_name = old_param_name + ('.weight' if is_weight else '.bias')
            pretrained_weight = ckpt['model'][weight_name]
            prev_cls = pretrained_weight.size(0)
            if 'cls_score' in old_param_name:
                prev_cls -= 1
            
            new_weight_name = new_param_name + ('.weight' if is_weight else '.bias')
            # print(weight_name, " shape: ", pretrained_weight.shape)
            avg = None
            if is_weight:
                feat_size = pretrained_weight.shape[1:]
                new_weight = torch.rand((tar_size, *feat_size)).to(pretrained_weight.device)
                torch.nn.init.normal_(new_weight, 0, 0.01) 
            else:
                # avg = torch.mean(pretrained_weight)
                # print(avg)
                new_weight = torch.zeros(tar_size).to(pretrained_weight.device) # + avg
                if 'cls_score' in old_param_name:
                    avg = torch.mean(pretrained_weight[:-1])
                else:
                    avg = torch.mean(pretrained_weight)
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
            # if avg is not None:
            #     new_weight[prev_cls:] = avg
            if 'cls_score' in old_param_name:
                new_weight[-1] = pretrained_weight[-1]  # bg class
            print(f"New Weight {new_weight_name} Shape: ", pretrained_weight.shape, 
                  "  -->> ", new_weight.shape)
            ckpt['model'][new_weight_name] = new_weight
        print("\n")
    surgery_loop(args, surgery)


def combine_ckpts(args):
    """
    Combine base detector with novel detector. Feature extractor weights are
    from the base detector. Only the final layer weights are combined.
    """
    def surgery(param_name, tar_size, ckpt, ckpt2=None):
        print("dev: ",  param_name)
        for is_weight in (True, False):
            if not is_weight and param_name + '.bias' not in ckpt['model']:
                return
            weight_name = param_name + ('.weight' if is_weight else '.bias')
            pretrained_weight = ckpt['model'][weight_name]
            prev_cls = pretrained_weight.size(0)
            feat_size = pretrained_weight.shape[1:]
            if is_weight:
                # feat_size = pretrained_weight.size(1)
                new_weight = torch.rand((tar_size, *feat_size))
            else:
                new_weight = torch.zeros(tar_size)
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
            ckpt2_weight = ckpt2['model'][weight_name]
            prev_cls = ckpt2_weight.size(0)
            print(ckpt2_weight.shape, new_weight.shape)
            
            if 'cls_score' in param_name:
                new_weight[prev_cls:-1] = ckpt2_weight[:-1]
                new_weight[-1] = pretrained_weight[-1]
            else:
                new_weight[:prev_cls] = ckpt2_weight
            ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)

def compare(args):
    def surgery(param_names, tar_size, ckpt, ckpt2=None):
        print("dev: ",  param_names)
        param_name = param_names[0]
        state_dict = ckpt['model']
        for is_weight in (True, False):
            weight_name = param_name + ('.weight' if is_weight else '.bias')
            new_weight_name = param_names[1] + ('.weight' if is_weight else '.bias')
            pretrained_weight = state_dict[weight_name]
            new_pretrained_weight = state_dict[new_weight_name]

            prev_cls = pretrained_weight.size(0)
            print(torch.mean(pretrained_weight - new_pretrained_weight[:prev_cls]))

    compare_surgery_loop(args, surgery, False)

def compare_surgery_loop(args, surgery, save=True):
    # Load checkpoints
    ckpt = torch.load(args.src1)
    # state = ckpt["state_dict"]
    # for name in state.keys():
    #     print(name)
    # return
    if args.method == 'combine':
        ckpt2 = torch.load(args.src2)
        save_name = args.tar_name + '_combine.pth'
    else:
        ckpt2 = None
        save_name = args.tar_name + '_' + \
            ('remove' if args.method == 'remove' else 'surgery') + '.pth'
    if args.save_dir == '':
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    # Surgery
    param_names = [
        ("bbox_head.fam_cls", "novel_bbox_head.fam_cls"), 
        ("bbox_head.odm_cls", "novel_bbox_head.odm_cls"), 
    ]

    tar_sizes = [TAR_SIZE, TAR_SIZE]
    for idx, (param_name, tar_size) in \
            enumerate(zip(param_names, tar_sizes)):
        # print(param_name)
        surgery(param_name, tar_size, ckpt, ckpt2)

def copy_surgery(ckpt, param_names):
    for idx, (pn1, pn2) in enumerate(param_names):
        for is_weight in (True, False):
            # if not is_weight and param_name + '.bias' not in ckpt['model']:
            #     continue
            pname1 = pn1 + ('.weight' if is_weight else '.bias')
            pname2 = pn2 + ('.weight' if is_weight else '.bias')

            ckpt['model'][pname2] = copy.deepcopy(ckpt['model'][pname1])

def surgery_loop(args, surgery: "callable"):
    # Load checkpoints
    ckpt = torch.load(args.src1)
    # state = ckpt["state_dict"]
    # for name in state.keys():
    #     print(name)
    # return

    ckpt2 = None
    save_name = args.tar_name + '_' + \
        ('remove' if args.method == 'remove' else 'surgery') + '.pth'
    if args.save_dir == '':
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    ## copy 
    copy_param_names = args.copy_param_name
    copy_param_names = [(x, y) for x, y in zip(copy_param_names[::2], copy_param_names[1::2])]
    copy_surgery(ckpt, copy_param_names)
    # Surgery
    param_names = args.param_name
    ## bsf.c name to another
    param_names = [(x, y) for x, y in zip(param_names[::2], param_names[1::2])]

    print("===============")
    # return
    tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
    for idx, (param_name, tar_size) in enumerate(zip(param_names, tar_sizes)):
        # print(param_name)
        surgery(param_name, tar_size, ckpt, ckpt2)
    save_ckpt(ckpt, save_path)


def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))


def reset_ckpt(ckpt):
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0
    if 'meta' not in ckpt:
        return
    m = ckpt['meta']
    if 'CLASSES' in m:
        m.pop("CLASSES")

    if 'epoch' in m:
        m['epoch'] = 0

    if 'iter' in m:
        m['iter'] = 0


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if len(args) == 0:

        args = [
            # "--src1", "weights/RSOD_R101_split1_anchor32.pth",
            "--src1", "weights/RSOD_R101_split1.pth",
            
            "--method", "randinit", 
            "--save-dir", "work_dirs/rsod_resnet101_base1_all_redetect/"
        ]
        args = [
           "--src1", "weights/RSOD_R101_split2.pth",
           "--method", "randinit", 
           "--save-dir", "work_dirs/rsod_resnet101_base2_all_redetect/"
        ]
        # args = [
        #     "--src1", "weights/RSOD_R101_split3.pth",
        #     "--method", "randinit", 
        #     "--save-dir", "work_dirs/rsod_resnet101_base3_all_redetect/"
        # ]
    args = parse_args(args)

    if args.method == 'combine':
        combine_ckpts(args)
    elif args.method == "compare":
        compare(args)
    else:
        ckpt_surgery(args)

