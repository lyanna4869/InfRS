import torch
import os
import argparse


def parse_args(args = None):
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--src2', type=str, default='',
                        help='Path to the secondary checkpoint (for combining)')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Save directory')
    # Surgery method
    parser.add_argument('--method', choices=['combine', 'remove', 'randinit'],
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
                            'roi_heads.box_predictor.bbox_pred'],
                        help='Target parameter names')
    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')
    # Dataset
    parser.add_argument('--coco', action='store_true',
                        help='For COCO models')
    parser.add_argument('--lvis', action='store_true',
                        help='For LVIS models')
    parser.add_argument('--voc', action='store_true',
                        help='For voc models')
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
    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'cls_score' in param_name:
            prev_cls -= 1
        avg = None
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)
            if 'cls_score' in param_name:
                avg = torch.mean(pretrained_weight[:-1])
            else:
                avg = torch.mean(pretrained_weight)
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]
        # if avg is not None:
        #     new_weight[prev_cls:] = avg
        print(weight_name)

        if 'cls_score' in param_name:
            new_weight[-1] = pretrained_weight[-1]  # bg class
        ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)


def surgery_loop(args, surgery):
    # Load checkpoints
    ckpt = torch.load(args.src1)
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

    # Remove parameters
    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['model'][param_name + '.weight']
            if param_name+'.bias' in ckpt['model']:
                del ckpt['model'][param_name+'.bias']
        save_ckpt(ckpt, save_path)
        return

    # Surgery
    tar_sizes = [
        TAR_SIZE + 1, 
        TAR_SIZE * 4
        ]
    for idx, (param_name, tar_size) in enumerate(zip(args.param_name,
                                                     tar_sizes)):
        surgery(param_name, True, tar_size, ckpt, ckpt2)
        surgery(param_name, False, tar_size, ckpt, ckpt2)

    # Save to file
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


if __name__ == '__main__':
    args = None
    # args = [
    #     "--src1", "weights/model_0015999.pth",
    #     "--method", "randinit", 
    #     "--save-dir", "work_dirs/fsce_resnet50_base1"
    # ]
    args = parse_args(args)

    BASE_CLASSES = [c for c in range(7)]
    ALL_CLASSES = [c for c in range(10)]
    if args.voc:
        TAR_SIZE = 20
    else:
        TAR_SIZE = 10

    ckpt_surgery(args)
