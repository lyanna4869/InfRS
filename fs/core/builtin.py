def extend_metasplits(METASPLITS, prefix, dirname, dataset='nwpu'):
    """
        name: nwpu_trainval_novel1_10shot_seedx
        split: all_10shot_split_1_trainval
    """
    for sid in range(1, 4):  # dataset idx
        for shot in [1, 2, 3, 5, 10]:
            for seed in range(10):
                seed = '' if seed == 0 else f'_seed{seed}'
                name = f"{dataset}_trainval_{prefix}{sid}_{shot}shot{seed}"
                split = f"{prefix}_{shot}shot_split_{sid}_trainval"
                if prefix == 'all':
                    keepclasses = f"base_novel_{sid}" 
                else:
                    keepclasses = f"novel{sid}"
                METASPLITS.append(
                    (name, dirname, split, keepclasses, sid))
from collections import namedtuple
DatasetSplit = namedtuple('DatasetSplit', ['name', 'dir', 'split', 'keep_classes', 'sid'])
def extend_metasplits_split(prefix, sid, default_dirname, shot, seed = 0, dataset='dior'):
    """ sid: split id
        name: dior_trainval_novel1_10shot_seedx
        split: all_10shot_split_1_trainval
    """
    # for shot in [1, 2, 3, 5, 10, 20]:
    #     for seed in range(10):
    seed = '' if seed == 0 else f'_seed{seed}'
    name = f"{dataset}_trainval_{prefix}{sid}_{shot}shot{seed}"
    split = f"{prefix}_{shot}shot_split_{sid}_trainval"
    if prefix == 'all':
        keepclasses = f"base_novel_{sid}" 
    else:
        keepclasses = f"novel{sid}"
    return (DatasetSplit(name, default_dirname, split, keepclasses, sid))

def extend_morebase(WITH_MORE_BASE, prefix, sid, dirname):
    """
    sid: split id
    """
    ploidy = 3
    for shot in [1, 2, 3, 5, 10]:
        for seed in range(100):
            seed = '' if seed == 0 else f'_seed{seed}'
            name = f"nwpu_trainval_{prefix}{sid}_{shot}shot{seed}_{ploidy}ploidy"
            split = f"{prefix}_{shot}shot_split_{sid}_trainval"
            if prefix == 'all':
                keepclasses = f"base_novel_{sid}"
            else:
                keepclasses = f"novel{sid}"
            WITH_MORE_BASE.append(
                (name, dirname, split, keepclasses, sid))
