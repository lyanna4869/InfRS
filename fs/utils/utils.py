import torch
import random
import numpy as np
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True
import datetime, os, os.path as osp
def check_save_git_info(root: "str", force_commit=False):
    import git, logging
    logger = logging.getLogger("fsdet.git")
    repo = git.Repo(root)
    if not repo.is_dirty():
        return
    
    if not force_commit:
        while True:
            resp = input("[Git] Do you want to make an automated commit? (yes/no) ")
            if resp in ["y", "yes"]:
                break
            elif resp in ["no", "n"]:
                raise ValueError("You must keep repo clean by using: 'git commit' to clean unstaged objects")
            elif resp in ["skip"]:
                return
    changedFiles = [item.a_path for item in repo.index.diff(None)]
    exist_files = list(filter(lambda x: osp.exists(x), changedFiles)) 
    
    delete_files = list(filter(lambda x: not osp.exists(x), changedFiles))
    if len(exist_files):
        repo.index.add(items=exist_files)
    if len(repo.untracked_files):
        repo.index.add(items=repo.untracked_files)
    if len(delete_files):
        repo.index.remove(items=delete_files)
    count = 0
    count_dst = osp.join(root, ".git/bf_auto_git.txt")
    if osp.exists(count_dst):
        with open(count_dst, "r") as f:
            count = int(f.read())
    count += 1
    with open(count_dst, "w") as f:
        f.write(f"{count}")
    com = repo.index.commit(f'auto commit [{count}] for train val stage')
    sha = com.hexsha
    logger.info(f'auto commit [{count}] for train val stage: {sha}')
from fsdet.utils.img import *
import pycocotools.mask as maskUtils

        
def get_pred_from_cls_score(cls_score: "torch.Tensor"):
    cs = cls_score.sigmoid()
    max_scores, _ = cs.max(dim=-1)
    _, topk_inds = max_scores.topk(1000)
    scores = cs[topk_inds, :]
    pred_cls = scores.argmax(-1)

from time import sleep 
#
from tqdm import tqdm
class MyProcessPool():
    def __init__(self, size = 4) -> None:
        self.size = size
        assert size > 0
        self._processes = []
        self._close = False
        self.idx = 0
        self._running = 0

    def append(self, process):
        if self._close:
            return
        self._processes.append(process)

    def _start(self):
        if self.idx >= len(self._processes):
            return
        p = self._processes[self.idx]
        self.idx += 1
        self._running += 1
        p.start()
        return p

    def start(self, show_bar=False):
        waitting = []
        if show_bar:
            bar = tqdm(total=len(self._processes), desc="Waiting for all process done")
        while self.idx < len(self._processes):
            for i in range(self._running, self.size):
                if self.idx >= len(self._processes):
                    continue
                p = self._start()
                waitting.append(p)
            
            while len(waitting) == self.size:
                alives = []
                for w in waitting:
                    w.join(1)
                    if w.is_alive():
                        alives.append(w)
                waitting = alives
                self._running = len(waitting)
            sleep(1)
            if show_bar:
                bar.update()
                # print(self._running)

    def close(self):
        self._close = True
        
    def join(self):
        for p in tqdm(self._processes):
            p.join()
