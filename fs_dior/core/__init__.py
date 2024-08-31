# from . import builtin
# from . import wf_roi_heads
from .meta import register_all
# builtin.register_all_dior()
COMMON_CONFG = {"CROP_EN": False}


def init():
    """初始化 nwpu 核心模块
    """
    register_all("datasets/")