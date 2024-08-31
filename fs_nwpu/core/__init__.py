from .meta import register_all

def init():
    """初始化 nwpu 核心模块
    """
    register_all("datasets/")

def deinit():
    """卸载 nwpu 模块
    """
    pass
