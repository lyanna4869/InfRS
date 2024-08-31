from .meta import register_all

def init():
    """初始化 dota 核心模块
    """
    register_all("datasets/")

def deinit():
    """卸载 dota 模块
    """
    pass
