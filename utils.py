# utils.py
import sys
from pathlib import Path

def get_base_path():
    """
    获取程序运行的基准路径。
    如果是通过 exe 运行，则返回 exe 所在的目录；
    如果是 Python 脚本运行，则返回脚本所在目录。
    """
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent

def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent
    return base_path / relative_path