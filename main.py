import warnings
import faulthandler
import os

# 完全禁用所有警告信息
warnings.filterwarnings("ignore")

# 禁用环境变量相关警告
os.environ['PYTHONWARNINGS'] = 'ignore'

# 禁用TensorFlow/CUDA相关警告（如果存在）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.load_config import CONFIG
from src.run import run, run_pyg

# 禁用PyTorch警告
try:
    import torch
    # PyTorch没有自己的warnings系统，使用标准warnings模块
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
except ImportError:
    pass

# 禁用RDKit警告
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    pass

# 禁用NumPy警告
try:
    import numpy as np
    # 使用标准warnings模块禁用NumPy相关警告
    warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
except ImportError:
    pass

# 禁用Pandas警告
try:
    import pandas as pd
    pd.options.mode.chained_assignment = None
    # copy_on_write 只能设置为 True/False/"warn"，不能是 None
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
except (ImportError, AttributeError):
    pass

faulthandler.enable()


def main():
    if CONFIG['is_pyg']:
        run_pyg()
    else:
        run()


if __name__ == '__main__':
    main()

