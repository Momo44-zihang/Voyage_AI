# -*- coding: utf-8 -*-
"""
TOV方程的PINN求解
使用物理信息神经网络求解Tolman-Oppenheimer-Volkoff方程
得到中子星的质量-半径关系和压强-半径关系

Tensorflow 版本

Created on Wed Dec 10 17:13:56 2025
@author: zhang
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 添加父目录到路径，以便导入PINN模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入配置路径（如果存在）
try:
    from config_hub_ai import _importpath
    sys.path.append(_importpath)
except ImportError:
    pass

from PINN.src.models import tov_pinn
from PINN.src.training import train 
from PINN.src.visualization import plot_mr

# %%
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # 创建PINN模型
    # 初始条件：中心压强 P(0.01) = 10.0, 中心质量 M(0.01) = 1e-10
    model = tov_pinn.TOV_PINN_with_IC(
        initial_p=10.0,
        initial_m=1e-10,
        r_initial=0.01
    )
    
    # 训练模型
    train.train_pinn(
        model, 
        epochs=2000, 
        learning_rate=1e-3,
        r_min=0.01,
        r_max=10,
        n_points=100
    )
    
    # 绘制结果：M-r 和 P-r 曲线
    plot_mr.plot_mass_radius(model, r_max=15, n_points=200)
