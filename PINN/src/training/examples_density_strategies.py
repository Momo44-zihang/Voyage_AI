# -*- coding: utf-8 -*-
"""
训练点密度策略示例
展示如何使用不同的密度分布策略来训练PINN模型

Created on Wed Dec 10 2025
@author: zhang
"""

import sys
import os
import tensorflow as tf
import numpy as np

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PINN.src.models import tov_pinn
from PINN.src.training import train

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

# 创建模型
model = tov_pinn.TOV_PINN_with_IC(
    initial_p=10.0,
    initial_m=1e-10,
    r_initial=0.01
)

print("=" * 70)
print("训练点密度策略示例")
print("=" * 70)

# ============================================================================
# 示例1: 均匀分布
# ============================================================================
print("\n【示例1】均匀分布")
print("-" * 70)
r_train_1 = train.train_pinn(
    model, 
    epochs=500,  # 示例中减少epochs以便快速演示
    learning_rate=1e-3,
    r_min=0.01,
    r_max=20,
    n_points=100,
    density_strategy='uniform'
)
train.plot_training_points(r_train_1, r_min=0.01, r_max=20, title="均匀分布")

# ============================================================================
# 示例2: 中心区域密集（推荐用于TOV方程）
# ============================================================================
print("\n【示例2】中心区域密集")
print("-" * 70)
model2 = tov_pinn.TOV_PINN_with_IC(
    initial_p=10.0,
    initial_m=1e-10,
    r_initial=0.01
)
r_train_2 = train.train_pinn(
    model2, 
    epochs=500,
    learning_rate=1e-3,
    r_min=0.01,
    r_max=20,
    n_points=100,
    density_strategy='center_focused',
    density_params={
        'center_weight': 3.0,  # 中心区域密度权重
        'center_region': 0.15  # 前15%的区域更密集
    }
)
train.plot_training_points(r_train_2, r_min=0.01, r_max=20, title="中心区域密集")

# ============================================================================
# 示例3: 边界区域密集
# ============================================================================
print("\n【示例3】边界区域密集")
print("-" * 70)
model3 = tov_pinn.TOV_PINN_with_IC(
    initial_p=10.0,
    initial_m=1e-10,
    r_initial=0.01
)
r_train_3 = train.train_pinn(
    model3, 
    epochs=500,
    learning_rate=1e-3,
    r_min=0.01,
    r_max=20,
    n_points=100,
    density_strategy='boundary_focused',
    density_params={
        'boundary_weight': 2.5,  # 边界区域密度权重
        'boundary_region': 0.2   # 最后20%的区域更密集
    }
)
train.plot_training_points(r_train_3, r_min=0.01, r_max=20, title="边界区域密集")

# ============================================================================
# 示例4: 多区域密集
# ============================================================================
print("\n【示例4】多区域密集（中心+边界）")
print("-" * 70)
model4 = tov_pinn.TOV_PINN_with_IC(
    initial_p=10.0,
    initial_m=1e-10,
    r_initial=0.01
)
r_train_4 = train.train_pinn(
    model4, 
    epochs=500,
    learning_rate=1e-3,
    r_min=0.01,
    r_max=20,
    n_points=100,
    density_strategy='multi_region',
    density_params={
        'regions': [
            (0.01, 0.5, 4.0),   # 中心区域 [0.01, 0.5]，权重4.0
            (15.0, 20.0, 2.0)   # 边界区域 [15.0, 20.0]，权重2.0
        ]
    }
)
train.plot_training_points(r_train_4, r_min=0.01, r_max=20, title="多区域密集")

# ============================================================================
# 示例5: 自定义密度函数
# ============================================================================
print("\n【示例5】自定义密度函数（基于物理特性）")
print("-" * 70)

def custom_density(r):
    """
    自定义密度函数：在梯度大的区域增加密度
    对于TOV方程，中心区域和某些中间区域可能需要更高密度
    """
    # 中心区域高密度
    if r < 1.0:
        return 5.0
    # 中间区域中等密度
    elif r < 5.0:
        return 1.0
    # 边界区域较高密度
    elif r < 15.0:
        return 0.5
    else:
        return 2.0

model5 = tov_pinn.TOV_PINN_with_IC(
    initial_p=10.0,
    initial_m=1e-10,
    r_initial=0.01
)
r_train_5 = train.train_pinn(
    model5, 
    epochs=500,
    learning_rate=1e-3,
    r_min=0.01,
    r_max=20,
    n_points=100,
    density_strategy='custom',
    density_params={
        'density_func': custom_density
    }
)
train.plot_training_points(r_train_5, r_min=0.01, r_max=20, title="自定义密度函数")

print("\n" + "=" * 70)
print("所有示例完成！")
print("=" * 70)
print("\n推荐策略：")
print("  - TOV方程中心梯度大：使用 'center_focused'")
print("  - 需要高精度边界：使用 'boundary_focused'")
print("  - 多个关键区域：使用 'multi_region'")
print("  - 复杂物理特性：使用 'custom' 自定义函数")

