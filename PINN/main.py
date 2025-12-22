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
from PINN.src.models.tov_pinn_soft import TOV_PINN_with_Soft_IC
from PINN.src.training import train 
from PINN.src.visualization import plot_mr

# %%
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # ========================================================================
    # 选择约束类型
    # ========================================================================
    # 'hard': 硬约束 - 精确满足初始条件，但在初始点可能有梯度突变
    # 'soft': 软约束 - 通过损失函数约束，梯度平滑，但初始条件可能不完全精确
    CONSTRAINT_TYPE = 'soft'  # 可选: 'hard' 或 'soft'
    
    # 软约束时的初始条件权重（仅在CONSTRAINT_TYPE='soft'时使用）
    IC_WEIGHT = 1000.0  # 越大，初始条件越精确，但可能影响训练稳定性
    
    # ========================================================================
    # 创建PINN模型
    # ========================================================================
    # 初始条件：中心压强 P(0.01) = 10.0, 中心质量 M(0.01) = 1e-10
    if CONSTRAINT_TYPE == 'hard':
        # 使用硬约束模型（仅在初始点使用硬约束）
        model = tov_pinn.TOV_PINN_with_IC(
            initial_p=10,
            initial_m=1e-10,
            r_initial=0.01
        )
        use_soft_constraint = False
    elif CONSTRAINT_TYPE == 'soft':
        # 使用软约束模型（通过损失函数约束）
        model = TOV_PINN_with_Soft_IC(
            initial_p=10,
            initial_m=1e-10,
            r_initial=0.01
        )
        use_soft_constraint = True
    else:
        raise ValueError(f"未知的约束类型: {CONSTRAINT_TYPE}，请选择 'hard' 或 'soft'")
    
    # ========================================================================
    # 训练模型 - 使用自适应密度分布
    # ========================================================================
    # 密度策略选项：
    # 1. 'center_focused': 中心区域更密集（推荐用于TOV方程，解决中心梯度大的问题）
    # 2. 'uniform': 均匀分布
    # 3. 'boundary_focused': 边界区域更密集
    # 4. 'multi_region': 多个关键区域密集
    # 5. 'custom': 自定义密度函数
    
    r_train_points = train.train_pinn(
        model, 
        epochs=3000, 
        learning_rate=1e-3,
        r_min=0.01,
        r_max=20,
        n_points=100,
        density_strategy='center_focused',  # 中心区域密集
        density_params={
            'center_weight': 3.0,  # 中心区域密度权重（越大越密集）
            'center_region': 0.15   # 中心区域范围（前15%的区域）
        },
        use_soft_constraint=use_soft_constraint,  # 自动根据模型类型设置
        ic_weight=IC_WEIGHT  # 软约束时的初始条件权重
    )
    
    # 可视化训练点分布（可选）
    train.plot_training_points(r_train_points, r_min=0.01, r_max=20)
    
    # 绘制结果：M-r 和 P-r 曲线
    plot_mr.plot_mass_radius(model, r_max=15, n_points=200)
    
    # 验证初始条件（可选）
    print("\n" + "="*50)
    print("初始条件验证:")
    print("="*50)
    r_initial_test = np.array([[0.01]], dtype=np.float32)
    predictions = model.predict(r_initial_test, verbose=0)
    p_initial_pred = predictions[0, 0]
    m_initial_pred = predictions[0, 1]
    print(f"期望值: P(0.01) = {10.0:.6e}, M(0.01) = {1e-10:.6e}")
    print(f"预测值: P(0.01) = {p_initial_pred:.6e}, M(0.01) = {m_initial_pred:.6e}")
    print(f"误差: ΔP = {abs(p_initial_pred - 10.0):.6e}, ΔM = {abs(m_initial_pred - 1e-10):.6e}")
    print("="*50)
