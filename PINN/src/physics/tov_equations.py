# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:10:34 2025

@author: zhang
"""

import tensorflow as tf
import numpy as np
from . import eos

# 物理常数
# c = G/(c^2) 在自然单位制下的值，用于TOV方程
c = 0.121467

# 初始条件参数
INITIAL_P = 10.0  # 初始压强（中心压强）
INITIAL_M = 1e-10  # 初始质量（接近0）
R_INITIAL = 0.01  # 初始半径

# 损失函数：包括TOV方程的约束和初始条件
def compute_loss(model, r):
    """
    计算PINN的损失函数
    
    参数:
    model: PINN模型
    r: 训练点半径，形状为 (n_points, 1)
    
    返回:
    loss: 总损失值
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(r)
        output = model(r)
        p, m = output[:, 0:1], output[:, 1:2]
        
        # 计算能量密度 epsilon
        epsilon = eos.eos(p)
        
        # 自动微分计算 dp/dr 和 dm/dr
        dp_dr = tape.gradient(p, r)
        dm_dr = tape.gradient(m, r)
        
        # TOV方程：
        # dp/dr = -(epsilon + p) * (m + c*r^3*p) / (r^2 - 2*m*r)
        # dm/dr = c * r^2 * epsilon
        
        # 计算分母，添加小的epsilon避免数值不稳定
        # 当r接近0时，r^2 - 2*m*r 接近0，需要防止除零
        denominator = r ** 2 - 2 * m * r
        epsilon_small = 1e-8  # 防止除零的小值
        # 使用maximum确保分母的绝对值不会太小
        denominator = tf.where(
            tf.abs(denominator) < epsilon_small,
            tf.sign(denominator) * epsilon_small,
            denominator
        )
        
        # TOV方程的残差
        tov_eq1 = dp_dr + (epsilon + p) * (m + c * r ** 3 * p) / denominator
        tov_eq2 = dm_dr - c * r ** 2 * epsilon

        # TOV方程的损失
        loss_tov = tf.reduce_mean(tf.square(tov_eq1)) + tf.reduce_mean(tf.square(tov_eq2))
        
        # 注意：初始条件已经在模型中通过硬约束实现（见tov_pinn.py）
        # 这里只需要TOV方程的损失即可
    
    return loss_tov