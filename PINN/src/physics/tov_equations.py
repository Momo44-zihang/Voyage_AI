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
INITIAL_P = 0.01  # 初始压强（中心压强）
INITIAL_M = 1e-10  # 初始质量（接近0）
R_INITIAL = 0.01  # 初始半径

# 损失函数：包括TOV方程的约束和初始条件
def compute_loss(model, r, use_soft_constraint=False, ic_weight=1000.0):
    """
    计算PINN的损失函数
    
    参数:
    model: PINN模型
    r: 训练点半径，形状为 (n_points, 1)
    use_soft_constraint: 是否使用软约束（如果模型是TOV_PINN_with_Soft_IC，应设为True）
    ic_weight: 初始条件损失的权重（仅在use_soft_constraint=True时使用）
    
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
        
        # 根据约束类型处理初始条件
        if use_soft_constraint:
            # 软约束：通过损失函数约束初始条件
            r_flat = tf.reshape(r, [-1])
            is_initial = tf.abs(r_flat - model.r_initial) < 1e-6
            
            if tf.reduce_any(is_initial):
                # 获取初始点的预测值
                initial_indices = tf.where(is_initial)
                p_initial = tf.gather_nd(p, initial_indices)
                m_initial = tf.gather_nd(m, initial_indices)
                
                # 初始条件损失
                loss_ic = ic_weight * (
                    tf.reduce_mean(tf.square(p_initial - model.initial_p)) +
                    tf.reduce_mean(tf.square(m_initial - model.initial_m))
                )
            else:
                loss_ic = 0.0
            
            return loss_tov + loss_ic
        else:
            # 硬约束：初始条件通过模型中的硬约束精确满足（仅在初始点）
            # 这里只需要TOV方程的损失即可
            return loss_tov