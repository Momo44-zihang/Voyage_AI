# -*- coding: utf-8 -*-
"""
状态方程 (Equation of State, EOS)
用于计算能量密度 epsilon 与压强 p 的关系

Created on Wed Dec 10 17:06:15 2025
@author: zhang
"""

import tensorflow as tf

# 状态方程参数
anr = 2.4216
ar = 2.8663

def eos(p):
    """
    状态方程：计算能量密度
    
    参数:
    p: 压强，TensorFlow张量
    
    返回:
    epsilon: 能量密度，epsilon = anr * p^(3/5) + ar * p
    """
    epsilon = anr * p ** (3 / 5) + ar * p
    return epsilon
