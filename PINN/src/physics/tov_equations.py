# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:10:34 2025

@author: zhang
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config_hub2 import _importpath
sys.path.append(_importpath)
from PINN.src.physics import eos

c = 0.121467

# 损失函数：包括TOV方程的约束
def compute_loss(model, r):
    # 在r=0附近的初始点
    r_initial = tf.constant([[0.01]], dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(r)
        output = model(r)
        p, m = output[:, 0:1], output[:, 1:2]
        
        # 计算epsilon
        epsilon = eos.eos(p)
        
        # 自动微分计算 dp/dr 和 dm/dr
        dp_dr = tape.gradient(p, r)
        dm_dr = tape.gradient(m, r)
        
        # TOV方程中的误差
        tov_eq1 = dp_dr + (epsilon + p) * (m + c * r ** 3 * p) / (r ** 2 - 2 * m * r)
        tov_eq2 = dm_dr - c * r ** 2 * epsilon

        # 定义损失函数，包含TOV方程的约束
        loss_tov = tf.reduce_mean(tf.square(tov_eq1)) + tf.reduce_mean(tf.square(tov_eq2))
    
    return loss_tov