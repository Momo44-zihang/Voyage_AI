# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:09:22 2025

@author: zhang
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from config_hub2 import _importpath
# sys.path.append(_importpath)

# 定义带初始条件的PINN网络
class TOV_PINN_with_IC(tf.keras.Model):
    def __init__(self):
        super(TOV_PINN_with_IC, self).__init__()
        # 创建隐藏层
        self.dense1 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(20, activation='tanh')
        # 输出层：对于非初始点的数据使用 softplus 激活函数
        self.dense4 = tf.keras.layers.Dense(2, activation='softplus')

    def call(self, r):
        x = self.dense1(r)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        
        # 初始点的硬编码约束
        initial_p = 10.0  # 固定初始压强
        initial_m = 1e-10 # 固定初始质量
        
        # 将初始点固定为所需值，其他点按网络输出
        initial_condition = tf.constant([[initial_p, initial_m]], dtype=tf.float32)
        output_with_ic = tf.concat([initial_condition, output[1:]], axis=0)
        
        return output_with_ic