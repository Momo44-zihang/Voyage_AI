# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:09:22 2025

@author: zhang
"""

import tensorflow as tf
import numpy as np

# 定义带初始条件的PINN网络
class TOV_PINN_with_IC(tf.keras.Model):
    def __init__(self, initial_p=10.0, initial_m=1e-10, r_initial=0.01):
        """
        初始化TOV PINN模型
        
        参数:
        initial_p: 初始压强 (中心压强)
        initial_m: 初始质量 (接近0)
        r_initial: 初始半径值
        """
        super(TOV_PINN_with_IC, self).__init__()
        # 创建隐藏层
        self.dense1 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(20, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(20, activation='tanh')
        # 输出层：使用 softplus 激活函数确保输出为正
        self.dense4 = tf.keras.layers.Dense(2, activation='softplus')
        
        # 存储初始条件
        self.initial_p = tf.constant(initial_p, dtype=tf.float32)
        self.initial_m = tf.constant(initial_m, dtype=tf.float32)
        self.r_initial = tf.constant(r_initial, dtype=tf.float32)

    def call(self, r):
        """
        前向传播
        
        参数:
        r: 半径值，形状为 (batch_size, 1)
        
        返回:
        output: [p, m]，形状为 (batch_size, 2)
        """
        x = self.dense1(r)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.dense4(x)
        
        # 对于初始点，应用硬约束
        # 使用掩码来识别初始点（允许小的数值误差）
        r_flat = tf.reshape(r, [-1])  # 展平r以便比较
        is_initial = tf.abs(r_flat - self.r_initial) < 1e-6
        is_initial = tf.cast(is_initial, tf.float32)  # 转换为float32
        
        # 创建初始条件张量，形状为 (batch_size, 2)
        # 使用tf.shape获取动态batch_size
        batch_size = tf.shape(r)[0]
        initial_p_tensor = tf.fill([batch_size], self.initial_p)
        initial_m_tensor = tf.fill([batch_size], self.initial_m)
        initial_condition = tf.stack([initial_p_tensor, initial_m_tensor], axis=1)
        
        # 对于初始点，使用硬约束；对于其他点，使用网络输出
        # 使用broadcasting: is_initial形状为[batch_size]，expand到[batch_size, 2]
        is_initial_expanded = tf.expand_dims(is_initial, axis=1)  # [batch_size, 1]
        is_initial_expanded = tf.tile(is_initial_expanded, [1, 2])  # [batch_size, 2]
        
        # 使用stop_gradient确保初始条件的梯度不会传播，但允许网络输出有梯度
        initial_condition_stopped = tf.stop_gradient(initial_condition)
        output = output * (1.0 - is_initial_expanded) + initial_condition_stopped * is_initial_expanded
        
        return output