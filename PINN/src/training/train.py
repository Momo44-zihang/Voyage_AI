# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:11:41 2025

@author: zhang
"""

import tensorflow as tf
import numpy as np
from ..physics import tov_equations

# 训练PINN
def train_pinn(model, epochs=1000, learning_rate=1e-3, r_min=0.01, r_max=10, n_points=100):
    """
    训练PINN模型
    
    参数:
    model: PINN模型
    epochs: 训练轮数
    learning_rate: 学习率
    r_min: 最小半径
    r_max: 最大半径
    n_points: 训练点数量
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 生成训练点：确保包含初始点
    r_train_np = np.linspace(r_min, r_max, n_points).reshape(-1, 1)
    # 确保初始点在训练集中
    if r_min == 0.01:
        r_train_np[0] = 0.01
    r_train = tf.convert_to_tensor(r_train_np, dtype=tf.float32)
    
    print(f"开始训练PINN模型...")
    print(f"训练参数: epochs={epochs}, learning_rate={learning_rate}")
    print(f"训练范围: r ∈ [{r_min}, {r_max}], 训练点数: {n_points}")
    print("-" * 50)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = tov_equations.compute_loss(model, r_train)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.numpy():.6e}")
    
    print("-" * 50)
    print(f"训练完成！最终损失: {loss.numpy():.6e}")
