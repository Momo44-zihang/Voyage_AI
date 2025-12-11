# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:11:41 2025

@author: zhang
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config_hub2 import _importpath
sys.path.append(_importpath)
from PINN.src.physics import tov_equations

# 训练PINN
def train_pinn(model, epochs=1000, learning_rate=1e-3):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    r_train = tf.convert_to_tensor(np.linspace(0.01, 10, 100).reshape(-1, 1), dtype=tf.float32)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = tov_equations.compute_loss(model, r_train)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.numpy()}")
