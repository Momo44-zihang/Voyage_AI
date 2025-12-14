# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:12:47 2025

@author: zhang
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_mass_radius(model, r_max=15, n_points=200):
    """
    绘制TOV方程的数值解：M-r 和 P-r 曲线
    
    参数:
    model: 训练好的PINN模型
    r_max: 最大半径
    n_points: 绘图点数
    """
    # 生成测试点
    r_test = np.linspace(0.01, r_max, n_points).reshape(-1, 1).astype(np.float32)
    
    # 获取模型预测
    predictions = model.predict(r_test, verbose=0)
    p_pred = predictions[:, 0]  # 压强
    m_pred = predictions[:, 1]  # 质量
    
    # 创建图形：两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: M-r 曲线（质量-半径关系）
    ax1.plot(r_test, m_pred, 'b-', linewidth=2, label='PINN Solution')
    ax1.set_xlabel('Radius r (km)', fontsize=12)
    ax1.set_ylabel('Mass M (M☉)', fontsize=12)
    ax1.set_title('TOV Equation: Mass-Radius Relation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, r_max])
    
    # 子图2: P-r 曲线（压强-半径关系）
    ax2.plot(r_test, p_pred, 'r-', linewidth=2, label='PINN Solution')
    ax2.set_xlabel('Radius r (km)', fontsize=12)
    ax2.set_ylabel('Pressure P', fontsize=12)
    ax2.set_title('TOV Equation: Pressure-Radius Relation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, r_max])
    ax2.set_yscale('log')  # 使用对数刻度，因为压强变化范围可能很大
    
    plt.tight_layout()
    plt.show()
    
    # 打印一些统计信息
    print("\n" + "="*50)
    print("TOV方程数值解统计信息:")
    print("="*50)
    print(f"半径范围: r ∈ [0.01, {r_max:.2f}] km")
    print(f"质量范围: M ∈ [{m_pred.min():.6f}, {m_pred.max():.6f}] M☉")
    print(f"压强范围: P ∈ [{p_pred.min():.6e}, {p_pred.max():.6e}]")
    print(f"中心压强: P(0.01) = {p_pred[0]:.6e}")
    print(f"中心质量: M(0.01) = {m_pred[0]:.6e}")
    print("="*50)