# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 17:11:41 2025

@author: zhang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ..physics import tov_equations

def generate_adaptive_points(r_min, r_max, n_points, density_strategy='uniform', 
                            r_initial=None, density_params=None):
    """
    生成自适应密度的训练点
    
    参数:
    r_min: 最小半径
    r_max: 最大半径
    n_points: 总训练点数
    density_strategy: 密度策略
        - 'uniform': 均匀分布
        - 'center_focused': 中心区域更密集
        - 'boundary_focused': 边界区域更密集
        - 'multi_region': 多个关键区域密集
        - 'custom': 自定义密度函数
    r_initial: 初始半径（用于某些策略）
    density_params: 密度参数字典，例如：
        - {'center_weight': 3.0, 'center_region': 0.1}  # center_focused策略
        - {'boundary_weight': 2.0, 'boundary_region': 0.2}  # boundary_focused策略
        - {'regions': [(r1, r2, weight1), (r3, r4, weight2)]}  # multi_region策略
        - {'density_func': lambda r: ...}  # custom策略
    
    返回:
    r_points: 训练点数组，形状为 (n_points, 1)
    """
    if density_params is None:
        density_params = {}
    
    if density_strategy == 'uniform':
        # 均匀分布
        r_points = np.linspace(r_min, r_max, n_points).reshape(-1, 1)
        
    elif density_strategy == 'center_focused':
        # 中心区域更密集
        center_weight = density_params.get('center_weight', 3.0)
        center_region = density_params.get('center_region', 0.1)  # 前10%的区域
        
        # 使用逆累积分布函数方法
        def inv_cdf(u):
            """逆累积分布函数：u在[0,1]，返回对应的r值"""
            center_end = r_min + center_region * (r_max - r_min)
            if u < center_region:
                # 中心区域：使用权重分布
                return r_min + (center_end - r_min) * (u / center_region) ** (1.0 / center_weight)
            else:
                # 外围区域：均匀分布
                u_normalized = (u - center_region) / (1 - center_region)
                return center_end + (r_max - center_end) * u_normalized
        
        u_samples = np.linspace(0, 1, n_points)
        r_points = np.array([inv_cdf(u) for u in u_samples]).reshape(-1, 1)
        
    elif density_strategy == 'boundary_focused':
        # 边界区域更密集
        boundary_weight = density_params.get('boundary_weight', 2.0)
        boundary_region = density_params.get('boundary_region', 0.2)  # 边界20%的区域
        
        def inv_cdf(u):
            """边界密集的逆累积分布函数"""
            boundary_start = r_max - boundary_region * (r_max - r_min)
            if u < 1 - boundary_region:
                # 内部区域：均匀分布
                return r_min + (boundary_start - r_min) * u / (1 - boundary_region)
            else:
                # 边界区域：使用权重分布
                u_boundary = (u - (1 - boundary_region)) / boundary_region
                return boundary_start + (r_max - boundary_start) * u_boundary ** (1.0 / boundary_weight)
        
        u_samples = np.linspace(0, 1, n_points)
        r_points = np.array([inv_cdf(u) for u in u_samples]).reshape(-1, 1)
        
    elif density_strategy == 'multi_region':
        # 多个关键区域密集
        regions = density_params.get('regions', [])
        if not regions:
            # 默认：中心区域和边界区域都密集
            regions = [
                (r_min, r_min + 0.1 * (r_max - r_min), 3.0),  # 中心区域
                (r_max - 0.2 * (r_max - r_min), r_max, 2.0)  # 边界区域
            ]
        
        # 构建累积密度函数
        total_weight = 0
        segments = []
        for r_start, r_end, weight in regions:
            length = r_end - r_start
            total_weight += weight * length
            segments.append((r_start, r_end, weight, length))
        
        # 其他区域均匀分布
        other_length = r_max - r_min - sum(s[3] for s in segments)
        total_weight += other_length
        
        # 分配点数
        points_per_segment = []
        remaining_points = n_points
        for r_start, r_end, weight, length in segments:
            n_seg = int(n_points * (weight * length) / total_weight)
            points_per_segment.append((r_start, r_end, n_seg))
            remaining_points -= n_seg
        
        # 其他区域
        other_regions = []
        last_end = r_min
        for r_start, r_end, _, _ in sorted(segments, key=lambda x: x[0]):
            if r_start > last_end:
                other_regions.append((last_end, r_start))
            last_end = max(last_end, r_end)
        if last_end < r_max:
            other_regions.append((last_end, r_max))
        
        if other_regions:
            n_other = remaining_points // len(other_regions)
            for r_start, r_end in other_regions:
                points_per_segment.append((r_start, r_end, n_other))
        
        # 生成点
        r_points_list = []
        for r_start, r_end, n_seg in points_per_segment:
            if n_seg > 0:
                r_points_list.extend(np.linspace(r_start, r_end, n_seg))
        
        r_points = np.array(sorted(r_points_list)).reshape(-1, 1)
        # 确保点数正确
        if len(r_points) < n_points:
            # 补充点
            r_points = np.linspace(r_min, r_max, n_points).reshape(-1, 1)
        elif len(r_points) > n_points:
            # 均匀采样
            indices = np.linspace(0, len(r_points) - 1, n_points, dtype=int)
            r_points = r_points[indices]
        
    elif density_strategy == 'custom':
        # 自定义密度函数
        density_func = density_params.get('density_func', lambda r: 1.0)
        
        # 使用拒绝采样或逆变换采样
        # 这里使用简单的逆变换采样
        r_samples = np.linspace(r_min, r_max, n_points * 10)  # 更密集的采样
        densities = np.array([density_func(r) for r in r_samples])
        densities = densities / densities.sum()  # 归一化
        cumsum = np.cumsum(densities)
        
        # 从累积分布中采样
        u_samples = np.linspace(0, 1, n_points)
        r_points = np.interp(u_samples, cumsum, r_samples).reshape(-1, 1)
        
    else:
        raise ValueError(f"未知的密度策略: {density_strategy}")
    
    # 确保包含初始点（如果指定）
    if r_initial is not None and r_min <= r_initial <= r_max:
        idx = np.argmin(np.abs(r_points.flatten() - r_initial))
        r_points[idx] = r_initial
    
    return r_points

# 训练PINN
def train_pinn(model, epochs=1000, learning_rate=1e-3, r_min=0.01, r_max=20, n_points=100,
               density_strategy='center_focused', density_params=None,
               use_soft_constraint=None, ic_weight=1000.0):
    """
    训练PINN模型
    
    参数:
    model: PINN模型
    epochs: 训练轮数
    learning_rate: 学习率
    r_min: 最小半径
    r_max: 最大半径
    n_points: 训练点数量
    density_strategy: 训练点密度策略
        - 'uniform': 均匀分布
        - 'center_focused': 中心区域更密集（推荐用于TOV方程）
        - 'boundary_focused': 边界区域更密集
        - 'multi_region': 多个关键区域密集
        - 'custom': 自定义密度函数
    density_params: 密度参数字典，例如：
        - {'center_weight': 3.0, 'center_region': 0.1}  # center_focused策略
        - {'boundary_weight': 2.0, 'boundary_region': 0.2}  # boundary_focused策略
        - {'regions': [(r1, r2, weight1), (r3, r4, weight2)]}  # multi_region策略
        - {'density_func': lambda r: ...}  # custom策略
    use_soft_constraint: 是否使用软约束（None时自动检测模型类型）
    ic_weight: 初始条件损失的权重（仅在软约束时使用，默认1000.0）
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # 生成自适应密度的训练点
    r_train_np = generate_adaptive_points(
        r_min=r_min,
        r_max=r_max,
        n_points=n_points,
        density_strategy=density_strategy,
        r_initial=model.r_initial.numpy() if hasattr(model, 'r_initial') else None,
        density_params=density_params
    )
    r_train = tf.convert_to_tensor(r_train_np, dtype=tf.float32)
    
    # 自动检测约束类型（如果未指定）
    if use_soft_constraint is None:
        # 根据模型类名自动判断
        model_class_name = model.__class__.__name__
        use_soft_constraint = 'Soft_IC' in model_class_name
    
    print(f"开始训练PINN模型...")
    print(f"训练参数: epochs={epochs}, learning_rate={learning_rate}")
    print(f"训练范围: r ∈ [{r_min}, {r_max}], 训练点数: {n_points}")
    print(f"密度策略: {density_strategy}")
    print(f"约束类型: {'软约束' if use_soft_constraint else '硬约束'}")
    if use_soft_constraint:
        print(f"初始条件权重: {ic_weight}")
    if density_params:
        print(f"密度参数: {density_params}")
    print("-" * 50)
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = tov_equations.compute_loss(model, r_train, 
                                             use_soft_constraint=use_soft_constraint,
                                             ic_weight=ic_weight)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.numpy():.6e}")
    
    print("-" * 50)
    print(f"训练完成！最终损失: {loss.numpy():.6e}")
    
    return r_train_np  # 返回训练点，方便后续分析

def plot_training_points(r_points, r_min=None, r_max=None, title="训练点分布"):
    """
    可视化训练点的分布
    
    参数:
    r_points: 训练点数组，形状为 (n_points, 1) 或 (n_points,)
    r_min: 最小半径（可选）
    r_max: 最大半径（可选）
    title: 图表标题
    """
    r_points = np.array(r_points).flatten()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 子图1: 训练点位置
    ax1.scatter(r_points, np.ones_like(r_points), alpha=0.6, s=20, c='blue')
    ax1.set_xlabel('半径 r', fontsize=12)
    ax1.set_ylabel('训练点', fontsize=12)
    ax1.set_title(f'{title} - 训练点位置', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if r_min is not None and r_max is not None:
        ax1.set_xlim([r_min, r_max])
    
    # 子图2: 训练点密度（直方图）
    n_bins = min(50, len(r_points) // 2)
    ax2.hist(r_points, bins=n_bins, density=True, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('半径 r', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title(f'{title} - 训练点密度分布', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if r_min is not None and r_max is not None:
        ax2.set_xlim([r_min, r_max])
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n训练点分布统计:")
    print(f"  总点数: {len(r_points)}")
    print(f"  半径范围: [{r_points.min():.4f}, {r_points.max():.4f}]")
    print(f"  平均间距: {(r_points.max() - r_points.min()) / len(r_points):.6f}")
    
    # 计算不同区域的点数
    if r_min is not None and r_max is not None:
        center_region = r_min + 0.1 * (r_max - r_min)
        boundary_region = r_max - 0.2 * (r_max - r_min)
        n_center = np.sum(r_points <= center_region)
        n_boundary = np.sum(r_points >= boundary_region)
        n_middle = len(r_points) - n_center - n_boundary
        print(f"  中心区域 (r ≤ {center_region:.4f}): {n_center} 点 ({100*n_center/len(r_points):.1f}%)")
        print(f"  中间区域: {n_middle} 点 ({100*n_middle/len(r_points):.1f}%)")
        print(f"  边界区域 (r ≥ {boundary_region:.4f}): {n_boundary} 点 ({100*n_boundary/len(r_points):.1f}%)")
