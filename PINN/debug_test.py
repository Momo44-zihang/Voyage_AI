#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试脚本"""

import sys
import traceback

print("Python版本:", sys.version)
print("当前工作目录:", sys.path[0] if sys.path else "N/A")
print("\n开始测试导入...")

try:
    print("\n1. 测试导入tensorflow...")
    import tensorflow as tf
    print(f"   ✓ TensorFlow版本: {tf.__version__}")
except Exception as e:
    print(f"   ✗ 错误: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. 测试导入numpy...")
    import numpy as np
    print(f"   ✓ NumPy版本: {np.__version__}")
except Exception as e:
    print(f"   ✗ 错误: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. 测试导入PINN模块...")
    # 添加父目录到路径
    import os
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(parent_dir))
    print(f"   添加路径: {os.path.dirname(parent_dir)}")
    
    from PINN.src.models import tov_pinn
    print("   ✓ 成功导入 tov_pinn")
    
    from PINN.src.training import train
    print("   ✓ 成功导入 train")
    
    from PINN.src.visualization import plot_mr
    print("   ✓ 成功导入 plot_mr")
    
    from PINN.src.physics import tov_equations
    print("   ✓ 成功导入 tov_equations")
    
except Exception as e:
    print(f"   ✗ 导入错误: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n4. 测试创建模型...")
    model = tov_pinn.TOV_PINN_with_IC(
        initial_p=10.0,
        initial_m=1e-10,
        r_initial=0.01
    )
    print("   ✓ 模型创建成功")
except Exception as e:
    print(f"   ✗ 模型创建错误: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n5. 测试模型前向传播...")
    r_test = tf.constant([[0.01], [0.1], [1.0]], dtype=tf.float32)
    output = model(r_test)
    print(f"   ✓ 前向传播成功，输出形状: {output.shape}")
except Exception as e:
    print(f"   ✗ 前向传播错误: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n6. 测试损失函数...")
    r_train = tf.constant(np.linspace(0.01, 10, 10).reshape(-1, 1), dtype=tf.float32)
    loss = tov_equations.compute_loss(model, r_train)
    print(f"   ✓ 损失计算成功，损失值: {loss.numpy()}")
except Exception as e:
    print(f"   ✗ 损失计算错误: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("所有测试通过！")
print("="*50)

