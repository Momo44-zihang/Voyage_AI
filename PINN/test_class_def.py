#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试类定义"""

import sys
import os

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 强制清除缓存
import importlib
modules_to_remove = [
    'PINN.src.models.tov_pinn',
    'PINN.src.models',
    'PINN.src.models.__init__'
]
for mod_name in modules_to_remove:
    sys.modules.pop(mod_name, None)

# 导入模块
print("正在导入模块...")
from PINN.src.models import tov_pinn

# 检查类定义
print(f"\n模块文件: {getattr(tov_pinn, '__file__', 'unknown')}")
print(f"类: {tov_pinn.TOV_PINN_with_IC}")

import inspect
sig = inspect.signature(tov_pinn.TOV_PINN_with_IC.__init__)
params = list(sig.parameters.keys())
print(f"\n__init__ 参数: {params}")

# 检查关键参数
if 'initial_p' in params:
    print("✓ 'initial_p' 参数存在")
else:
    print("✗ 'initial_p' 参数不存在！")
    print(f"  实际参数: {params}")

# 尝试实例化
print("\n尝试实例化...")
try:
    model = tov_pinn.TOV_PINN_with_IC(
        initial_p=10.0,
        initial_m=1e-10,
        r_initial=0.01,
        use_soft_constraint=False
    )
    print("✓ 实例化成功！")
    print(f"  模型类型: {type(model)}")
except Exception as e:
    print(f"✗ 实例化失败: {e}")
    import traceback
    traceback.print_exc()

