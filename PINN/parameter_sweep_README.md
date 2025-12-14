# 参数扫描脚本使用说明

## 概述

`parameter_sweep.py` 是一个系统化的参数扫描工具，用于测试不同参数组合，找到最优的PINN模型配置。

## 功能特性

1. **自动参数扫描**：测试多个参数组合
2. **性能评估**：评估初始条件误差、TOV方程残差等指标
3. **结果保存**：自动保存JSON格式的结果和文本报告
4. **可视化**：生成参数影响分析图表
5. **最佳配置推荐**：基于不同指标推荐最佳参数

## 使用方法

### 基本使用

```python
from parameter_sweep import ParameterSweep

# 创建扫描器
sweep = ParameterSweep(
    constraint_type='soft',  # 或 'hard'
    initial_p=10.0,
    initial_m=1e-10,
    r_initial=0.01
)

# 定义参数网格
param_grid = {
    'ic_weight': [100, 1000, 5000],
    'learning_rate': [1e-4, 1e-3, 5e-3],
    'density_strategy': ['uniform', 'center_focused'],
    'n_points': [100],
    'r_min': [0.01],
    'r_max': [20]
}

# 执行扫描
results = sweep.sweep(
    param_grid=param_grid,
    epochs=1000,  # 每个配置的训练轮数
    save_results=True,
    output_dir='sweep_results'
)
```

### 方法1：参数网格（自动生成所有组合）

```python
param_grid = {
    'ic_weight': [100, 1000, 5000],        # 3个值
    'learning_rate': [1e-4, 1e-3, 5e-3],  # 3个值
    'density_strategy': ['uniform', 'center_focused']  # 2个值
}
# 总共会测试 3 × 3 × 2 = 18 个组合
```

### 方法2：手动指定参数组合（更灵活）

```python
param_combinations = [
    {
        'ic_weight': 100,
        'learning_rate': 1e-3,
        'density_strategy': 'uniform',
        'density_params': None,
        'n_points': 100,
        'r_min': 0.01,
        'r_max': 20
    },
    {
        'ic_weight': 1000,
        'learning_rate': 1e-3,
        'density_strategy': 'center_focused',
        'density_params': {'center_weight': 3.0, 'center_region': 0.15},
        'n_points': 100,
        'r_min': 0.01,
        'r_max': 20
    }
]

results = sweep.sweep(param_grid=param_combinations, epochs=1000)
```

## 参数说明

### 可扫描的参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `ic_weight` | float | 初始条件损失权重（仅软约束） | 1000.0 |
| `learning_rate` | float | 学习率 | 1e-3 |
| `density_strategy` | str | 训练点密度策略 | 'uniform' |
| `density_params` | dict | 密度策略参数 | None |
| `n_points` | int | 训练点数量 | 100 |
| `r_min` | float | 最小半径 | 0.01 |
| `r_max` | float | 最大半径 | 20 |

### 密度策略参数

**center_focused策略：**
```python
'density_params': {
    'center_weight': 3.0,    # 中心区域密度权重（越大越密集）
    'center_region': 0.15    # 中心区域范围（前15%的区域）
}
```

**uniform策略：**
```python
'density_params': None  # 不需要参数
```

## 评估指标

扫描脚本会评估以下指标：

1. **初始条件误差**
   - `ic_error_p`: 压强初始条件误差
   - `ic_error_m`: 质量初始条件误差
   - `ic_error_total`: 总初始条件误差
   - `ic_error_p_rel`: 相对压强误差
   - `ic_error_m_rel`: 相对质量误差

2. **TOV方程残差**
   - `tov_loss`: TOV方程的残差损失

3. **预测值合理性**
   - `p_min`, `p_max`: 压强范围
   - `m_min`, `m_max`: 质量范围
   - `p_monotonic`: 压强是否单调递减
   - `m_monotonic`: 质量是否单调递增

4. **训练信息**
   - `training_time`: 训练时间（秒）
   - `success`: 是否成功

## 输出文件

扫描完成后，会在 `output_dir` 目录下生成：

1. **results.json**: 所有实验结果的JSON文件
2. **report.txt**: 文本格式的报告，包含：
   - 最佳配置（按不同指标）
   - 参数影响分析
   - 统计信息
3. **results_plots.png**: 可视化图表，包含：
   - IC权重 vs IC误差
   - 学习率 vs TOV损失
   - IC误差 vs TOV损失（帕累托前沿）
   - 训练时间分布

## 示例：快速扫描

```python
# 快速测试：只扫描IC权重和学习率
quick_grid = {
    'ic_weight': [100, 1000, 5000],
    'learning_rate': [1e-3, 5e-3],
    'density_strategy': ['center_focused'],
    'n_points': [100],
    'r_min': [0.01],
    'r_max': [20]
}

sweep = ParameterSweep(constraint_type='soft')
results = sweep.sweep(quick_grid, epochs=500)  # 快速测试用500轮
```

## 示例：完整扫描

```python
# 完整扫描：测试更多参数组合
full_grid = {
    'ic_weight': [100, 500, 1000, 5000, 10000],
    'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'density_strategy': ['uniform', 'center_focused'],
    'n_points': [50, 100, 200],
    'r_min': [0.01],
    'r_max': [20]
}
# 总共 5 × 5 × 2 × 3 = 150 个组合

sweep = ParameterSweep(constraint_type='soft')
results = sweep.sweep(full_grid, epochs=2000)  # 完整训练用2000轮
```

## 最佳实践

1. **分阶段扫描**：
   - 第一阶段：快速扫描（epochs=500），找到大致范围
   - 第二阶段：精细扫描（epochs=2000），在最优范围内细化

2. **参数优先级**：
   - 先扫描 `ic_weight`（最重要）
   - 再扫描 `learning_rate`
   - 最后扫描 `density_strategy` 和 `density_params`

3. **组合数量控制**：
   - 快速测试：< 20个组合
   - 中等扫描：20-50个组合
   - 完整扫描：50-200个组合

4. **结果分析**：
   - 查看 `report.txt` 了解最佳配置
   - 查看 `results_plots.png` 了解参数影响
   - 在 `results.json` 中查找特定配置的详细信息

## 注意事项

1. **计算时间**：参数扫描可能需要较长时间，建议：
   - 先用少量epochs快速测试
   - 在关键参数范围内进行精细扫描
   - 使用GPU加速训练

2. **内存管理**：如果组合数很多，注意：
   - 每10个组合自动保存一次（防止数据丢失）
   - 可以分批运行不同的参数范围

3. **参数验证**：脚本会自动验证和修正参数：
   - `density_params` 会自动匹配 `density_strategy`
   - 无效的参数组合会被跳过

## 故障排除

**问题：所有实验都失败**
- 检查参数范围是否合理
- 尝试降低学习率
- 检查初始条件是否合理

**问题：结果文件太大**
- 减少扫描的参数组合数
- 只保存关键指标

**问题：训练时间过长**
- 减少epochs数量
- 减少训练点数量（n_points）
- 使用更小的参数网格

## 示例输出

```
======================================================================
参数扫描开始
======================================================================
约束类型: soft
总组合数: 18
每个配置训练轮数: 500
======================================================================

[1/18] 测试参数组合:
  ic_weight: 100
  learning_rate: 0.0001
  density_strategy: uniform
✓ 完成: IC误差=0.123456, TOV损失=1.234567e-05

...

======================================================================
参数扫描完成！
======================================================================

最佳配置（综合评分）:
推荐参数配置:
----------------------------------------------------------------------
  ic_weight: 1000
  learning_rate: 0.001
  density_strategy: center_focused
  density_params: {'center_weight': 3.0, 'center_region': 0.15}
  ic_error_total: 0.000123
  tov_loss: 1.234567e-06
  score: 6.172835e-05
----------------------------------------------------------------------
```

