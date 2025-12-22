# -*- coding: utf-8 -*-
"""
参数扫描脚本
用于系统化测试不同参数组合，找到最优参数配置

Created on 2025
@author: zhang
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from itertools import product
import pandas as pd

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从 src/sweep/ 到项目根目录需要向上两级
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config_hub_ai import _importpath
    sys.path.append(_importpath)
except ImportError:
    pass

# 强制重新加载模块（确保使用最新代码）
import importlib

# 删除可能已缓存的模块，强制重新导入
modules_to_remove = [
    'PINN.src.models.tov_pinn',
    'PINN.src.models',
    'PINN.src.models.__init__'
]
for mod_name in modules_to_remove:
    sys.modules.pop(mod_name, None)

# 重新导入模块
from PINN.src.models import tov_pinn

# 验证类定义是否正确
try:
    import inspect
    sig = inspect.signature(tov_pinn.TOV_PINN_with_IC.__init__)
    expected_params = ['initial_p', 'initial_m', 'r_initial', 'use_soft_constraint']
    actual_params = list(sig.parameters.keys())
    if not all(p in actual_params for p in expected_params):
        print(f"警告: 类定义可能不正确。期望参数: {expected_params}, 实际参数: {actual_params}")
        # 强制重新加载
        if 'PINN.src.models.tov_pinn' in sys.modules:
            importlib.reload(sys.modules['PINN.src.models.tov_pinn'])
        from PINN.src.models import tov_pinn
except Exception as e:
    print(f"警告: 无法验证类定义: {e}")
from PINN.src.training import train
from PINN.src.physics import tov_equations


class ParameterSweep:
    """参数扫描类"""
    
    def __init__(self, constraint_type='soft', initial_p=10.0, initial_m=1e-10, r_initial=0.01):
        """
        初始化参数扫描器
        
        参数:
        constraint_type: 约束类型 'hard' 或 'soft'
        initial_p: 初始压强
        initial_m: 初始质量
        r_initial: 初始半径
        """
        self.constraint_type = constraint_type
        self.initial_p = initial_p
        self.initial_m = initial_m
        self.r_initial = r_initial
        self.results = []
        
    def create_model(self, use_soft_constraint):
        """创建模型"""
        return tov_pinn.TOV_PINN_with_IC(
            initial_p=self.initial_p,
            initial_m=self.initial_m,
            r_initial=self.r_initial,
            use_soft_constraint=use_soft_constraint
        )
    
    def evaluate_model(self, model, r_test=None):
        """
        评估模型性能
        
        参数:
        model: 训练好的模型
        r_test: 测试点（可选）
        
        返回:
        metrics: 评估指标字典
        """
        metrics = {}
        
        # 1. 初始条件误差
        r_initial_test = np.array([[self.r_initial]], dtype=np.float32)
        predictions = model.predict(r_initial_test, verbose=0)
        p_initial_pred = predictions[0, 0]
        m_initial_pred = predictions[0, 1]
        
        metrics['ic_error_p'] = abs(p_initial_pred - self.initial_p)
        metrics['ic_error_m'] = abs(m_initial_pred - self.initial_m)
        metrics['ic_error_total'] = metrics['ic_error_p'] + metrics['ic_error_m']
        metrics['ic_error_p_rel'] = metrics['ic_error_p'] / self.initial_p if self.initial_p > 0 else 0
        metrics['ic_error_m_rel'] = metrics['ic_error_m'] / self.initial_m if self.initial_m > 0 else 0
        
        # 2. TOV方程残差（在测试点上）
        if r_test is None:
            r_test = np.linspace(self.r_initial, 20, 50).reshape(-1, 1).astype(np.float32)
        
        r_test_tf = tf.convert_to_tensor(r_test, dtype=tf.float32)
        loss = tov_equations.compute_loss(
            model, r_test_tf,
            use_soft_constraint=(self.constraint_type == 'soft'),
            ic_weight=0.0  # 评估时只计算TOV损失
        )
        metrics['tov_loss'] = float(loss.numpy())
        
        # 3. 预测值的合理性检查
        predictions_all = model.predict(r_test, verbose=0)
        p_all = predictions_all[:, 0]
        m_all = predictions_all[:, 1]
        
        metrics['p_min'] = float(p_all.min())
        metrics['p_max'] = float(p_all.max())
        metrics['m_min'] = float(m_all.min())
        metrics['m_max'] = float(m_all.max())
        metrics['p_monotonic'] = self._check_monotonic(p_all)  # 压强应该单调递减
        metrics['m_monotonic'] = self._check_monotonic(m_all, increasing=True)  # 质量应该单调递增
        
        return metrics
    
    def _check_monotonic(self, arr, increasing=False):
        """检查数组是否单调"""
        if increasing:
            return float(np.all(np.diff(arr) >= -1e-6))  # 允许小的数值误差
        else:
            return float(np.all(np.diff(arr) <= 1e-6))
    
    def train_and_evaluate(self, params, epochs=1000, verbose=False):
        """
        训练模型并评估
        
        参数:
        params: 参数字典
        epochs: 训练轮数
        verbose: 是否打印详细信息
        
        返回:
        result: 结果字典
        """
        # 重置随机种子以确保可重复性
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # 创建模型
        use_soft_constraint = (self.constraint_type == 'soft')
        model = self.create_model(use_soft_constraint)
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        try:
            r_train_points = train.train_pinn(
                model=model,
                epochs=epochs,
                learning_rate=params.get('learning_rate', 1e-3),
                r_min=params.get('r_min', 0.01),
                r_max=params.get('r_max', 20),
                n_points=params.get('n_points', 100),
                density_strategy=params.get('density_strategy', 'uniform'),
                density_params=params.get('density_params', None),
                use_soft_constraint=use_soft_constraint,
                ic_weight=params.get('ic_weight', 1000.0)
            )
            
            # 评估模型
            metrics = self.evaluate_model(model)
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 组合结果
            result = {
                **params,  # 包含所有参数
                **metrics,  # 包含所有评估指标
                'training_time': training_time,
                'success': True,
                'error': None
            }
            
            if verbose:
                print(f"✓ 完成: IC误差={metrics['ic_error_total']:.6e}, TOV损失={metrics['tov_loss']:.6e}")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"✗ 失败: {str(e)}")
            return {
                **params,
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def _validate_params(self, params):
        """验证参数组合的有效性"""
        # 确保density_params与density_strategy匹配
        density_strategy = params.get('density_strategy', 'uniform')
        density_params = params.get('density_params', None)
        
        # 如果提供了独立的 center_weight 和 center_region 参数
        has_center_weight = 'center_weight' in params
        has_center_region = 'center_region' in params
        
        if has_center_weight or has_center_region:
            if density_strategy == 'center_focused':
                # 对于 center_focused，组合成 density_params
                center_weight = params.pop('center_weight', 3.0)  # 从params中移除，避免重复
                center_region = params.pop('center_region', 0.15)
                # 如果已经提供了 density_params，合并参数（独立参数优先）
                if density_params is None or not isinstance(density_params, dict):
                    density_params = {}
                density_params['center_weight'] = center_weight
                density_params['center_region'] = center_region
            else:
                # 如果不是 center_focused，忽略这些参数
                params.pop('center_weight', None)
                params.pop('center_region', None)
        
        # 根据 density_strategy 设置最终的 density_params
        if density_strategy == 'uniform':
            params['density_params'] = None  # uniform不需要参数
        elif density_strategy == 'center_focused':
            if density_params is None:
                # 如果没有提供任何参数，使用默认值
                params['density_params'] = {'center_weight': 3.0, 'center_region': 0.15}
            else:
                # 确保 density_params 是字典且包含所有必需的键
                if not isinstance(density_params, dict):
                    params['density_params'] = {'center_weight': 3.0, 'center_region': 0.15}
                else:
                    density_params.setdefault('center_weight', 3.0)
                    density_params.setdefault('center_region', 0.15)
                    params['density_params'] = density_params
        
        return params
    
    def sweep(self, param_grid, epochs=1000, save_results=True, output_dir=None):
        """
        执行参数扫描
        
        参数:
        param_grid: 参数网格字典，例如：
            {
                'ic_weight': [100, 1000, 5000],
                'learning_rate': [1e-4, 1e-3, 5e-3],
                'density_strategy': ['uniform', 'center_focused'],
                'center_weight': [2.0, 3.0, 5.0],  # center_focused 策略的参数（可选）
                'center_region': [0.1, 0.15, 0.2]  # center_focused 策略的参数（可选）
            }
            注意：当 density_strategy='uniform' 时，center_weight 和 center_region 会被自动忽略
            
            或者使用配对方式（推荐，更灵活）：
            [
                {'ic_weight': 100, 'learning_rate': 1e-3, 'density_strategy': 'uniform', 'density_params': None},
                {'ic_weight': 1000, 'learning_rate': 1e-3, 'density_strategy': 'center_focused', 
                 'center_weight': 3.0, 'center_region': 0.15},  # 使用独立参数
                {'ic_weight': 1000, 'learning_rate': 1e-3, 'density_strategy': 'center_focused', 
                 'density_params': {'center_weight': 5.0, 'center_region': 0.2}}  # 或直接使用 density_params
            ]
        epochs: 每个配置的训练轮数
        save_results: 是否保存结果
        output_dir: 结果保存目录（默认：PINN/sweep_results，与 src/ 并列）
        """
        # 设置默认输出目录（PINN/sweep_results，与 src/ 并列）
        if output_dir is None:
            # 从 src/sweep/ 向上两级到 PINN 目录，然后指向 sweep_results（与 src/ 并列）
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pinn_dir = os.path.dirname(os.path.dirname(current_dir))  # src/sweep -> src -> PINN
            output_dir = os.path.join(pinn_dir, 'sweep_results')
        
        # 创建输出目录
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(output_dir, f"sweep_{timestamp}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 检查param_grid是列表还是字典
        if isinstance(param_grid, list):
            # 直接使用提供的参数组合列表
            param_combinations = param_grid
            total_combinations = len(param_combinations)
        else:
            # 从网格生成所有组合
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            total_combinations = np.prod([len(v) for v in param_values])
            param_combinations = [dict(zip(param_names, combo)) 
                                 for combo in product(*param_values)]
        
        print("=" * 70)
        print("参数扫描开始")
        print("=" * 70)
        print(f"约束类型: {self.constraint_type}")
        print(f"总组合数: {total_combinations}")
        print(f"每个配置训练轮数: {epochs}")
        print("=" * 70)
        
        # 遍历所有参数组合
        self.results = []
        for idx, params in enumerate(param_combinations, 1):
            # 验证和修正参数
            params = self._validate_params(params.copy())
            
            print(f"\n[{idx}/{total_combinations}] 测试参数组合:")
            for key, value in params.items():
                if key != 'density_params' or value is not None:
                    print(f"  {key}: {value}")
            
            result = self.train_and_evaluate(params, epochs=epochs, verbose=True)
            self.results.append(result)
            
            # 每10个组合保存一次（防止数据丢失）
            if save_results and idx % 10 == 0:
                self._save_results()
        
        # 最终保存
        if save_results:
            self._save_results()
            self._generate_report()
            self._plot_results()
        
        print("\n" + "=" * 70)
        print("参数扫描完成！")
        print("=" * 70)
        
        return self.results
    
    def _save_results(self):
        """保存结果到JSON文件"""
        results_file = os.path.join(self.output_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {results_file}")
    
    def _generate_report(self):
        """生成文本报告"""
        # 过滤成功的实验结果
        successful_results = [r for r in self.results if r.get('success', False)]
        
        if not successful_results:
            print("警告: 没有成功的实验结果！")
            return
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame(successful_results)
        
        # 生成报告
        report_file = os.path.join(self.output_dir, 'report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("参数扫描报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"总实验数: {len(self.results)}\n")
            f.write(f"成功实验数: {len(successful_results)}\n")
            f.write(f"失败实验数: {len(self.results) - len(successful_results)}\n\n")
            
            # 最佳配置（按不同指标）
            f.write("最佳配置（按初始条件误差）:\n")
            f.write("-" * 70 + "\n")
            best_ic = df.loc[df['ic_error_total'].idxmin()]
            for key, value in best_ic.items():
                if key not in ['success', 'error']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("最佳配置（按TOV损失）:\n")
            f.write("-" * 70 + "\n")
            best_tov = df.loc[df['tov_loss'].idxmin()]
            for key, value in best_tov.items():
                if key not in ['success', 'error']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 综合评分（可以自定义）
            df['score'] = df['ic_error_total'] * 0.5 + df['tov_loss'] * 0.5
            f.write("最佳配置（综合评分）:\n")
            f.write("-" * 70 + "\n")
            best_overall = df.loc[df['score'].idxmin()]
            for key, value in best_overall.items():
                if key not in ['success', 'error']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 参数影响分析
            f.write("参数影响分析:\n")
            f.write("-" * 70 + "\n")
            numeric_cols = ['ic_weight', 'learning_rate', 'n_points']
            for col in numeric_cols:
                if col in df.columns:
                    correlation = df[col].corr(df['ic_error_total'])
                    f.write(f"  {col} vs IC误差相关性: {correlation:.4f}\n")
        
        print(f"报告已保存到: {report_file}")
    
    def _plot_results(self):
        """绘制结果可视化"""
        successful_results = [r for r in self.results if r.get('success', False)]
        if not successful_results:
            return
        
        df = pd.DataFrame(successful_results)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. IC权重 vs IC误差
        if 'ic_weight' in df.columns:
            ax = axes[0, 0]
            for lr in df['learning_rate'].unique():
                mask = df['learning_rate'] == lr
                ax.scatter(df[mask]['ic_weight'], df[mask]['ic_error_total'],
                          label=f'LR={lr}', alpha=0.6)
            ax.set_xlabel('IC权重')
            ax.set_ylabel('IC总误差')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_title('IC权重 vs IC误差')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. 学习率 vs TOV损失
        if 'learning_rate' in df.columns:
            ax = axes[0, 1]
            for ic_w in df['ic_weight'].unique()[:5]:  # 只显示前5个
                mask = df['ic_weight'] == ic_w
                ax.scatter(df[mask]['learning_rate'], df[mask]['tov_loss'],
                          label=f'IC={ic_w}', alpha=0.6)
            ax.set_xlabel('学习率')
            ax.set_ylabel('TOV损失')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title('学习率 vs TOV损失')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. IC误差 vs TOV损失（帕累托前沿）
        ax = axes[1, 0]
        ax.scatter(df['ic_error_total'], df['tov_loss'], alpha=0.6)
        ax.set_xlabel('IC总误差')
        ax.set_ylabel('TOV损失')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('IC误差 vs TOV损失（帕累托前沿）')
        ax.grid(True, alpha=0.3)
        
        # 4. 训练时间分布
        ax = axes[1, 1]
        ax.hist(df['training_time'], bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('训练时间 (秒)')
        ax.set_ylabel('频数')
        ax.set_title('训练时间分布')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.output_dir, 'results_plots.png')
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"可视化图表已保存到: {plot_file}")
        plt.close()


def main():
    """主函数：示例参数扫描"""
    
    # 创建参数扫描器
    sweep = ParameterSweep(
        constraint_type='soft',
        initial_p=10.0,
        initial_m=1e-10,
        r_initial=0.01
    )
    
    # 方法1：使用参数网格（自动生成所有组合）
    # 注意：density_params会自动匹配density_strategy
    # 可以使用独立的 center_weight 和 center_region 参数来遍历
    param_grid = {
        'ic_weight': [100, 1000, 5000],
        'learning_rate': [1e-4, 1e-3, 5e-3],
        'density_strategy': ['uniform', 'center_focused'],
        'center_weight': [2.0, 3.0, 5.0],  # center_focused 策略的参数
        'center_region': [0.1, 0.15, 0.2],  # center_focused 策略的参数
        'n_points': [100],
        'r_min': [0.01],
        'r_max': [20]
    }
    # 注意：当 density_strategy='uniform' 时，center_weight 和 center_region 会被忽略
    
    # 方法2：手动指定参数组合（更灵活，推荐用于复杂场景）
    # 可以使用独立的 center_weight 和 center_region，或直接使用 density_params
    # param_combinations = [
    #     {
    #         'ic_weight': 100,
    #         'learning_rate': 1e-3,
    #         'density_strategy': 'uniform',
    #         'density_params': None,  # uniform 不需要参数
    #         'n_points': 100,
    #         'r_min': 0.01,
    #         'r_max': 20
    #     },
    #     {
    #         'ic_weight': 1000,
    #         'learning_rate': 1e-3,
    #         'density_strategy': 'center_focused',
    #         'center_weight': 3.0,  # 方式1：使用独立参数
    #         'center_region': 0.15,
    #         'n_points': 100,
    #         'r_min': 0.01,
    #         'r_max': 20
    #     },
    #     {
    #         'ic_weight': 1000,
    #         'learning_rate': 1e-3,
    #         'density_strategy': 'center_focused',
    #         'density_params': {'center_weight': 5.0, 'center_region': 0.2},  # 方式2：直接使用 density_params
    #         'n_points': 100,
    #         'r_min': 0.01,
    #         'r_max': 20
    #     },
    #     # ... 更多组合
    # ]
    
    print("开始参数扫描...")
    print("注意：这是一个示例配置，实际使用时请根据需要调整参数网格")
    
    # 执行扫描
    results = sweep.sweep(
        param_grid=param_grid,
        epochs=500,  # 快速测试用500轮，实际可以更多（如2000-5000）
        save_results=True
        # output_dir 使用默认值（PINN/sweep_results）
    )
    
    # 打印最佳结果
    successful_results = [r for r in results if r.get('success', False)]
    if successful_results:
        df = pd.DataFrame(successful_results)
        print("\n最佳配置（综合评分）:")
        df['score'] = df['ic_error_total'] * 0.5 + df['tov_loss'] * 0.5
        best = df.loc[df['score'].idxmin()]
        print("\n推荐参数配置:")
        print("-" * 70)
        for key in ['ic_weight', 'learning_rate', 'density_strategy', 
                   'density_params', 'ic_error_total', 'tov_loss', 'score']:
            if key in best:
                print(f"  {key}: {best[key]}")
        print("-" * 70)
    else:
        print("\n警告：没有成功的实验结果！")


if __name__ == "__main__":
    main()

