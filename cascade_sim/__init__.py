"""cascade_sim 包初始化。

numpy 2.0.2 在 macOS Accelerate BLAS 下会对 matmul 发出假的
"divide by zero encountered in matmul" 警告(已知 BLAS bug),在此屏蔽。
"""
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
