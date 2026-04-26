"""物理常量、频率/波长参数及 GPU 检测"""

import numpy as np

C = 299792458  # 光速 (m/s)

# 当前工作频率及波长 — 通过 set_frequency() 修改
# 注意: 其他模块需用 import constants; constants.LAM 引用,
# 不能用 from constants import LAM (创建不可变副本)
FREQ = 1420e6       # Hz
LAM = C / FREQ      # m
BETA = 2 * np.pi / LAM

def set_frequency(freq_hz):
    """更新工作频率，自动重算波长、波数及默认参数"""
    global FREQ, LAM, BETA, DEF_RR
    FREQ = float(freq_hz)
    LAM = C / FREQ
    BETA = 2 * np.pi / LAM
    DEF_RR = LAM / 4

def get_dipole_types():
    """返回当前波长下的偶极子类型→长度映射"""
    return {"半波长 (λ/2)": LAM / 2, "全波长 (λ)": LAM}


# ======================== GPU 加速 (可选) ========================
_GPU_AVAIL = False
_HAS_NUMBA = False
try:
    import cupy as cp
    _GPU_AVAIL = True
except ImportError:
    try:
        import numba
        _HAS_NUMBA = True
    except ImportError:
        pass


def _to_np(arr):
    """将 CuPy 数组转回 NumPy (如需要)"""
    if _GPU_AVAIL and hasattr(arr, 'device'):
        return cp.asnumpy(arr)
    return arr


# ======================== 默认结构参数 ========================
DEF_A = 0.250     # 椭圆锅长轴半长 (m)
DEF_B = 0.235     # 椭圆锅短轴半长 (m)
DEF_F = 0.32      # 焦距 (m)
DEF_ANG = 60      # 偏馈角 (度)
DEF_RR = LAM / 4  # 半圆柱反射板半径
DEF_RL = 0.15     # 半圆柱反射板长度 (m)
