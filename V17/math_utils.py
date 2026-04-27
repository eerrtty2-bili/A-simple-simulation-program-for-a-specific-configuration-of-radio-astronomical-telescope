"""数学工具: Bessel J1, HPBW 提取, 指标计算"""

import numpy as np


def bessel_j1(x):
    """一阶第一类贝塞尔函数 J1(x) — 级数展开 + 渐近近似"""
    x = np.asarray(x, dtype=float)
    is_scalar = x.ndim == 0
    x = np.atleast_1d(x)
    r = np.zeros_like(x)
    # 小参数: 级数展开
    m = x <= 15
    if np.any(m):
        xs = x[m]
        t = xs / 2
        acc = np.zeros_like(xs)
        for k in range(120):
            acc += t
            t *= -xs * xs / (4 * (k + 1) * (k + 2))
            if np.all(np.abs(t) < 1e-14):
                break
        r[m] = acc
    # 大参数: 渐近近似
    if np.any(~m):
        xl = x[~m]
        r[~m] = np.sqrt(2 / (np.pi * xl)) * np.cos(xl - 3 * np.pi / 4)
    return r.item() if is_scalar else r


def _hpbw_1d(theta, Pn):
    """从 1D 功率方向图提取半功率波束宽度 (HPBW)"""
    ab = Pn >= 0.5
    if not np.any(ab):
        return 0.0
    idx = np.where(ab)[0]
    g = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    ml = max(g, key=len) if g else idx
    if len(ml) < 2:
        return 0.0
    w = np.degrees(theta[ml[-1]] - theta[ml[0]])
    if len(ml) < len(idx) and (ml[0] == 0 or ml[-1] == len(theta) - 1):
        w *= 2
    return w


def compute_metrics(func, length, nt=360, np_=720, **kw):
    """全方位指标: 指向性, HPBW, F/B"""
    t = np.linspace(0, np.pi, nt)
    p = np.linspace(0, 2 * np.pi, np_)
    TH, PH = np.meshgrid(t, p, indexing='ij')
    E = func(TH, PH, length, **kw)
    P = np.abs(E) ** 2
    Pm = np.max(P)
    if Pm <= 0:
        return {'D_dBi': 0, 'HPBW_E': 0, 'HPBW_H': 0, 'FB_dB': 0}
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            from scipy.integrate import simpson
            Pr = simpson(simpson(P * np.sin(TH), t, axis=0), p)
        except Exception:
            Pr = np.trapz(np.trapz(P * np.sin(TH), t, axis=0), p)
    if Pr <= 0:
        return {'D_dBi': 0, 'HPBW_E': 0, 'HPBW_H': 0, 'FB_dB': 0}
    D = 10 * np.log10(4 * np.pi * Pm / Pr)

    HE = _hpbw_1d(t, np.abs(func(t, np.pi / 2, length, **kw)) ** 2)
    HH = _hpbw_1d(t, np.abs(func(t, 0, length, **kw)) ** 2)

    it = np.argmin(np.abs(t - np.pi / 2))
    ip = np.argmin(np.abs(p - np.pi))
    Pf = np.abs(func(t[it], p[0], length, **kw)) ** 2
    Pb = np.abs(func(t[it], p[ip], length, **kw)) ** 2
    FB = 10 * np.log10(Pf / Pb) if Pb > 0 else 40
    return {'D_dBi': D, 'HPBW_E': HE, 'HPBW_H': HH, 'FB_dB': FB}
