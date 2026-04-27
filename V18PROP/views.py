"""
方向图渲染视图: 独立的绘图函数,返回 matplotlib Figure 对象
"""

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D

import constants as C
from rendering import render_structure, render_overlay
from pattern import pattern_ray_trace, pattern_bare
from math_utils import _hpbw_1d


def embed_figure(fig, parent, root):
    """嵌入 matplotlib Figure 到 tkinter 框架"""
    for w in parent.winfo_children():
        w.destroy()
    c = FigureCanvasTkAgg(fig, master=parent)
    c.draw()
    c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    t_ = NavigationToolbar2Tk(c, parent, pack_toolbar=False)
    t_.update()
    t_.pack(side=tk.BOTTOM, fill=tk.X)


def _normalize(arr):
    arr = np.asarray(arr, dtype=float)
    mx = np.max(arr)
    return arr / mx if mx > 0 else arr


def _polar_radius(arr, floor=-30):
    """线性归一化方向图 → 极坐标半径 0~1"""
    p = _normalize(np.abs(arr))
    return 10 ** (np.clip(10 * np.log10(np.clip(p, 1e-10, None)), floor, 0) / 20)


def _rect_scale_db(arr, floor=-40):
    """直角坐标 dB 曲线值 (dB)"""
    p = _normalize(np.abs(arr))
    return np.clip(10 * np.log10(np.clip(p, 1e-10, None)), floor, 0)


# ======================== 极坐标方向图 ========================

def render_polar(dip_type_name, t_fine, E_e, E_h, metrics):
    fig = Figure(figsize=(7, 6), dpi=100)
    fig.suptitle(f"{dip_type_name} — 全金属模型", fontsize=12, fontweight="bold")

    ax0 = fig.add_subplot(1, 2, 1, projection='polar')
    ax0.plot(t_fine, _polar_radius(E_e), '#00bc8c', lw=1.5)
    ax0.set_title("E 面 (φ=90°)", va="bottom", fontsize=10)
    ax0.set_ylim(0, 1.05)
    ax0.grid(True, alpha=0.3)
    h = metrics.get('HPBW_E', 0)
    if h > 0:
        hh = np.radians(h / 2)
        ax0.plot([np.pi - hh, np.pi + hh], [0.707, 0.707], 'r--', lw=1.5, alpha=0.7)
        ax0.annotate(f"HPBW={h:.1f}°", xy=(np.pi, 0.85), fontsize=9, color='red', ha='center')

    ax1 = fig.add_subplot(1, 2, 2, projection='polar')
    ax1.plot(t_fine, _polar_radius(E_h), '#e67e22', lw=1.5)
    ax1.set_title("H 面 (φ=180° 主瓣)", va="bottom", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    h = metrics.get('HPBW_H', 0)
    if 10 < h < 170:
        hh = np.radians(h / 2)
        ax1.plot([np.pi - hh, np.pi + hh], [0.707, 0.707], 'r--', lw=1.5, alpha=0.7)
        ax1.annotate(f"HPBW={h:.1f}°", xy=(np.pi, 0.85), fontsize=9, color='red', ha='center')

    fig.tight_layout()
    return fig


# ======================== 3D 方向图 ========================

def render_3d(dip_type_name, t_2d, p_2d_array, E_2d):
    fig = Figure(figsize=(8, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    TH, PH = np.meshgrid(t_2d, p_2d_array, indexing='ij')
    P = _normalize(np.abs(E_2d))
    Pd = np.clip(10 * np.log10(np.clip(P, 1e-10, None)), -20, 0)
    n = plt.Normalize(-20, 0)
    c = plt.cm.viridis(n(Pd))
    R = P
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)
    ax.plot_surface(X, Y, Z, facecolors=c, rstride=1, cstride=1, alpha=0.9, lw=0)
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    ax.plot_wireframe(0.3 * np.sin(u) * np.cos(v), 0.3 * np.sin(u) * np.sin(v),
                      0.3 * np.cos(u), color='gray', alpha=0.06, lw=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{dip_type_name} — 全金属模型", fontsize=11, fontweight="bold")
    for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a_.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    m = plt.cm.ScalarMappable(norm=n, cmap=plt.cm.viridis)
    m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1).set_label('归一化方向图 (dB)', fontsize=9)
    fig.tight_layout()
    return fig


# ======================== 天线结构 ========================

def render_structure_view(length, params, qr):
    """天线结构 3D 视图, params 是 var_d 的 get() 快照 (包含 show_struct)"""
    fig = Figure(figsize=(8, 7), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    render_structure(ax, length,
                     params['dish_a'], params['dish_b'], params['dish_f'],
                     params['offset_ang'],
                     params.get('show_shield', True), params['can_D'], params['can_H'],
                     params['cyl_R'], params['cyl_L'], params['cyl_ang'],
                     params['feed_dx'], params['feed_dz'],
                     params.get('can_rot_x', 0), params.get('can_rot_y', 0),
                     nr=qr['dish_nr'] if qr else 20,
                     np_=qr['dish_np'] if qr else 30,
                     cyl_nu=qr['cyl_nu'] if qr else 36,
                     cyl_nv=qr['cyl_nv'] if qr else 28,
                     show_dish=params.get('use_dish', True),
                     show_reflector=params.get('use_reflector', True))
    parts = ['偶极子']
    if params.get('use_dish', True):
        parts.append('抛物面锅')
    if params.get('use_reflector', True):
        parts.append('半圆柱反射板')
    if params.get('show_shield', True):
        parts.append('屏蔽桶')
    ax.set_title("天线结构: " + " + ".join(parts), fontsize=11, fontweight="bold")
    h = [Line2D([0], [0], color='blue', lw=3, label='偶极子')]
    if params.get('use_reflector', True):
        h.append(Line2D([0], [0], color='gold', lw=6, alpha=0.4, label='半圆柱反射板'))
    if params.get('use_dish', True):
        h.append(Line2D([0], [0], color='silver', lw=1, alpha=0.4, label='抛物面锅'))
    if params.get('show_shield', True):
        h.append(Line2D([0], [0], color='gray', lw=1, alpha=0.4, label='屏蔽桶'))
    if len(h) > 1:
        ax.legend(handles=h, loc='upper right', fontsize=7)
    fig.tight_layout()
    return fig


# ======================== 叠加视图 ========================

def render_overlay_view(t, p_, Ep, params, length):
    """结构 + 方向图叠加"""
    fig = Figure(figsize=(9, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    render_overlay(ax, t, p_, Ep,
                   params['dish_a'], params['dish_b'], params['dish_f'],
                   params['offset_ang'],
                   params.get('show_shield', True), params['can_D'], params['can_H'],
                   params['cyl_R'], params['cyl_L'], params['cyl_ang'],
                   length, params['feed_dx'], params['feed_dz'],
                   params.get('can_rot_x', 0), params.get('can_rot_y', 0),
                   show_dish=params.get('use_dish', True),
                   show_reflector=params.get('use_reflector', True))
    ax.set_title("结构 + 方向图叠加 (颜色=增益强度)", fontsize=11, fontweight="bold")
    Pn = _normalize(np.abs(Ep))
    n_ = plt.Normalize(0, 1)
    m_ = plt.cm.ScalarMappable(norm=n_, cmap=plt.cm.hot)
    m_.set_array([])
    fig.colorbar(m_, ax=ax, shrink=0.5, pad=0.1).set_label('归一化增益', fontsize=9)
    fig.tight_layout()
    return fig


# ======================== 点源增益分析 ========================

def render_point_source(t_1d, t_2d, p_2d, E_2d, E_e, E_h, metrics, params,
                        dip_type_name, length, stage_data=None):
    """点源增益分析: 4级渐进结构分解 (由近及远)"""
    fig = Figure(figsize=(12, 8.5), dpi=100)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3.5, 2.5])

    # ===== (0,0) 3D 最终方向图 =====
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    TH, PH = np.meshgrid(t_2d, p_2d, indexing='ij')
    P = _normalize(np.abs(E_2d))
    Pd = np.clip(10 * np.log10(np.clip(P, 1e-10, None)), -20, 0)
    n_cmap = plt.Normalize(-20, 0)
    colors = plt.cm.viridis(n_cmap(Pd))
    R = P
    X = R * np.sin(TH) * np.cos(PH)
    Y = R * np.sin(TH) * np.sin(PH)
    Z = R * np.cos(TH)
    ax_3d.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, alpha=0.9, lw=0)
    ax_3d.scatter([0], [0], [0], color='red', s=120, marker='o', zorder=10, label='点源馈点')
    ax_3d.text(0, 0, 0.05, ' 馈点', color='red', fontsize=9, fontweight='bold')
    ax_3d.set_xlim(-1.1, 1.1); ax_3d.set_ylim(-1.1, 1.1); ax_3d.set_zlim(-1.1, 1.1)
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
    ax_3d.set_title("最终方向图 (全结构)", fontsize=10, fontweight="bold")
    for a_ in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        a_.pane.fill = False
    ax_3d.set_xticks([]); ax_3d.set_yticks([]); ax_3d.set_zticks([])
    ax_3d.legend(fontsize=7, loc='upper right')
    m_sm = plt.cm.ScalarMappable(norm=n_cmap, cmap=plt.cm.viridis)
    m_sm.set_array([])
    fig.colorbar(m_sm, ax=ax_3d, shrink=0.5, pad=0.05).set_label('归一化增益 (dB)', fontsize=7)

    # ===== 4 级渐进极坐标图 + 指标表 =====
    if stage_data:
        polar_pos = [(0, 1), (0, 2), (1, 0), (1, 1)]
        c_plane = ['#00bc8c', '#e67e22']
        for idx, sd in enumerate(stage_data):
            row, col = polar_pos[idx]
            ax = fig.add_subplot(gs[row, col], projection='polar')
            ax.plot(t_1d, _polar_radius(sd['E_e'], floor=-30), c_plane[0], lw=1.5, label='E面')
            ax.plot(t_1d, _polar_radius(sd['E_h'], floor=-30), c_plane[1], lw=1.5, label='H面')
            ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
            ax.set_title(sd['name'], fontsize=9, fontweight="bold", pad=8)
            for key, clr in [('HPBW_E', '#00bc8c'), ('HPBW_H', '#e67e22')]:
                h = sd.get(key, 0)
                if 10 < h < 170:
                    hh = np.radians(h / 2)
                    ax.plot([np.pi - hh, np.pi + hh], [0.707, 0.707],
                            color=clr, ls='--', lw=1, alpha=0.6)
            if idx == 0:
                ax.legend(fontsize=6, loc='upper right')
        # 指标演进表
        ax_t = fig.add_subplot(gs[1, 2])
        ax_t.axis('off')
        tbl_d = [[sd['name'], f"{sd['D_dBi']:.1f}", f"{sd['HPBW_E']:.1f}°",
                  f"{sd['HPBW_H']:.1f}°", f"{sd['FB_dB']:.1f}"] for sd in stage_data]
        tbl = ax_t.table(cellText=tbl_d,
            colLabels=['阶段', 'D(dBi)', 'HPBW_E', 'HPBW_H', 'F/B(dB)'],
            loc='center', cellLoc='center', colWidths=[0.24, 0.14, 0.14, 0.14, 0.16])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.9)
        for j in range(5):
            tbl[0, j].set_facecolor('#094771')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(1, 5):
            tbl[i, 0].set_facecolor('#2d2d2d')
        ax_t.set_title("指标演进 (逐级叠加)", fontsize=10, fontweight="bold", pad=12)
    else:
        ax_fb = fig.add_subplot(gs[:, 1:])
        ax_fb.axis('off')
        ax_fb.text(0.5, 0.5, "计算阶段数据中...\n请重新运行仿真",
                  ha='center', va='center', fontsize=12, color='#888888')

    fig.suptitle("点源增益分析 — 4级渐进结构分解 (由近及远)",
                fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ======================== 配置对比 ========================

def render_comparison(params, dip_type_name, length):
    """多口径对比: 0.8× / 1.0× / 1.2×"""
    a = params['dish_a']
    b = params['dish_b']
    f = params['dish_f']
    d0 = params['ref_dist']
    feed_dx = params['feed_dx']
    feed_dz = params['feed_dz']
    shield_D = params['can_D']
    shield_H = params['can_H']
    oa = params['offset_ang']
    ur = params.get('use_reflector', True)
    ud = params.get('use_dish', True)
    us = params.get('use_shield', True)

    scales = [0.8, 1.0, 1.2]
    D0 = 2 * np.sqrt(a * b)
    Ds = [D0 * s for s in scales]
    nm = [f"D={D:.2f}m" for D in Ds]

    def _mm():
        results = []
        tf = np.linspace(0, np.pi, 360)
        for s in scales:
            aa = a * s
            bb = b * s
            def rt(t, p, L, dd=aa, ee=bb):
                return pattern_ray_trace(t, p, L, d0, dd, ee, f,
                    feed_dx, feed_dz, shield_D,
                    shield_H=shield_H, offset_ang=oa,
                    use_reflector=ur, use_dish=ud, use_shield=us)
            D_dBi = 10 * np.log10(4 * np.pi * 0.55 * np.pi * aa * bb / C.LAM**2)
            Ee = np.abs(rt(tf, np.pi / 2, length))
            Eh = np.abs(rt(tf, np.pi, length))
            pe = _normalize(Ee)
            ph = _normalize(Eh)
            he = _hpbw_1d(tf, pe**2)
            hh = _hpbw_1d(tf, ph**2)
            bs_t = np.radians(60)
            Pf = np.abs(rt(bs_t, np.pi, length))**2
            Pb = np.abs(rt(np.pi - bs_t, 0.0, length))**2
            fb = 10 * np.log10(Pf / Pb) if Pb > 0 else 40.0
            results.append({'D_dBi': D_dBi, 'HPBW_E': he, 'HPBW_H': hh, 'FB_dB': fb})
        return results
    mm = _mm()

    fig = Figure(figsize=(12, 13), dpi=100)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("1420 MHz PO 光线追踪 — 口径对比", fontsize=14, fontweight="bold")
    cl = ['#00bc8c', '#3498db', '#e67e22']
    th = np.linspace(0, 2 * np.pi, 720)

    # E面
    ax_e = fig.add_subplot(gs[0, 0], projection='polar')
    for i, s in enumerate(scales):
        aa = a * s; bb = b * s
        def _po(t, p, dd=aa, ee=bb):
            return pattern_ray_trace(t, p, length, d0, dd, ee, f,
                feed_dx, feed_dz, shield_D,
                shield_H=shield_H, offset_ang=oa,
                use_reflector=ur, use_dish=ud, use_shield=us)
        E = np.abs(_po(th, np.pi / 2))
        ax_e.plot(th, _polar_radius(E), color=cl[i], lw=1.2, label=nm[i])
    ax_e.set_title("E 面 (φ=90°)", fontsize=10)
    ax_e.set_ylim(0, 1.05)
    ax_e.legend(fontsize=7)

    # H面
    ax_h = fig.add_subplot(gs[0, 1], projection='polar')
    for i, s in enumerate(scales):
        aa = a * s; bb = b * s
        def _po(t, p, dd=aa, ee=bb):
            return pattern_ray_trace(t, p, length, d0, dd, ee, f,
                feed_dx, feed_dz, shield_D,
                shield_H=shield_H, offset_ang=oa,
                use_reflector=ur, use_dish=ud, use_shield=us)
        E = np.abs(_po(th, 0))
        ax_h.plot(th, _polar_radius(E), color=cl[i], lw=1.2, label=nm[i])
    ax_h.set_title("H 面 (φ=0)", fontsize=10)
    ax_h.set_ylim(0, 1.05)
    ax_h.legend(fontsize=7)

    # 指标对比表
    ax_t = fig.add_subplot(gs[0, 2])
    ax_t.axis('off')
    d_ = [[nm[i], f"{mm[i]['D_dBi']:.1f}", f"{mm[i]['HPBW_E']:.0f}°",
           f"{mm[i]['HPBW_H']:.0f}°", f"{mm[i]['FB_dB']:.1f}"] for i in range(3)]
    t_ = ax_t.table(cellText=d_,
                    colLabels=["口径", "D(dBi)", "HPBW_E", "HPBW_H", "F/B(dB)"],
                    loc='center', cellLoc='center', colWidths=[0.22, 0.16, 0.16, 0.16, 0.16])
    t_.auto_set_font_size(False)
    t_.set_fontsize(9)
    t_.scale(1, 1.8)
    ax_t.set_title("指标对比", fontsize=11, fontweight="bold", pad=20)

    # 增益 vs 口径
    ax_g = fig.add_subplot(gs[1, :])
    Dr = np.linspace(0.3, 1.0, 20)
    gs_plot = [10 * np.log10(4 * np.pi * 0.55 * np.pi * (d / 2)**2 / C.LAM**2) for d in Dr]
    ax_g.plot(Dr, gs_plot, '#e74c3c', lw=2)
    ax_g.set_xlabel("有效口径 D (m)", fontsize=10)
    ax_g.set_ylabel("指向性 (dBi)", fontsize=10)
    ax_g.set_title("PO 模型增益 vs 口径 (理论极限)", fontsize=11, fontweight="bold")
    ax_g.grid(True, alpha=0.3)

    # 频率-增益扫频
    ax_f = fig.add_subplot(gs[2, :])
    f_sweep = np.logspace(np.log10(0.5), np.log10(1700), 300)
    for i, s in enumerate(scales):
        aa_s = a * s; bb_s = b * s
        D_sweep = [10 * np.log10(4 * np.pi * 0.55 * np.pi * aa_s * bb_s /
                                 (C.C / (f_mhz * 1e6))**2) for f_mhz in f_sweep]
        ax_f.semilogx(f_sweep, D_sweep, color=cl[i], lw=1.5, label=nm[i])
    ax_f.axvline(C.FREQ / 1e6, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax_f.annotate(f"{C.FREQ/1e6:.0f} MHz\n(当前)", xy=(C.FREQ/1e6, 0),
                 fontsize=8, color='gray', ha='center', va='bottom')
    ax_f.set_xlabel("频率 (MHz)", fontsize=10)
    ax_f.set_ylabel("指向性 (dBi)", fontsize=10)
    ax_f.set_title("同一天线 — 频率 vs 最大增益/指向性 (理论极限)",
                   fontsize=11, fontweight="bold")
    ax_f.legend(fontsize=8)
    ax_f.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    return fig


# ======================== 导出报告 ========================

def render_export_report(params, dip_type_name, length, save_path, qr):
    """导出完整报告 (另存为 PNG/PDF)"""
    a = params['dish_a']
    b = params['dish_b']
    f = params['dish_f']
    d_ref = params['ref_dist']
    feed_dx = params['feed_dx']
    feed_dz = params['feed_dz']
    shield_D = params['can_D']
    shield_H = params['can_H']
    oa = params['offset_ang']
    rx = params.get('can_rot_x', 0)
    ry = params.get('can_rot_y', 0)
    ur = params.get('use_reflector', True)
    ud = params.get('use_dish', True)
    us = params.get('use_shield', True)
    show_shield = params.get('show_shield', True)

    def rt(t, p, L):
        return pattern_ray_trace(t, p, L, d_ref, a, b, f,
            feed_dx, feed_dz, shield_D, shield_H=shield_H,
            offset_ang=oa, n_uv=60,
            use_reflector=ur, use_dish=ud, use_shield=us)

    th = np.linspace(0, np.pi, qr['n_pol'])
    Ee = np.abs(rt(th, np.pi / 2, length))
    Eh = np.abs(rt(th, np.pi, length))
    Pe = _normalize(Ee)
    Ph = _normalize(Eh)

    D_dBi = 10 * np.log10(4 * np.pi * 0.55 * np.pi * a * b / C.LAM**2)
    hpbw_e = _hpbw_1d(th, Pe**2)
    hpbw_h = _hpbw_1d(th, Ph**2)
    bs_t = np.radians(60)
    Pf = np.abs(rt(bs_t, np.pi, length))**2
    Pb = np.abs(rt(np.pi - bs_t, 0.0, length))**2
    fb = 10 * np.log10(Pf / Pb) if Pb > 0 else 40.0

    fig = Figure(figsize=(16, 24), dpi=120)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 1.4, 1.4])

    # (0,0) 极坐标方向图
    a0 = fig.add_subplot(gs[0, 0], projection='polar')
    a0.plot(th, _polar_radius(Ee), '#00bc8c', lw=1.2, label='E 面 (φ=90°)')
    a0.plot(th, _polar_radius(Eh), '#e67e22', lw=1.2, label='H 面 (φ=180° 主瓣)')
    a0.set_ylim(0, 1.05)
    a0.grid(True, alpha=0.3)
    a0.set_title("极坐标方向图", fontsize=12, fontweight="bold", pad=12)
    a0.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.3, 1.0))
    if hpbw_e > 0:
        hh = np.radians(hpbw_e / 2)
        a0.plot([np.pi - hh, np.pi + hh], [0.707] * 2, 'b--', lw=1.5, alpha=0.5)
        a0.annotate(f"E HPBW={hpbw_e:.1f}°", xy=(np.pi, 0.8), fontsize=8, color='blue', ha='center')
    if hpbw_h > 0:
        hh = np.radians(hpbw_h / 2)
        a0.plot([np.pi - hh, np.pi + hh], [0.707] * 2, 'r--', lw=1.5, alpha=0.5)
        a0.annotate(f"H HPBW={hpbw_h:.1f}°", xy=(np.pi, 0.7), fontsize=8, color='red', ha='center')

    # (0,1) 3D 方向图
    a1 = fig.add_subplot(gs[0, 1], projection='3d')
    t3 = np.linspace(0, np.pi, qr['n_3d_t'])
    p3 = np.linspace(0, 2 * np.pi, qr['n_3d_p'])
    T3, P3 = np.meshgrid(t3, p3, indexing='ij')
    E3 = np.abs(rt(T3, P3, length))
    P3d = _normalize(E3)
    Pd = np.clip(10 * np.log10(np.clip(P3d, 1e-10, None)), -20, 0)
    n_c = plt.Normalize(-20, 0)
    c_c = plt.cm.viridis(n_c(Pd))
    R3 = P3d
    X3 = R3 * np.sin(T3) * np.cos(P3)
    Y3 = R3 * np.sin(T3) * np.sin(P3)
    Z3 = R3 * np.cos(T3)
    a1.plot_surface(X3, Y3, Z3, facecolors=c_c, rstride=2, cstride=2, alpha=0.9, lw=0)
    a1.set_xlim(-1.1, 1.1)
    a1.set_ylim(-1.1, 1.1)
    a1.set_zlim(-1.1, 1.1)
    a1.set_title("3D 方向图", fontsize=12, fontweight="bold")
    for ax_ in [a1.xaxis, a1.yaxis, a1.zaxis]:
        ax_.pane.fill = False
    a1.set_xticks([]); a1.set_yticks([]); a1.set_zticks([])

    # (1,0) 天线结构
    a2 = fig.add_subplot(gs[1, 0], projection='3d')
    render_structure(a2, length, a, b, f, oa, show_shield, shield_D, shield_H,
                     params['cyl_R'], params['cyl_L'], params['cyl_ang'],
                     feed_dx, feed_dz, rx, ry,
                     show_dish=ud, show_reflector=ur)
    a2.set_title("天线结构", fontsize=12, fontweight="bold")

    # (1,1) 叠加视图
    a3 = fig.add_subplot(gs[1, 1], projection='3d')
    t_ = np.linspace(0, np.pi, qr['n_3d_t'])
    p_ = np.linspace(0, 2 * np.pi, qr['n_3d_p'])
    TT, PP = np.meshgrid(t_, p_, indexing='ij')
    Ep = rt(TT, PP, length)
    render_overlay(a3, t_, p_, Ep, a, b, f, oa, show_shield, shield_D, shield_H,
                   params['cyl_R'], params['cyl_L'], params['cyl_ang'],
                   length, feed_dx, feed_dz, rx, ry,
                   show_dish=ud, show_reflector=ur)
    a3.set_title("结构 + 方向图叠加", fontsize=12, fontweight="bold")

    # (2,0) 指标汇总
    a4 = fig.add_subplot(gs[2, 0])
    a4.axis('off')
    txt = (f"频率: {C.FREQ/1e6:.1f} MHz  λ={C.LAM*100:.1f} cm\n"
           f"偶极子: {dip_type_name}  L={length*1000:.1f} mm\n"
           f"锅: a={a*100:.1f}cm b={b*100:.1f}cm  f={f*100:.1f}cm\n"
           f"偏馈角: {oa:.1f}°  反射板距离: {d_ref*100:.1f}cm\n"
           f"桶: D={shield_D*100:.1f}cm  H={shield_H*100:.1f}cm\n"
           f"旋转: X={rx:.1f}°  Y={ry:.1f}°\n\n"
           f"✦ 指向性: {D_dBi:.2f} dBi\n"
           f"✦ E面 HPBW: {hpbw_e:.1f}°\n"
           f"✦ H面 HPBW: {hpbw_h:.1f}°\n"
           f"✦ 前后比 F/B: {fb:.1f} dB\n"
           f"✦ 有效口径: {2 * np.sqrt(a * b):.3f}m  "
           f"f/D={(f / (2 * np.sqrt(a * b)) if a * b > 0 else 0):.3f}")
    a4.text(0.05, 0.95, txt, transform=a4.transAxes,
           fontsize=13, va='top', color='#d4d4d4',
           bbox=dict(boxstyle='round', facecolor='#2d2d2d',
                     edgecolor='#666666', alpha=0.95, pad=0.6))

    # (2,1) 增益 vs 口径
    a5 = fig.add_subplot(gs[2, 1])
    Dr = np.linspace(0.3, 1.0, 50)
    G = [10 * np.log10(4 * np.pi * 0.55 * np.pi * (d / 2)**2 / C.LAM**2) for d in Dr]
    a5.plot(Dr, G, '#e74c3c', lw=2)
    if a * b > 0:
        D_eff = 2 * np.sqrt(a * b)
        G_pt = 10 * np.log10(4 * np.pi * 0.55 * np.pi * (D_eff / 2)**2 / C.LAM**2)
        a5.plot(D_eff, G_pt, 'bo', markersize=8)
        a5.annotate(f"当前 D={D_eff:.2f}m\nG={G_pt:.1f}dBi",
                   xy=(D_eff, G_pt), fontsize=9, color='blue',
                   xytext=(5, 5), textcoords='offset points')
    a5.set_xlabel("有效口径 D (m)", fontsize=10)
    a5.set_ylabel("指向性 (dBi)", fontsize=10)
    a5.set_title("增益 vs 口径 (理论极限)", fontsize=11, fontweight="bold")
    a5.grid(True, alpha=0.3)

    custom_len_mm = params.get('custom_len', 0)
    dipole_label = dip_type_name
    if custom_len_mm > 0:
        dipole_label += f" (自定义{length*1000:.1f}mm)"
    comp_parts = []
    if ud: comp_parts.append('抛物面')
    if ur: comp_parts.append('反射板')
    if us: comp_parts.append('屏蔽桶')
    comp_str = " + ".join(comp_parts) if comp_parts else "裸偶极子"

    # (3,:) 频率-增益扫频
    a6 = fig.add_subplot(gs[3, :])
    f_sweep = np.logspace(np.log10(0.5), np.log10(1700), 300)
    D_sweep = [10 * np.log10(4 * np.pi * 0.55 * np.pi * a * b /
                             (C.C / (f_mhz * 1e6))**2) for f_mhz in f_sweep]
    a6.semilogx(f_sweep, D_sweep, '#00bc8c', lw=2)
    a6.axvline(C.FREQ / 1e6, color='gray', ls='--', lw=1, alpha=0.5)
    a6.annotate(f"{C.FREQ/1e6:.0f} MHz (当前)", xy=(C.FREQ/1e6, 0),
               fontsize=9, color='gray', ha='center', va='bottom')
    a6.set_xlabel("频率 (MHz)", fontsize=11)
    a6.set_ylabel("指向性 (dBi)", fontsize=11)
    a6.set_title("频率 vs 最大增益 / 指向性 (理论极限)",
                 fontsize=12, fontweight="bold")
    a6.grid(True, alpha=0.3, which='both')

    fig.suptitle(f"{C.FREQ/1e6:.1f} MHz 天线仿真报告 — {dipole_label} | {comp_str}",
                fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    is_pdf = save_path.lower().endswith('.pdf')

    # ---- Stage data (for page 2 & 3) ----
    th_s = np.linspace(0, np.pi, 360)
    n_uv_s = max(qr['n_uv'] // 2, 14)
    def s_rt(t, p, L, use_r, use_d, use_s):
        return pattern_ray_trace(t, p, L, d_ref, a, b, f,
            feed_dx, feed_dz, shield_D, shield_H=shield_H,
            offset_ang=oa, n_uv=n_uv_s,
            use_reflector=use_r, use_dish=use_d, use_shield=use_s)
    stage_cfgs = [
        ("① 裸偶极子", False, False, False),
        ("② +反射板",  True,  False, False),
        ("③ +抛物面", True,  True,  False),
        ("④ +屏蔽桶", True,  True,  True),
    ]
    st_data = []
    s_bs_t = np.radians(60)
    for sn, ur, ud, us in stage_cfgs:
        Ee_s = np.abs(s_rt(th_s, np.pi/2, length, ur, ud, us))
        Eh_s = np.abs(s_rt(th_s, np.pi, length, ur, ud, us))
        Pe_s = Ee_s / np.max(Ee_s) if np.max(Ee_s) > 0 else Ee_s
        Ph_s = Eh_s / np.max(Eh_s) if np.max(Eh_s) > 0 else Eh_s
        he_s = _hpbw_1d(th_s, Pe_s**2)
        hh_s = _hpbw_1d(th_s, Ph_s**2)
        Pf_s = np.abs(s_rt(s_bs_t, np.pi, length, ur, ud, us))**2
        Pb_s = np.abs(s_rt(np.pi - s_bs_t, 0.0, length, ur, ud, us))**2
        fb_s = 10 * np.log10(Pf_s / Pb_s) if Pb_s > 0 else 40.0
        Pavg_s = (Ee_s**2 + Eh_s**2) / 2
        integ_s = Pavg_s * np.sin(th_s)
        Prad_s = np.trapezoid(integ_s, th_s) * 2 * np.pi
        Umax_s = np.max(Pavg_s)
        Dest_s = 4 * np.pi * Umax_s / Prad_s if Prad_s > 0 else 1
        st_data.append(dict(name=sn, E_e=Ee_s, E_h=Eh_s,
            D_dBi=10*np.log10(Dest_s), HPBW_E=he_s, HPBW_H=hh_s, FB_dB=fb_s))

    # ---- Page 2: Stage Analysis ----
    fig2 = Figure(figsize=(16, 20), dpi=120)
    gs2 = GridSpec(3, 2, figure=fig2, height_ratios=[2, 2, 1.5])
    for i, sd in enumerate(st_data):
        ax = fig2.add_subplot(gs2[i//2, i%2], projection='polar')
        ax.plot(th_s, _polar_radius(sd['E_e'], floor=-30), '#00bc8c', lw=1.5, label='E面')
        ax.plot(th_s, _polar_radius(sd['E_h'], floor=-30), '#e67e22', lw=1.5, label='H面')
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
        ax.set_title(f"阶段 {sd['name']}", fontsize=11, fontweight="bold", pad=10)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
    ax_t2 = fig2.add_subplot(gs2[2, :])
    ax_t2.axis('off')
    tbl_d2 = [[sd['name'], f"{sd['D_dBi']:.1f}", f"{sd['HPBW_E']:.1f}°",
               f"{sd['HPBW_H']:.1f}°", f"{sd['FB_dB']:.1f}"] for sd in st_data]
    tbl2 = ax_t2.table(cellText=tbl_d2,
        colLabels=['阶段', 'D(dBi)', 'HPBW_E', 'HPBW_H', 'F/B(dB)'],
        loc='center', cellLoc='center', colWidths=[0.2, 0.15, 0.15, 0.15, 0.15])
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(11); tbl2.scale(1, 2.2)
    for j in range(5):
        tbl2[0, j].set_facecolor('#094771')
        tbl2[0, j].set_text_props(color='white', fontweight='bold')
    ax_t2.set_title("阶段指标演进", fontsize=14, fontweight="bold", pad=15)
    fig2.suptitle(f"{C.FREQ/1e6:.1f} MHz 天线仿真 — 渐进结构分析 | {dipole_label}",
                 fontsize=16, fontweight="bold", y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- Page 3: Progressive Optimization ----
    fig3 = render_progressive_opt(th_s, st_data, figsize=(16, 12))

    if is_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(save_path) as pdf:
            pdf.savefig(fig, dpi=200, bbox_inches='tight')
            pdf.savefig(fig2, dpi=200, bbox_inches='tight')
            pdf.savefig(fig3, dpi=200, bbox_inches='tight')
        plt.close(fig); plt.close(fig2); plt.close(fig3)
    else:
        # Composite PNG: stack all 3 pages vertically
        import io
        arrs = []
        for f in [fig, fig2, fig3]:
            buf = io.BytesIO()
            f.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.3)
            buf.seek(0)
            arrs.append(plt.imread(buf))
        min_w = min(a.shape[1] for a in arrs)
        arrs = [a[:, :min_w] for a in arrs]
        plt.imsave(save_path, np.vstack(arrs), dpi=150)
        plt.close(fig); plt.close(fig2); plt.close(fig3)


# ======================== 逐级优化 ========================

def render_progressive_opt(t_1d, stage_data, figsize=(10, 8.5)):
    """逐级优化视图: 4 级极坐标图(每阶段自包含,无跨级叠加)+指标演进表"""
    fig = Figure(figsize=figsize, dpi=100)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1.2])

    if not stage_data:
        ax_fb = fig.add_subplot(gs[:, :])
        ax_fb.axis('off')
        ax_fb.text(0.5, 0.5, "计算阶段数据中...\n请重新运行仿真",
                  ha='center', va='center', fontsize=12, color='#888888')
        fig.suptitle("逐级优化 — 渐进结构分解", fontsize=12, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    c_plane = ['#00bc8c', '#e67e22']
    for idx, sd in enumerate(stage_data):
        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(gs[row, col], projection='polar')
        ax.plot(t_1d, _polar_radius(sd['E_e'], floor=-30), c_plane[0], lw=1.5, label='E面')
        ax.plot(t_1d, _polar_radius(sd['E_h'], floor=-30), c_plane[1], lw=1.5, label='H面')
        ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
        ax.set_title(sd['name'], fontsize=10, fontweight="bold", pad=10)
        for key, clr in [('HPBW_E', '#00bc8c'), ('HPBW_H', '#e67e22')]:
            h = sd.get(key, 0)
            if 10 < h < 170:
                hh = np.radians(h / 2)
                ax.plot([np.pi - hh, np.pi + hh], [0.707, 0.707],
                        color=clr, ls='--', lw=1.2, alpha=0.6)
                ax.annotate(f"{h:.0f}°", xy=(np.pi, 0.75), fontsize=7,
                           color=clr, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.1', facecolor='#1e1e1e', alpha=0.5))
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    # 指标演进表
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.axis('off')
    tbl_d = [[sd['name'], f"{sd['D_dBi']:.1f}", f"{sd['HPBW_E']:.1f}°",
              f"{sd['HPBW_H']:.1f}°", f"{sd['FB_dB']:.1f}"] for sd in stage_data]
    tbl = ax_t.table(cellText=tbl_d,
        colLabels=['阶段', 'D (dBi)', 'HPBW_E', 'HPBW_H', 'F/B (dB)'],
        loc='center', cellLoc='center', colWidths=[0.24, 0.14, 0.14, 0.14, 0.16])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
    for j in range(5):
        tbl[0, j].set_facecolor('#094771')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, 5):
        tbl[i, 0].set_facecolor('#2d2d2d')

    fig.suptitle("逐级优化 — 各阶段独立方向图 (无跨级叠加,渲染更轻量)",
                fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
