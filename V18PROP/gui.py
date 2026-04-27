"""
GUI 主界面: 1420 MHz 天线仿真器
"""
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import constants as C
from pattern import pattern_ray_trace, pattern_bare
from math_utils import _hpbw_1d
from views import (embed_figure, render_polar, render_3d,
                   render_structure_view, render_overlay_view,
                   render_point_source, render_progressive_opt,
                   render_comparison, render_export_report)

warnings.filterwarnings('ignore')

# 标签页索引
TAB_POLAR, TAB_3D, TAB_STRUCT, TAB_OVERLAY, TAB_POINT_SRC, TAB_COMPARE, TAB_PROGRESSIVE = range(7)
_TAB_FRAMES = None  # set in _make_tabs


class App:
    def __init__(self, root):
        self.root = root
        root.title("1420 MHz 天线仿真器 v2.1")
        root.geometry("1480x960")
        root.minsize(1200, 750)

        self.dip_type = tk.StringVar(value="半波长 (λ/2)")
        self.freq_var = tk.StringVar(value=f"{C.FREQ/1e6:.1f}")
        self.freq_display = None

        self.var_d = {}
        self._make_vars()
        self._last_pat = None

        self._update_timer = None
        self._compute_gen = 0
        self._pending_gen = None
        self._compute_lock = threading.Lock()
        self._geo_cache = {}
        self._geo_cache_key = None
        self._full_pat_data = None

        # 懒渲染缓存
        self._cache = {}
        self._rendered = set()

        self._setup_theme()
        self._build_ui()
        self._update()

    def _make_vars(self):
        V = self.var_d
        V['ref_dist']   = tk.DoubleVar(value=C.LAM/4)
        V['can_D']      = tk.DoubleVar(value=2*C.DEF_A)
        V['can_H']      = tk.DoubleVar(value=0.3)
        V['cyl_R']      = tk.DoubleVar(value=C.DEF_RR)
        V['cyl_L']      = tk.DoubleVar(value=C.DEF_RL)
        V['cyl_ang']    = tk.DoubleVar(value=C.DEF_ANG)
        V['can_rot_x']  = tk.DoubleVar(value=0.0)
        V['can_rot_y']  = tk.DoubleVar(value=0.0)
        V['dish_a']     = tk.DoubleVar(value=C.DEF_A)
        V['dish_b']     = tk.DoubleVar(value=C.DEF_B)
        V['dish_f']     = tk.DoubleVar(value=C.DEF_F)
        V['offset_ang'] = tk.DoubleVar(value=C.DEF_ANG)
        V['feed_dx']    = tk.DoubleVar(value=0.0)
        V['feed_dz']    = tk.DoubleVar(value=0.0)
        V['show_struct'] = tk.BooleanVar(value=True)
        V['show_shield'] = tk.BooleanVar(value=True)
        V['quality']    = tk.StringVar(value="高")
        V['use_reflector'] = tk.BooleanVar(value=True)
        V['use_dish']   = tk.BooleanVar(value=True)
        V['use_shield'] = tk.BooleanVar(value=True)
        V['custom_len'] = tk.DoubleVar(value=0.0)
        V['ignore_shield'] = tk.BooleanVar(value=False)

    def _setup_theme(self):
        """深色主题 + 彩色点缀"""
        style = ttk.Style()
        style.theme_use("clam")
        bg = '#1e1e1e'; bg2 = '#252526'; fg = '#d4d4d4'
        sel = '#094771'; border = '#3c3c3c'
        accent = '#00bc8c'; accent2 = '#e67e22'

        style.configure('.', background=bg, foreground=fg,
                        fieldbackground=bg2, troughcolor=bg2,
                        selectbackground=sel, selectforeground=fg)
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('TFrame', background=bg)
        style.configure('TLabelframe', background=bg, foreground=fg,
                        bordercolor=border, relief='solid')
        style.configure('TLabelframe.Label', background=bg, foreground=accent,
                        font=('Segoe UI', 9, 'bold'))
        style.configure('TButton', background=bg2, foreground=fg,
                        bordercolor=border, focuscolor='none')
        style.map('TButton', background=[('active', sel), ('pressed', '#0d3b66')])
        style.configure('Accent.TButton', background=accent, foreground='#ffffff',
                        bordercolor=accent, font=('Segoe UI', 9, 'bold'))
        style.map('Accent.TButton', background=[('active', '#00a67a'), ('pressed', '#008f6b')])
        style.configure('TEntry', fieldbackground=bg2, foreground=fg, bordercolor=border)
        style.configure('TCombobox', fieldbackground=bg2, foreground=fg, arrowcolor=fg)
        style.map('TCombobox', fieldbackground=[('readonly', bg2)])
        style.configure('TSeparator', background=border)
        style.configure('TNotebook', background=bg, bordercolor=border)
        style.configure('TNotebook.Tab', background=bg2, foreground=fg,
                        bordercolor=border, padding=[10, 4])
        style.map('TNotebook.Tab', background=[('selected', sel)],
                  foreground=[('selected', '#ffffff')])
        style.configure('TCheckbutton', background=bg, foreground=fg)
        style.map('TCheckbutton', background=[('active', bg2)])
        style.configure('TPanedwindow', background=bg)
        style.configure('Horizontal.TScrollbar', background=bg2, troughcolor=bg,
                        bordercolor=border, arrowcolor=fg)
        style.configure('Color.TLabelframe', background=bg, foreground=fg,
                        bordercolor=accent)
        style.configure('Color.TLabelframe.Label', background=bg, foreground=accent2,
                        font=('Segoe UI', 9, 'bold'))
        self.root.configure(bg=bg)

        plt_rc = {
            'figure.facecolor': '#1e1e1e', 'axes.facecolor': '#252526',
            'axes.edgecolor': '#555555', 'axes.labelcolor': '#d4d4d4',
            'text.color': '#d4d4d4', 'xtick.color': '#999999',
            'ytick.color': '#999999', 'grid.color': '#333333',
        }
        import matplotlib.pyplot as plt_main
        plt_main.rcParams.update(plt_rc)

    def _build_ui(self):
        mp = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        mp.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        cf = ttk.Frame(mp, width=320)
        mp.add(cf, weight=0)
        self.nb = ttk.Notebook(mp)
        mp.add(self.nb, weight=1)

        # --- 控制面板 ---
        r = 0
        header = ttk.Frame(cf)
        header.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(8, 4), padx=4)
        ttk.Label(header, text="天线仿真配置 v2.1",
                  font=("Segoe UI", 15, "bold"), foreground='#00bc8c').pack(side=tk.LEFT)
        gpu_color = '#4ec9b0' if C._GPU_AVAIL else '#888888'
        ttk.Label(header, text=f"GPU{' ON' if C._GPU_AVAIL else ' OFF'}",
                  font=("Consolas", 8), foreground=gpu_color).pack(side=tk.RIGHT, padx=(8, 0))
        r += 1

        self.freq_label = ttk.Label(
            cf, font=("Segoe UI", 9), foreground='#17a2b8')
        self.freq_label.grid(row=r, column=0, columnspan=2, sticky="w", pady=1, padx=4)
        self._update_freq_display()
        r += 1

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=3)
        r += 1

        ttk.Label(cf, text="频率 (MHz):", foreground='#e67e22').grid(
            row=r, column=0, sticky="w", pady=3, padx=4)
        freq_frame = ttk.Frame(cf)
        freq_frame.grid(row=r, column=1, sticky="ew", pady=3)
        fe = ttk.Entry(freq_frame, textvariable=self.freq_var,
                       width=10, font=("Consolas", 9))
        fe.pack(side=tk.LEFT)
        fe.bind("<Return>", lambda _: self._on_freq_change())
        fe.bind("<FocusOut>", lambda _: self._on_freq_change())
        ttk.Label(freq_frame, text="MHz",
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=3)
        r += 1

        ttk.Label(cf, text="偶极子:").grid(
            row=r, column=0, sticky="w", pady=3, padx=4)
        dt = ttk.Combobox(cf, textvariable=self.dip_type,
                          values=list(C.get_dipole_types().keys()),
                          state="readonly", width=18)
        dt.grid(row=r, column=1, sticky="ew", pady=3)
        dt.bind("<<ComboboxSelected>>", lambda _: self._update())
        r += 1

        ttk.Label(cf, text="自定义长 (mm):", foreground='#17a2b8').grid(
            row=r, column=0, sticky="w", pady=1, padx=4)
        cl_frame = ttk.Frame(cf)
        cl_frame.grid(row=r, column=1, sticky="ew", pady=1)
        cl_sv = tk.StringVar(value="0")
        cl_ent = ttk.Entry(cl_frame, textvariable=cl_sv,
                           width=10, font=("Consolas", 9))
        cl_ent.pack(side=tk.LEFT)
        ttk.Label(cl_frame, text="mm (0=预设)",
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=3)
        cl_ent.bind("<Return>", lambda _, v=self.var_d['custom_len'], sv=cl_sv:
                    self._entry_update('custom_len', v, sv))
        cl_ent.bind("<FocusOut>", lambda _, v=self.var_d['custom_len'], sv=cl_sv:
                    self._entry_update('custom_len', v, sv))
        r += 1

        ttk.Label(cf, text="渲染质量:").grid(
            row=r, column=0, sticky="w", pady=3, padx=4)
        qc = ttk.Combobox(cf, textvariable=self.var_d['quality'],
                          values=["低", "中", "高", "极致", "Ultra"],
                          state="readonly", width=10)
        qc.grid(row=r, column=1, sticky="w", pady=3)
        qc.bind("<<ComboboxSelected>>", lambda _: self._update())
        r += 1

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=5)
        r += 1

        comp_frame = ttk.LabelFrame(cf, text="部件开关", padding=4)
        comp_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=3, padx=4)
        for i, (key, label) in enumerate([
                ('use_reflector', '半圆柱反射板'),
                ('use_dish', '抛物面锅'),
                ('use_shield', '屏蔽桶')]):
            cb = ttk.Checkbutton(comp_frame, text=label,
                                 variable=self.var_d[key],
                                 command=self._update)
            cb.grid(row=0, column=i, sticky="w", padx=4)
        r += 1

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=3)
        r += 1

        self._build_param_section(cf, r)
        r += 20

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=5)
        r += 1

        ttk.Button(cf, text="🔄 更新仿真",
                   command=self._update, style='Accent.TButton').grid(
            row=r, column=0, columnspan=2, pady=5)
        r += 1

        bf = ttk.Frame(cf)
        bf.grid(row=r, column=0, columnspan=2, pady=3)
        r += 1
        ttk.Button(bf, text="📊 全部对比",
                   command=self._compare, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="📥 导入参数",
                   command=self._import_params, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="📤 导出参数",
                   command=self._export_params, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(bf, text="📄 导出图表",
                   command=self._export_all, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=5)
        r += 1

        self.st = ttk.Label(cf, text="就绪", font=("Segoe UI", 9))
        self.st.grid(row=r, column=0, columnspan=2, sticky="w", pady=2)
        r += 1

        mf = ttk.LabelFrame(cf, text="仿真指标", padding=6)
        mf.grid(row=r, column=0, columnspan=2, sticky="ew", pady=5)
        r += 1
        self.mt = tk.StringVar(value="等待计算...")
        ttk.Label(mf, textvariable=self.mt,
                  font=("Consolas", 10), justify=tk.LEFT).pack(anchor="w")

        cf.columnconfigure(1, weight=1)
        cf.rowconfigure(r, weight=1)
        self._make_tabs()

    def _update_freq_display(self):
        if self.freq_label:
            self.freq_label.config(
                text=f"频率: {C.FREQ/1e6:.1f} MHz | λ = {C.LAM*100:.1f} cm | "
                     f"β = {C.BETA:.2f} rad/m")

    def _on_freq_change(self):
        try:
            new_freq = float(self.freq_var.get()) * 1e6
            if new_freq < 1e6 or new_freq > 1e12:
                raise ValueError
            C.set_frequency(new_freq)
            self.freq_var.set(f"{C.FREQ/1e6:.1f}")
            self._update_freq_display()
            dt = self.var_d.get('_dip_combobox')
            if dt:
                dt['values'] = list(C.get_dipole_types().keys())
            self.var_d['ref_dist'].set(C.LAM / 4)
            self.var_d['cyl_R'].set(C.DEF_RR)
            self._update()
        except (ValueError, TypeError):
            self.freq_var.set(f"{C.FREQ/1e6:.1f}")

    def _build_param_section(self, parent, start_row):
        groups = [
            ("锅面参数", ['dish_a', 'dish_b', 'dish_f', 'offset_ang']),
            ("馈源偏移", ['feed_dx', 'feed_dz']),
            ("反射板", ['ref_dist', 'cyl_R', 'cyl_L', 'cyl_ang']),
            ("屏蔽桶", ['can_D', 'can_H', 'can_rot_x', 'can_rot_y', 'show_shield', 'ignore_shield']),
            ("显示", ['show_struct']),
        ]
        labels = {
            'dish_a': '锅长轴 a (m)', 'dish_b': '锅短轴 b (m)',
            'dish_f': '锅焦距 f (m)', 'offset_ang': '偏馈角 (°)',
            'feed_dx': '馈源偏移 X (m)', 'feed_dz': '馈源偏移 Z (m)',
            'ref_dist': '反射板距离 (m)', 'cyl_R': '半圆柱半径 R (m)',
            'cyl_L': '反射板长度 L (m)', 'cyl_ang': '反射板倾斜角 (°)',
            'can_D': '桶直径 (m)', 'can_H': '桶高度 (m)',
            'can_rot_x': '桶旋转 X (°)', 'can_rot_y': '桶旋转 Y (°)',
            'show_shield': '显示屏蔽桶', 'ignore_shield': '忽略内外位置 (桶开孔)',
            'show_struct': '显示天线结构',
        }
        self._param_ranges = {
            'dish_a': (1e-6, 1e3), 'dish_b': (1e-6, 1e3), 'dish_f': (1e-6, 1e3),
            'offset_ang': (0, 360),
            'feed_dx': (-1e3, 1e3), 'feed_dz': (-1e3, 1e3),
            'ref_dist': (1e-6, 1e3), 'cyl_R': (1e-6, 1e3),
            'cyl_L': (1e-6, 1e3), 'cyl_ang': (0, 360),
            'can_D': (1e-6, 1e3), 'can_H': (1e-6, 1e3),
            'can_rot_x': (-360, 360), 'can_rot_y': (-360, 360),
            'custom_len': (0, 10000),
        }
        _fmts = {k: '.3f' for k in self._param_ranges}
        _fmts.update({'offset_ang': '.1f', 'cyl_ang': '.1f',
                      'can_rot_x': '.1f', 'can_rot_y': '.1f'})
        bools = {'show_shield', 'show_struct', 'ignore_shield'}
        frame = ttk.Frame(parent)
        frame.grid(row=start_row, column=0, columnspan=2, sticky="nsew")
        self._param_widgets = {}
        for gname, keys in groups:
            if not keys:
                continue
            lf = ttk.LabelFrame(frame, text=gname, padding=3)
            lf.pack(fill=tk.X, pady=2)
            for k in keys:
                v = self.var_d[k]
                if k in bools:
                    cb = ttk.Checkbutton(lf, text=labels.get(k, k),
                                         variable=v, command=self._update)
                    cb.pack(anchor="w", pady=1)
                    self._param_widgets[k] = cb
                else:
                    f2 = ttk.Frame(lf)
                    f2.pack(fill=tk.X, pady=1)
                    ttk.Label(f2, text=labels.get(k, k),
                              width=16, anchor="w").pack(side=tk.LEFT)
                    sv = tk.StringVar(value=f"{v.get():{_fmts.get(k, '.3f')}}")
                    ent = ttk.Entry(f2, textvariable=sv,
                                    width=9, font=("Consolas", 9))
                    ent.pack(side=tk.RIGHT, padx=(3, 3))
                    ent.bind("<Return>",
                             lambda _, kk=k, vv=v, svv=sv: self._entry_update(kk, vv, svv))
                    ent.bind("<FocusOut>",
                             lambda _, kk=k, vv=v, svv=sv: self._entry_update(kk, vv, svv))
                    self._param_widgets[k + '_ent'] = sv

    def _entry_update(self, k, var, sv):
        try:
            val = float(sv.get())
            lo, hi = self._param_ranges.get(k, (0, 1e6))
            val = max(lo, min(hi, val))
            var.set(val)
            sv.set(f"{val:.3f}" if k not in ('offset_ang', 'cyl_ang', 'can_rot_x', 'can_rot_y')
                   else f"{val:.1f}")
            self._update()
        except ValueError:
            sv.set(f"{var.get():.3f}" if k not in ('offset_ang', 'cyl_ang', 'can_rot_x', 'can_rot_y')
                   else f"{var.get():.1f}")

    def _make_tabs(self):
        global _TAB_FRAMES
        self.t_pol = ttk.Frame(self.nb)
        self.nb.add(self.t_pol, text="极坐标方向图")
        self.t_3d = ttk.Frame(self.nb)
        self.nb.add(self.t_3d, text="3D 方向图")
        self.t_str = ttk.Frame(self.nb)
        self.nb.add(self.t_str, text="天线结构")
        self.t_ov = ttk.Frame(self.nb)
        self.nb.add(self.t_ov, text="叠加视图")
        self.t_ps = ttk.Frame(self.nb)
        self.nb.add(self.t_ps, text="点源增益分析")
        self.t_comp = ttk.Frame(self.nb)
        self.nb.add(self.t_comp, text="配置对比")
        self.t_prog = ttk.Frame(self.nb)
        self.nb.add(self.t_prog, text="逐级优化")
        _TAB_FRAMES = [self.t_pol, self.t_3d, self.t_str, self.t_ov, self.t_ps, self.t_prog]
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # ======================== 更新 / 计算 ========================

    def _update(self):
        self.st.config(text="计算中...")
        self.root.update_idletasks()
        if self._update_timer:
            self.root.after_cancel(self._update_timer)
        self._update_timer = self.root.after(30, self._do_update)

    def _do_update(self):
        self._update_timer = None
        self._compute_gen += 1
        gen = self._compute_gen
        if not self._compute_lock.acquire(blocking=False):
            self._pending_gen = gen
            return
        self._pending_gen = None
        threading.Thread(target=self._compute, args=(gen,), daemon=True).start()

    def _cancel_pending(self):
        if self._update_timer:
            self.root.after_cancel(self._update_timer)
            self._update_timer = None

    def _get_quality_res(self, q=None):
        if q is None:
            q = self.var_d['quality'].get()
        levels = {
            '低':    {'n_uv': 14, 'n_pol': 360, 'n_3d_t': 40,  'n_3d_p': 60,
                     'dish_nr': 14, 'dish_np': 20, 'cyl_nu': 18, 'cyl_nv': 14},
            '中':    {'n_uv': 18, 'n_pol': 720, 'n_3d_t': 60,  'n_3d_p': 90,
                     'dish_nr': 20, 'dish_np': 30, 'cyl_nu': 25, 'cyl_nv': 20},
            '高':    {'n_uv': 28, 'n_pol': 720, 'n_3d_t': 80,  'n_3d_p': 120,
                     'dish_nr': 30, 'dish_np': 48, 'cyl_nu': 36, 'cyl_nv': 28},
            '极致':  {'n_uv': 40, 'n_pol': 1080, 'n_3d_t': 120, 'n_3d_p': 180,
                     'dish_nr': 50, 'dish_np': 72, 'cyl_nu': 50, 'cyl_nv': 40},
            'Ultra': {'n_uv': 60, 'n_pol': 1440, 'n_3d_t': 160, 'n_3d_p': 240,
                     'dish_nr': 70, 'dish_np': 100, 'cyl_nu': 60, 'cyl_nv': 50},
        }
        return levels.get(q, levels['高'])

    def _get_dish_params_hash(self):
        V = self.var_d
        return (V['dish_a'].get(), V['dish_b'].get(), V['dish_f'].get(),
                V['offset_ang'].get(), V['feed_dx'].get(), V['feed_dz'].get())

    def _get_params(self):
        """从 var_d 提取所有参数快照"""
        return {k: v.get() for k, v in self.var_d.items()}

    def _get_dipole_length(self):
        """获取偶极子实际长度 (m)"""
        dip_types = C.get_dipole_types()
        length = dip_types[self.dip_type.get()]
        custom = self.var_d['custom_len'].get()
        return custom / 1000.0 if custom > 0 else length

    def _compute(self, gen):
        try:
            V = self.var_d
            qr = self._get_quality_res()
            length = self._get_dipole_length()
            a = V['dish_a'].get()
            b = V['dish_b'].get()
            f = V['dish_f'].get()
            D_eff = 2 * np.sqrt(a * b)
            d_ref = V['ref_dist'].get()
            feed_dx = V['feed_dx'].get()
            feed_dz = V['feed_dz'].get()
            shield_D = V['can_D'].get()
            shield_H = V['can_H'].get()
            oa = V['offset_ang'].get()

            if gen != self._compute_gen:
                return

            dish_hash = self._get_dish_params_hash()
            if dish_hash != self._geo_cache_key:
                self._geo_cache = {}
                self._geo_cache_key = dish_hash

            def ray_trace(t, p, L):
                use_s = V['use_shield'].get()
                ig = V.get('ignore_shield')
                if ig is not None and ig.get():
                    use_s = False
                return pattern_ray_trace(t, p, L, d_ref, a, b, f,
                                          feed_dx, feed_dz, shield_D,
                                          shield_H=shield_H,
                                          offset_ang=oa, n_uv=qr['n_uv'],
                                          use_reflector=V['use_reflector'].get(),
                                          use_dish=V['use_dish'].get(),
                                          use_shield=use_s)

            # 1D 切面
            t_fine = np.linspace(0, np.pi, qr['n_pol'])
            E_e = np.abs(ray_trace(t_fine, np.pi / 2, length))
            E_h = np.abs(ray_trace(t_fine, np.pi, length))

            if gen != self._compute_gen:
                return

            A_phys = np.pi * a * b
            D_dBi = 10 * np.log10(4 * np.pi * 0.55 * A_phys / C.LAM**2)

            P_e = E_e / np.max(E_e) if np.max(E_e) > 0 else E_e
            P_h = E_h / np.max(E_h) if np.max(E_h) > 0 else E_h
            hpbw_e = _hpbw_1d(t_fine, P_e**2)
            hpbw_h = _hpbw_1d(t_fine, P_h**2)

            bs_t = np.radians(60)
            Pf = np.abs(ray_trace(bs_t, np.pi, length))**2
            Pb = np.abs(ray_trace(np.pi - bs_t, 0.0, length))**2
            fb = 10 * np.log10(Pf / Pb) if Pb > 0 else 40.0

            metrics = {'D_dBi': D_dBi, 'HPBW_E': hpbw_e, 'HPBW_H': hpbw_h, 'FB_dB': fb}

            if gen != self._compute_gen:
                return

            txt = (f"指向性 D: {D_dBi:.2f} dBi (PO 物理光学)\n"
                   f"E面 HPBW: {hpbw_e:.1f}°\n"
                   f"H面 HPBW: {hpbw_h:.1f}°\n"
                   f"前后比 F/B: {fb:.1f} dB"
                   f"\n有效口径 D={D_eff:.3f}m"
                   f"\n反射板距离 d={d_ref*100:.1f}cm")
            self.root.after(0, lambda: self.mt.set(txt))
            self.root.after(0, lambda: self.st.config(text="计算方向图中..."))

            if gen != self._compute_gen:
                return

            # 2D 方向图 (复用于 3D + 叠加 + 点源)
            t_2d = np.linspace(0, np.pi, qr['n_3d_t'])
            p_2d = np.linspace(0, 2 * np.pi, qr['n_3d_p'])
            TH, PH = np.meshgrid(t_2d, p_2d, indexing='ij')
            E_2d = ray_trace(TH, PH, length)
            self._last_pat = (t_2d, p_2d, E_2d)

            if gen != self._compute_gen:
                return

            # ---- 4 级渐进结构分解 (点源分析) ----
            stage_configs = [
                ("① 裸偶极子",        False, False, False),
                ("② +反射板",     True,  False, False),
                ("③ +抛物面", True,  True,  False),
                ("④ +屏蔽桶",    True,  True,  True),
            ]
            def stage_rt(t, p, L, use_r, use_d, use_s):
                return pattern_ray_trace(t, p, L, d_ref, a, b, f,
                    feed_dx, feed_dz, shield_D, shield_H=shield_H,
                    offset_ang=oa, n_uv=qr['n_uv'],
                    use_reflector=use_r, use_dish=use_d, use_shield=use_s)
            stage_data = []
            s_bs_t = np.radians(60)
            for sname, use_r, use_d, use_s in stage_configs:
                Ee_s = np.abs(stage_rt(t_fine, np.pi/2, length, use_r, use_d, use_s))
                Eh_s = np.abs(stage_rt(t_fine, np.pi, length, use_r, use_d, use_s))
                Pe_s = Ee_s / np.max(Ee_s) if np.max(Ee_s) > 0 else Ee_s
                Ph_s = Eh_s / np.max(Eh_s) if np.max(Eh_s) > 0 else Eh_s
                he_s = _hpbw_1d(t_fine, Pe_s**2)
                hh_s = _hpbw_1d(t_fine, Ph_s**2)
                Pf_s = np.abs(stage_rt(s_bs_t, np.pi, length, use_r, use_d, use_s))**2
                Pb_s = np.abs(stage_rt(np.pi - s_bs_t, 0.0, length, use_r, use_d, use_s))**2
                fb_s = 10 * np.log10(Pf_s / Pb_s) if Pb_s > 0 else 40.0
                Pavg_s = (Ee_s**2 + Eh_s**2) / 2
                integ_s = Pavg_s * np.sin(t_fine)
                Prad_s = np.trapezoid(integ_s, t_fine) * 2 * np.pi
                Umax_s = np.max(Pavg_s)
                Dest_s = 4 * np.pi * Umax_s / Prad_s if Prad_s > 0 else 1
                stage_data.append(dict(name=sname, E_e=Ee_s, E_h=Eh_s,
                    D_dBi=10*np.log10(Dest_s), HPBW_E=he_s, HPBW_H=hh_s, FB_dB=fb_s))

            if gen != self._compute_gen:
                return

            # 存入缓存, 仅渲染当前活动的标签页
            params = self._get_params()
            self._cache = {
                't_fine': t_fine, 'E_e': E_e, 'E_h': E_h,
                'metrics': metrics, 'E_2d': E_2d,
                't_2d': t_2d, 'p_2d': p_2d,
                'length': length, 'qr': qr, 'params': params,
                'stage_data': stage_data,
            }
            self._rendered = set()
            self.root.after(0, lambda: self.st.config(text="✓ 完成"))
            self.root.after(0, lambda: self._render_active_tab())
        except Exception as e:
            self.root.after(0, lambda: self.st.config(text=f"❌ {e}"))
            import traceback
            traceback.print_exc()
        finally:
            self._compute_lock.release()
            if self._pending_gen is not None:
                self._pending_gen = None
                self.root.after(0, self._do_update)

    # ======================== 懒渲染 ========================

    def _on_tab_change(self, event=None):
        self._render_active_tab()

    def _render_active_tab(self):
        idx = self.nb.index(self.nb.select())
        if 0 <= idx < len(_TAB_FRAMES):
            self._render_tab(_TAB_FRAMES[idx])

    def _render_tab(self, frame):
        if not self._cache or frame in self._rendered:
            return
        if frame == self.t_str and not self.var_d['show_struct'].get():
            self._rendered.add(frame)
            return
        self._rendered.add(frame)
        c = self._cache
        try:
            if frame == self.t_pol:
                fig = render_polar(self.dip_type.get(), c['t_fine'], c['E_e'], c['E_h'], c['metrics'])
            elif frame == self.t_3d:
                fig = render_3d(self.dip_type.get(), c['t_2d'], c['p_2d'], c['E_2d'])
            elif frame == self.t_str:
                fig = render_structure_view(c['length'], c['params'], c['qr'])
            elif frame == self.t_ov:
                fig = render_overlay_view(c['t_2d'], c['p_2d'], c['E_2d'], c['params'], c['length'])
            elif frame == self.t_ps:
                fig = render_point_source(c['t_fine'], c['t_2d'], c['p_2d'], c['E_2d'],
                                          c['E_e'], c['E_h'], c['metrics'], c['params'],
                                          self.dip_type.get(), c['length'],
                                          stage_data=c.get('stage_data'))
            elif frame == self.t_prog:
                fig = render_progressive_opt(c['t_fine'], c.get('stage_data'))
            else:
                return
            embed_figure(fig, frame, self.root)
        except Exception as e:
            import traceback
            traceback.print_exc()

    # ======================== 配置对比 ========================

    def _compare(self):
        self.st.config(text="计算对比...")
        self.root.update_idletasks()
        threading.Thread(target=self._comp_run, daemon=True).start()

    def _comp_run(self):
        try:
            params = self._get_params()
            length = self._get_dipole_length()
            fig = render_comparison(params, self.dip_type.get(), length)
            embed_figure(fig, self.t_comp, self.root)
            self.root.after(0, lambda: self.nb.select(TAB_COMPARE))
            self.root.after(0, lambda: self.st.config(text="✓ 对比完成"))
        except Exception as e:
            self.root.after(0, lambda: self.st.config(text=f"❌ {e}"))
            import traceback
            traceback.print_exc()

    # ======================== 导入 / 导出 ========================

    def _export_params(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            title="导出参数")
        if not path:
            return
        try:
            data = {k: v.get() for k, v in self.var_d.items()}
            data['dip_type'] = self.dip_type.get()
            data['frequency_mhz'] = C.FREQ / 1e6
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.st.config(text=f"✓ 已导出: {os.path.basename(path)}")
        except Exception as e:
            self.st.config(text=f"❌ 导出失败: {e}")

    def _import_params(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            title="导入参数")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'frequency_mhz' in data:
                try:
                    C.set_frequency(float(data['frequency_mhz']) * 1e6)
                    self.freq_var.set(f"{C.FREQ/1e6:.1f}")
                    self._update_freq_display()
                except (ValueError, TypeError):
                    pass
            dip_types = C.get_dipole_types()
            if 'dip_type' in data and data['dip_type'] in dip_types:
                self.dip_type.set(data['dip_type'])
            _fmts = {k: '.3f' for k in self._param_ranges}
            _fmts.update({'offset_ang': '.1f', 'cyl_ang': '.1f'})
            for k, var in self.var_d.items():
                if k in data:
                    try:
                        val = float(data[k])
                        lo, hi = self._param_ranges.get(k, (0, 1e6))
                        val = max(lo, min(hi, val))
                        var.set(val)
                        ent_key = k + '_ent'
                        if ent_key in self._param_widgets:
                            self._param_widgets[ent_key].set(
                                f"{val:{_fmts.get(k, '.3f')}}")
                    except (ValueError, TypeError):
                        pass
            self.st.config(text=f"✓ 已导入: {os.path.basename(path)}")
            self._update()
        except Exception as e:
            self.st.config(text=f"❌ 导入失败: {e}")

    def _export_all(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("PDF 文档", "*.pdf"), ("所有文件", "*.*")],
            title="导出图表")
        if not path:
            return
        self.st.config(text="渲染中...")
        self.root.update_idletasks()
        try:
            params = self._get_params()
            length = self._get_dipole_length()
            qr = self._get_quality_res('Ultra')
            render_export_report(params, self.dip_type.get(), length, path, qr)
            self.st.config(text=f"✓ 已导出: {os.path.basename(path)}")
        except Exception as e:
            self.st.config(text=f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()
