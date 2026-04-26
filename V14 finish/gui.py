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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

import constants as C
from pattern import pattern_ray_trace
from rendering import render_structure, render_overlay
from math_utils import _hpbw_1d

warnings.filterwarnings('ignore')


class App:
    def __init__(self, root):
        self.root = root
        root.title("1420 MHz 天线仿真器 v2.1")
        root.geometry("1480x960")
        root.minsize(1200, 750)

        self.dip_type = tk.StringVar(value="半波长 (λ/2)")
        self.freq_var = tk.StringVar(value=f"{C.FREQ/1e6:.1f}")
        self.freq_display = None  # freq label widget

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

        plt.rcParams.update({
            'figure.facecolor': '#1e1e1e', 'axes.facecolor': '#252526',
            'axes.edgecolor': '#555555', 'axes.labelcolor': '#d4d4d4',
            'text.color': '#d4d4d4', 'xtick.color': '#999999',
            'ytick.color': '#999999', 'grid.color': '#333333',
        })

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
        # GPU 状态指示灯
        gpu_color = '#4ec9b0' if C._GPU_AVAIL else '#888888'
        gpu_text = f"GPU{' ON' if C._GPU_AVAIL else ' OFF'}"
        ttk.Label(header, text=gpu_text, font=("Consolas", 8),
                  foreground=gpu_color).pack(side=tk.RIGHT, padx=(8, 0))
        r += 1

        self.freq_label = ttk.Label(
            cf, font=("Segoe UI", 9), foreground='#17a2b8')
        self.freq_label.grid(row=r, column=0, columnspan=2, sticky="w", pady=1, padx=4)
        self._update_freq_display()
        r += 1

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=3)
        r += 1

        # 频率
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

        # 偶极子类型
        ttk.Label(cf, text="偶极子:").grid(
            row=r, column=0, sticky="w", pady=3, padx=4)
        dt = ttk.Combobox(cf, textvariable=self.dip_type,
                          values=list(C.get_dipole_types().keys()),
                          state="readonly", width=18)
        dt.grid(row=r, column=1, sticky="ew", pady=3)
        dt.bind("<<ComboboxSelected>>", lambda _: self._update())
        r += 1

        # 自定义偶极子长度
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

        # 渲染质量
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

        # 部件开关
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
            # Update dipole type combobox values
            dt = self.var_d.get('_dip_combobox')
            if dt:
                dt['values'] = list(C.get_dipole_types().keys())
            # Reset default params that depend on wavelength
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
            ("屏蔽桶", ['can_D', 'can_H', 'can_rot_x', 'can_rot_y', 'show_shield']),
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
            'show_shield': '显示屏蔽桶', 'show_struct': '显示天线结构',
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
        bools = {'show_shield', 'show_struct'}
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
        self.t_pol = ttk.Frame(self.nb)
        self.nb.add(self.t_pol, text="极坐标方向图")
        self.t_3d = ttk.Frame(self.nb)
        self.nb.add(self.t_3d, text="3D 方向图")
        self.t_str = ttk.Frame(self.nb)
        self.nb.add(self.t_str, text="天线结构")
        self.t_ov = ttk.Frame(self.nb)
        self.nb.add(self.t_ov, text="叠加视图")
        self.t_comp = ttk.Frame(self.nb)
        self.nb.add(self.t_comp, text="配置对比")

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

    def _compute(self, gen):
        try:
            V = self.var_d
            qr = self._get_quality_res()
            dip_types = C.get_dipole_types()
            length = dip_types[self.dip_type.get()]
            custom_len_mm = V['custom_len'].get()
            if custom_len_mm > 0:
                length = custom_len_mm / 1000.0
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
                return pattern_ray_trace(t, p, L, d_ref, a, b, f,
                                          feed_dx, feed_dz, shield_D,
                                          shield_H=shield_H,
                                          offset_ang=oa, n_uv=qr['n_uv'],
                                          use_reflector=V['use_reflector'].get(),
                                          use_dish=V['use_dish'].get(),
                                          use_shield=V['use_shield'].get())

            # 1D 切面
            t_fine = np.linspace(0, np.pi, qr['n_pol'])
            E_e = np.abs(ray_trace(t_fine, np.pi / 2, length))
            E_h = np.abs(ray_trace(t_fine, np.pi, length))

            if gen != self._compute_gen:
                return

            P_e = E_e / np.max(E_e) if np.max(E_e) > 0 else E_e
            P_h = E_h / np.max(E_h) if np.max(E_h) > 0 else E_h

            A_phys = np.pi * a * b
            eta_ap = 0.55
            D_dBi = 10 * np.log10(4 * np.pi * eta_ap * A_phys / C.LAM**2)

            hpbw_e = _hpbw_1d(t_fine, P_e**2)
            hpbw_h = _hpbw_1d(t_fine, P_h**2)

            bs_t = np.radians(60)
            Pf = np.abs(ray_trace(bs_t, np.pi, length))**2
            Pb = np.abs(ray_trace(np.pi - bs_t, 0.0, length))**2
            fb = 10 * np.log10(Pf / Pb) if Pb > 0 else 40.0

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

            # 立即渲染极坐标图 (快), 不等 2D
            self.root.after(0, lambda: self._plot_polar(t_fine, E_e, E_h,
                            {'D_dBi': D_dBi, 'HPBW_E': hpbw_e, 'HPBW_H': hpbw_h, 'FB_dB': fb}))

            if gen != self._compute_gen:
                return

            # 2D 方向图 (复用于 3D + 叠加)
            t_2d = np.linspace(0, np.pi, qr['n_3d_t'])
            p_2d_array = np.linspace(0, 2 * np.pi, qr['n_3d_p'])
            TH, PH = np.meshgrid(t_2d, p_2d_array, indexing='ij')
            E_2d = ray_trace(TH, PH, length)
            self._last_pat = (t_2d, p_2d_array, E_2d)

            if gen != self._compute_gen:
                return

            self.root.after(0, lambda: self.st.config(text="✓ 完成"))
            self.root.after(0, lambda: self._plot_3d(t_2d, p_2d_array, E_2d))
            self.root.after(0, lambda: self._plot_struct(length, qr))
            self.root.after(0, lambda: self._plot_overlay(length))
        except Exception as e:
            self.root.after(0, lambda: self.st.config(text=f"❌ {e}"))
            import traceback
            traceback.print_exc()
        finally:
            self._compute_lock.release()
            if self._pending_gen is not None:
                self._pending_gen = None
                self.root.after(0, self._do_update)

    def _plot_polar(self, th, E_e, E_h, m):
        fig = Figure(figsize=(7, 6), dpi=100)
        fig.suptitle(f"{self.dip_type.get()} — 全金属模型",
                     fontsize=12, fontweight="bold")
        ax0 = fig.add_subplot(1, 2, 1, projection='polar')
        Pe = np.abs(E_e) / np.max(np.abs(E_e)) if np.max(np.abs(E_e)) > 0 else np.abs(E_e)
        re = 10**(np.clip(10 * np.log10(np.clip(Pe, 1e-10, None)), -30, 0) / 20)
        ax0.plot(th, re, '#00bc8c', lw=1.5)
        ax0.set_title("E 面 (φ=90°)", va="bottom", fontsize=10)
        ax0.set_ylim(0, 1.05)
        ax0.grid(True, alpha=0.3)
        h = m.get('HPBW_E', 0)
        if h > 0:
            hh = np.radians(h / 2)
            ax0.plot([np.pi - hh, np.pi + hh], [0.707, 0.707], 'r--', lw=1.5, alpha=0.7)
            ax0.annotate(f"HPBW={h:.1f}°", xy=(np.pi, 0.85),
                         fontsize=9, color='red', ha='center')
        ax1 = fig.add_subplot(1, 2, 2, projection='polar')
        Ph = np.abs(E_h) / np.max(np.abs(E_h)) if np.max(np.abs(E_h)) > 0 else np.abs(E_h)
        rh = 10**(np.clip(10 * np.log10(np.clip(Ph, 1e-10, None)), -30, 0) / 20)
        ax1.plot(th, rh, '#e67e22', lw=1.5)
        ax1.set_title("H 面 (φ=180° 主瓣)", va="bottom", fontsize=10)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        h = m.get('HPBW_H', 0)
        if 10 < h < 170:
            hh = np.radians(h / 2)
            ax1.plot([np.pi - hh, np.pi + hh], [0.707, 0.707], 'r--', lw=1.5, alpha=0.7)
            ax1.annotate(f"HPBW={h:.1f}°", xy=(np.pi, 0.85),
                         fontsize=9, color='red', ha='center')
        fig.tight_layout()
        self._embed(fig, self.t_pol)

    def _plot_3d(self, t, p, E):
        fig = Figure(figsize=(8, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        TH, PH = np.meshgrid(t, p, indexing='ij')
        P = np.abs(E) / np.max(np.abs(E)) if np.max(np.abs(E)) > 0 else np.abs(E)
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
        ax.set_title(f"{self.dip_type.get()} — 全金属模型",
                     fontsize=11, fontweight="bold")
        for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
            a_.pane.fill = False
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        m = plt.cm.ScalarMappable(norm=n, cmap=plt.cm.viridis)
        m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1).set_label(
            '归一化方向图 (dB)', fontsize=9)
        fig.tight_layout()
        self._embed(fig, self.t_3d)

    def _plot_struct(self, length, qr=None):
        if not self.var_d['show_struct'].get():
            return
        fig = Figure(figsize=(8, 7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        V = self.var_d
        render_structure(ax, length,
                         V['dish_a'].get(), V['dish_b'].get(), V['dish_f'].get(),
                         V['offset_ang'].get(),
                         V['show_shield'].get(), V['can_D'].get(), V['can_H'].get(),
                         V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                         V['feed_dx'].get(), V['feed_dz'].get(),
                         V['can_rot_x'].get(), V['can_rot_y'].get(),
                         nr=qr['dish_nr'] if qr else 20,
                         np_=qr['dish_np'] if qr else 30,
                         cyl_nu=qr['cyl_nu'] if qr else 36,
                         cyl_nv=qr['cyl_nv'] if qr else 28,
                         show_dish=V['use_dish'].get(),
                         show_reflector=V['use_reflector'].get())
        parts = ['偶极子']
        if V['use_dish'].get():
            parts.append('抛物面锅')
        if V['use_reflector'].get():
            parts.append('半圆柱反射板')
        if V['show_shield'].get():
            parts.append('屏蔽桶')
        ax.set_title("天线结构: " + " + ".join(parts),
                     fontsize=11, fontweight="bold")
        h = [Line2D([0], [0], color='blue', lw=3, label='偶极子')]
        if V['use_reflector'].get():
            h.append(Line2D([0], [0], color='gold', lw=6, alpha=0.4, label='半圆柱反射板'))
        if V['use_dish'].get():
            h.append(Line2D([0], [0], color='silver', lw=1, alpha=0.4, label='抛物面锅'))
        if V['show_shield'].get():
            h.append(Line2D([0], [0], color='gray', lw=1, alpha=0.4, label='屏蔽桶'))
        if len(h) > 1:
            ax.legend(handles=h, loc='upper right', fontsize=7)
        fig.tight_layout()
        self._embed(fig, self.t_str)

    def _plot_overlay(self, length):
        if self._last_pat is None:
            return
        t, p_, Ep = self._last_pat
        fig = Figure(figsize=(9, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        V = self.var_d
        render_overlay(ax, t, p_, Ep,
                       V['dish_a'].get(), V['dish_b'].get(), V['dish_f'].get(),
                       V['offset_ang'].get(),
                       V['show_shield'].get(), V['can_D'].get(), V['can_H'].get(),
                       V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                       length, V['feed_dx'].get(), V['feed_dz'].get(),
                       V['can_rot_x'].get(), V['can_rot_y'].get(),
                       show_dish=V['use_dish'].get(),
                       show_reflector=V['use_reflector'].get())
        ax.set_title("结构 + 方向图叠加 (颜色=增益强度)",
                     fontsize=11, fontweight="bold")
        Pn = np.abs(Ep) / np.max(np.abs(Ep)) if np.max(np.abs(Ep)) > 0 else np.abs(Ep)
        n_ = plt.Normalize(0, 1)
        m_ = plt.cm.ScalarMappable(norm=n_, cmap=plt.cm.hot)
        m_.set_array([])
        fig.colorbar(m_, ax=ax, shrink=0.5, pad=0.1).set_label('归一化增益', fontsize=9)
        fig.tight_layout()
        self._embed(fig, self.t_ov)

    def _embed(self, fig, parent):
        def e():
            for w in parent.winfo_children():
                w.destroy()
            c = FigureCanvasTkAgg(fig, master=parent)
            c.draw()
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            t_ = NavigationToolbar2Tk(c, parent, pack_toolbar=False)
            t_.update()
            t_.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.after(0, e)

    def _compare(self):
        self.st.config(text="计算对比...")
        self.root.update_idletasks()
        threading.Thread(target=self._comp_run, daemon=True).start()

    def _comp_run(self):
        try:
            V = self.var_d
            dip_types = C.get_dipole_types()
            L = dip_types[self.dip_type.get()]
            custom_len_mm = V['custom_len'].get()
            if custom_len_mm > 0:
                L = custom_len_mm / 1000.0
            a = V['dish_a'].get()
            b = V['dish_b'].get()
            f = V['dish_f'].get()
            D0 = 2 * np.sqrt(a * b)
            d0 = V['ref_dist'].get()

            scales = [0.8, 1.0, 1.2]
            Ds = [D0 * s for s in scales]
            nm = [f"D={D:.2f}m" for D in Ds]
            feed_dx = V['feed_dx'].get()
            feed_dz = V['feed_dz'].get()
            shield_D = V['can_D'].get()
            shield_H = V['can_H'].get()
            oa = V['offset_ang'].get()

            ur = V['use_reflector'].get()
            ud = V['use_dish'].get()
            us = V['use_shield'].get()

            mm = []
            tf = np.linspace(0, np.pi, 360)
            for s in scales:
                aa = a * s
                bb = b * s
                rt = lambda t, p, L, dd=aa, ee=bb: pattern_ray_trace(
                    t, p, L, d0, dd, ee, f, feed_dx, feed_dz, shield_D,
                    shield_H=shield_H, offset_ang=oa, use_reflector=ur, use_dish=ud, use_shield=us)
                D_dBi = 10 * np.log10(4 * np.pi * 0.55 * np.pi * aa * bb / C.LAM**2)
                Ee = np.abs(rt(tf, np.pi / 2, L))
                Eh = np.abs(rt(tf, np.pi, L))
                pe = Ee / np.max(Ee) if np.max(Ee) > 0 else Ee
                ph = Eh / np.max(Eh) if np.max(Eh) > 0 else Eh
                he = _hpbw_1d(tf, pe**2)
                hh = _hpbw_1d(tf, ph**2)
                bs_t = np.radians(60)
                Pf = np.abs(rt(bs_t, np.pi, L))**2
                Pb = np.abs(rt(np.pi - bs_t, 0.0, L))**2
                fb = 10 * np.log10(Pf / Pb) if Pb > 0 else 40.0
                mm.append({'D_dBi': D_dBi, 'HPBW_E': he, 'HPBW_H': hh, 'FB_dB': fb})

            fig = Figure(figsize=(12, 13), dpi=100)
            gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
            fig.suptitle("1420 MHz PO 光线追踪 — 口径对比",
                         fontsize=14, fontweight="bold")
            cl = ['#00bc8c', '#3498db', '#e67e22']
            th = np.linspace(0, 2 * np.pi, 720)

            ax_e = fig.add_subplot(gs[0, 0], projection='polar')
            for i, (s, c) in enumerate(zip(scales, cl)):
                aa = a * s
                bb = b * s
                def _po(t, p, dd=aa, ee=bb): return pattern_ray_trace(
                    t, p, L, d0, dd, ee, f, feed_dx, feed_dz, shield_D,
                    shield_H=shield_H, offset_ang=oa, use_reflector=ur, use_dish=ud, use_shield=us)
                E = np.abs(_po(th, np.pi / 2))
                P = E / np.max(E) if np.max(E) > 0 else E
                r = 10**(np.clip(10 * np.log10(np.clip(P, 1e-10, None)), -30, 0) / 20)
                ax_e.plot(th, r, color=c, lw=1.2, label=nm[i])
            ax_e.set_title("E 面 (φ=90°)", fontsize=10)
            ax_e.set_ylim(0, 1.05)
            ax_e.legend(fontsize=7)

            ax_h = fig.add_subplot(gs[0, 1], projection='polar')
            for i, (s, c) in enumerate(zip(scales, cl)):
                aa = a * s
                bb = b * s
                def _po(t, p, dd=aa, ee=bb): return pattern_ray_trace(
                    t, p, L, d0, dd, ee, f, feed_dx, feed_dz, shield_D,
                    shield_H=shield_H, offset_ang=oa, use_reflector=ur, use_dish=ud, use_shield=us)
                E = np.abs(_po(th, 0))
                P = E / np.max(E) if np.max(E) > 0 else E
                r = 10**(np.clip(10 * np.log10(np.clip(P, 1e-10, None)), -30, 0) / 20)
                ax_h.plot(th, r, color=c, lw=1.2, label=nm[i])
            ax_h.set_title("H 面 (φ=0)", fontsize=10)
            ax_h.set_ylim(0, 1.05)
            ax_h.legend(fontsize=7)

            ax_t = fig.add_subplot(gs[0, 2])
            ax_t.axis('off')
            d_ = [[nm[i], f"{mm[i]['D_dBi']:.1f}", f"{mm[i]['HPBW_E']:.0f}°",
                   f"{mm[i]['HPBW_H']:.0f}°", f"{mm[i]['FB_dB']:.1f}"]
                  for i in range(3)]
            t_ = ax_t.table(cellText=d_,
                            colLabels=["口径", "D(dBi)", "HPBW_E", "HPBW_H", "F/B(dB)"],
                            loc='center', cellLoc='center', colWidths=[0.22, 0.16, 0.16, 0.16, 0.16])
            t_.auto_set_font_size(False)
            t_.set_fontsize(9)
            t_.scale(1, 1.8)
            ax_t.set_title("指标对比", fontsize=11, fontweight="bold", pad=20)

            ax_g = fig.add_subplot(gs[1, :])
            Dr = np.linspace(0.3, 1.0, 20)
            gs_plot = [10 * np.log10(4 * np.pi * 0.55 * np.pi * (d / 2)**2 / C.LAM**2)
                       for d in Dr]
            ax_g.plot(Dr, gs_plot, '#e74c3c', lw=2)
            ax_g.set_xlabel("有效口径 D (m)", fontsize=10)
            ax_g.set_ylabel("指向性 (dBi)", fontsize=10)
            ax_g.set_title("PO 模型增益 vs 口径 (理论极限)",
                           fontsize=11, fontweight="bold")
            ax_g.grid(True, alpha=0.3)

            # 频率-增益扫频图 (0.5 – 1700 MHz)
            ax_f = fig.add_subplot(gs[2, :])
            f_sweep = np.logspace(np.log10(0.5), np.log10(1700), 300)
            for i, s in enumerate(scales):
                aa_s = a * s
                bb_s = b * s
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
            self._embed(fig, self.t_comp)
            self.root.after(0, lambda: self.nb.select(4))
            self.root.after(0, lambda: self.st.config(text="✓ 对比完成"))
        except Exception as e:
            self.root.after(0, lambda: self.st.config(text=f"❌ {e}"))
            import traceback
            traceback.print_exc()

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
            V = self.var_d
            dip_types = C.get_dipole_types()
            L = dip_types[self.dip_type.get()]
            custom_len_mm = V['custom_len'].get()
            if custom_len_mm > 0:
                L = custom_len_mm / 1000.0
            a = V['dish_a'].get()
            b = V['dish_b'].get()
            f = V['dish_f'].get()
            d_ref = V['ref_dist'].get()
            feed_dx = V['feed_dx'].get()
            feed_dz = V['feed_dz'].get()
            shield_D = V['can_D'].get()
            shield_H = V['can_H'].get()
            oa = V['offset_ang'].get()
            rx = V['can_rot_x'].get()
            ry = V['can_rot_y'].get()

            def rt(t, p, L):
                return pattern_ray_trace(t, p, L, d_ref, a, b, f,
                                          feed_dx, feed_dz, shield_D,
                                          shield_H=shield_H,
                                          offset_ang=oa, n_uv=60,
                                          use_reflector=V['use_reflector'].get(),
                                          use_dish=V['use_dish'].get(),
                                          use_shield=V['use_shield'].get())

            qr = self._get_quality_res('Ultra')
            th = np.linspace(0, np.pi, qr['n_pol'])
            Ee = np.abs(rt(th, np.pi / 2, L))
            Eh = np.abs(rt(th, np.pi, L))
            Pe = Ee / np.max(Ee) if np.max(Ee) > 0 else Ee
            Ph = Eh / np.max(Eh) if np.max(Eh) > 0 else Eh

            A_phys = np.pi * a * b
            eta_ap = 0.55
            D_dBi = 10 * np.log10(4 * np.pi * eta_ap * A_phys / C.LAM**2)

            hpbw_e = _hpbw_1d(th, Pe**2)
            hpbw_h = _hpbw_1d(th, Ph**2)
            bs_t = np.radians(60)
            Pf = np.abs(rt(bs_t, np.pi, L))**2
            Pb = np.abs(rt(np.pi - bs_t, 0.0, L))**2
            fb = 10 * np.log10(Pf / Pb) if Pb > 0 else 40.0

            fig = Figure(figsize=(16, 24), dpi=120)
            gs = GridSpec(4, 2, figure=fig, height_ratios=[2, 2, 1.4, 1.4])

            # ---- (0,0) 极坐标方向图 ----
            a0 = fig.add_subplot(gs[0, 0], projection='polar')
            r_e = 10**(np.clip(10 * np.log10(np.clip(Pe, 1e-10, None)), -30, 0) / 20)
            r_h = 10**(np.clip(10 * np.log10(np.clip(Ph, 1e-10, None)), -30, 0) / 20)
            a0.plot(th, r_e, '#00bc8c', lw=1.2, label='E 面 (φ=90°)')
            a0.plot(th, r_h, '#e67e22', lw=1.2, label='H 面 (φ=180° 主瓣)')
            a0.set_ylim(0, 1.05)
            a0.grid(True, alpha=0.3)
            a0.set_title("极坐标方向图", fontsize=12, fontweight="bold", pad=12)
            a0.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.3, 1.0))
            if hpbw_e > 0:
                hh = np.radians(hpbw_e / 2)
                a0.plot([np.pi - hh, np.pi + hh], [0.707] * 2, 'b--', lw=1.5, alpha=0.5)
                a0.annotate(f"E HPBW={hpbw_e:.1f}°", xy=(np.pi, 0.8),
                           fontsize=8, color='blue', ha='center')
            if hpbw_h > 0:
                hh = np.radians(hpbw_h / 2)
                a0.plot([np.pi - hh, np.pi + hh], [0.707] * 2, 'r--', lw=1.5, alpha=0.5)
                a0.annotate(f"H HPBW={hpbw_h:.1f}°", xy=(np.pi, 0.7),
                           fontsize=8, color='red', ha='center')

            # ---- (0,1) 3D 方向图 ----
            a1 = fig.add_subplot(gs[0, 1], projection='3d')
            t3 = np.linspace(0, np.pi, qr['n_3d_t'])
            p3 = np.linspace(0, 2 * np.pi, qr['n_3d_p'])
            T3, P3 = np.meshgrid(t3, p3, indexing='ij')
            E3 = np.abs(rt(T3, P3, L))
            P3d = E3 / np.max(E3) if np.max(E3) > 0 else E3
            Pd = np.clip(10 * np.log10(np.clip(P3d, 1e-10, None)), -20, 0)
            n_ = plt.Normalize(-20, 0)
            c_ = plt.cm.viridis(n_(Pd))
            R3 = P3d
            X3 = R3 * np.sin(T3) * np.cos(P3)
            Y3 = R3 * np.sin(T3) * np.sin(P3)
            Z3 = R3 * np.cos(T3)
            a1.plot_surface(X3, Y3, Z3, facecolors=c_, rstride=2, cstride=2, alpha=0.9, lw=0)
            a1.set_xlim(-1.1, 1.1)
            a1.set_ylim(-1.1, 1.1)
            a1.set_zlim(-1.1, 1.1)
            a1.set_title("3D 方向图", fontsize=12, fontweight="bold")
            for ax_ in [a1.xaxis, a1.yaxis, a1.zaxis]:
                ax_.pane.fill = False
            a1.set_xticks([])
            a1.set_yticks([])
            a1.set_zticks([])

            # ---- (1,0) 天线结构 ----
            a2 = fig.add_subplot(gs[1, 0], projection='3d')
            render_structure(a2, L, a, b, f, oa,
                             V['show_shield'].get(), shield_D, shield_H,
                             V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                             feed_dx, feed_dz, rx, ry,
                             show_dish=V['use_dish'].get(),
                             show_reflector=V['use_reflector'].get())
            a2.set_title("天线结构", fontsize=12, fontweight="bold")

            # ---- (1,1) 叠加视图 ----
            a3 = fig.add_subplot(gs[1, 1], projection='3d')
            t_ = np.linspace(0, np.pi, qr['n_3d_t'])
            p_ = np.linspace(0, 2 * np.pi, qr['n_3d_p'])
            TT, PP = np.meshgrid(t_, p_, indexing='ij')
            Ep = rt(TT, PP, L)
            render_overlay(a3, t_, p_, Ep, a, b, f, oa,
                           V['show_shield'].get(), shield_D, shield_H,
                           V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                           L, feed_dx, feed_dz, rx, ry,
                           show_dish=V['use_dish'].get(),
                           show_reflector=V['use_reflector'].get())
            a3.set_title("结构 + 方向图叠加", fontsize=12, fontweight="bold")

            # ---- (2,0:2) 指标汇总 ----
            a4 = fig.add_subplot(gs[2, 0])
            a4.axis('off')
            txt = (f"频率: {C.FREQ/1e6:.1f} MHz  λ={C.LAM*100:.1f} cm\n"
                   f"偶极子: {self.dip_type.get()}  L={L*1000:.1f} mm\n"
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

            # 增益 vs 口径图
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

            dipole_label = self.dip_type.get()
            if custom_len_mm > 0:
                dipole_label += f" (自定义{L*1000:.1f}mm)"
            comp_parts = []
            if V['use_dish'].get(): comp_parts.append('抛物面')
            if V['use_reflector'].get(): comp_parts.append('反射板')
            if V['use_shield'].get(): comp_parts.append('屏蔽桶')
            comp_str = " + ".join(comp_parts) if comp_parts else "裸偶极子"

            # ---- (3,0:1) 频率-增益扫频图 ----
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
            fig.savefig(path, dpi=200, bbox_inches='tight')
            self.st.config(text=f"✓ 已导出: {os.path.basename(path)}")
        except Exception as e:
            self.st.config(text=f"❌ 导出失败: {e}")
            import traceback
            traceback.print_exc()
