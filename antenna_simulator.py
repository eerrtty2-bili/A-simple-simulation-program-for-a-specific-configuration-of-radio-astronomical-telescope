"""
1420 MHz 天线仿真器 v2.1
========================
半波长/全波长偶极子 + 半圆柱 180° 反射板 + 椭圆抛物面锅 + 屏蔽桶
结构与方向图 3D 叠加渲染
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import threading
import json
import warnings, os

warnings.filterwarnings('ignore')

# ======================== 常量 ========================
C = 299792458
FREQ = 1420e6
LAM = C / FREQ
BETA = 2 * np.pi / LAM

# ======================== Bessel J1 ========================
def bessel_j1(x):
    x = np.asarray(x, dtype=float)
    is_scalar = x.ndim == 0; x = np.atleast_1d(x)
    r = np.zeros_like(x)
    m = x <= 15
    if np.any(m):
        xs = x[m]; t = xs/2; acc = np.zeros_like(xs)
        for k in range(120):
            acc += t; t *= -xs*xs/(4*(k+1)*(k+2))
            if np.all(np.abs(t) < 1e-14): break
        r[m] = acc
    if np.any(~m):
        xl = x[~m]; r[~m] = np.sqrt(2/(np.pi*xl))*np.cos(xl-3*np.pi/4)
    return r.item() if is_scalar else r

# ======================== 方向图 ========================
def dipole_field(theta, length):
    kL2 = BETA*length/2
    with np.errstate(divide='ignore', invalid='ignore'):
        E = (np.cos(kL2*np.cos(theta))-np.cos(kL2))/np.sin(theta)
    if np.any(l := np.isnan(E)|np.isinf(E)): E = np.where(l, 0.0, E)
    return E

def _bcast(r, phi):
    if np.ndim(r)==0 and np.ndim(phi)>0: return np.broadcast_to(r, np.shape(phi))
    return r

def _angle_from_y(t, p):
    """球坐标(θ,φ)中与Y轴的夹角"""
    return np.arccos(np.clip(np.sin(t)*np.sin(p), -1, 1))

def pattern_bare(t, p, L):
    theta_axis = _angle_from_y(t, p)  # Y轴偶极子
    return _bcast(dipole_field(theta_axis, L), p)

def pattern_reflector(t, p, L, d_ref=None):
    if d_ref is None: d_ref = LAM/4
    theta_axis = _angle_from_y(t, p)  # Y轴偶极子
    E = dipole_field(theta_axis, L) * 2j*np.sin(BETA*d_ref*np.sin(t)*np.cos(p))
    x = np.sin(t)*np.cos(p)
    if np.ndim(x)==0:
        if x<0: E *= 0.1
    else:
        xa=np.asarray(x); ea=np.asarray(E,dtype=complex)
        w=(1+np.tanh(xa/0.12))/2; ea=ea*(w+0.02*(1-w)); E=ea
    return E

def pattern_dish(t, p, L, D=None, fD=None):
    if D is None: D=1.0
    if fD is None: fD=0.4
    u = BETA*D/2*np.sin(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        E = 2*bessel_j1(u)/u
        if np.ndim(E)==0:
            if np.isnan(E)|np.isinf(E): E=0.0
            elif np.abs(u)<1e-10: E=1.0
        else:
            E=np.asarray(E,dtype=float); E[np.isnan(E)|np.isinf(E)]=0.0
            E=np.where(np.abs(u)<1e-10,1.0,E)
    return _bcast(E*np.sqrt(0.55), p)

def pattern_combined(t, p, L, d_ref, D, fD):
    """统一模型 (偶极子 + 半圆柱反射板 + 抛物面锅 + 屏蔽桶)

    所有金属部件对方向图的贡献:
    - 偶极子沿 Y 轴放置 (振子方向∥锅面∥反射板轴)
    - 反射板 (镜像法) 增强前向、抑制后向
    - 抛物面锅 (孔径场) 形成主波束
    - 屏蔽桶 (角域滤波) 压制宽角旁瓣
    """
    # 1. 馈源方向图 = 偶极子 × 反射板 (镜像法)
    E_feed = pattern_reflector(t, p, L, d_ref)

    # 2. 抛物面锅孔径方向图 (Airy 盘), 波束沿 Z 轴 (锅面法向)
    u = BETA*D/2*np.sin(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        E_dish = 2*bessel_j1(u)/u
        if np.ndim(E_dish)==0:
            if np.isnan(E_dish)|np.isinf(E_dish): E_dish=0.0
            elif np.abs(u)<1e-10: E_dish=1.0
        else:
            E_dish=np.asarray(E_dish,dtype=float)
            E_dish[np.isnan(E_dish)|np.isinf(E_dish)]=0.0
            E_dish=np.where(np.abs(u)<1e-10,1.0,E_dish)
    E_dish *= np.sqrt(0.55)  # 口径效率

    # 3. 屏蔽桶角域滤波: 超出桶边缘 (相对锅面法向 Z) 的辐射被压制
    f_dish = D * fD  # 焦距 f = D * (f/D)
    rim_angle = np.arctan2(D/2, f_dish)
    alpha = t  # 从 Z 轴 (锅面法向) 的角度
    if np.ndim(alpha)>0:
        sf = 1.0/(1.0+np.exp((alpha-rim_angle)/0.04))
    else:
        sf = 1.0 if alpha<rim_angle else 0.01

    # 4. 合成: 馈源照亮锅面, 桶压制旁瓣
    E = E_feed * E_dish * sf
    P = np.abs(E); Pm = np.max(P)
    if Pm>0: E = E/np.sqrt(Pm)
    return E

def _offset_dish_geometry(a, b, f, offset_deg, feed_dx, feed_dz, n_uv):
    """生成偏馈抛物面锅几何数据

    偏馈抛物面: 焦点在馈源位, 光轴与 馈源→原点 连线成 offset_deg 夹角.
    锅面为母抛物面的一椭圆截面, 半轴 a, b 投影于光轴垂直平面.

    Returns
    -------
    dict — 含 X,Y,Z (3D坐标), nx,ny,nz (法向), dS (面积元),
            mask (椭圆口径), F (馈源位), û (光轴方向),
            ê1, ê2 (口径平面基矢), z_off (Z偏移)
    """
    α = np.radians(offset_deg)
    F = np.array([feed_dx, 0.0, f + feed_dz], dtype=float)

    # 光轴方向 û: 与 F (锅底→馈源) 成 α 角, 在 XZ 平面
    v = F / np.linalg.norm(F) if np.linalg.norm(F) > 1e-12 else np.array([0, 0, 1])
    β = np.arctan2(v[2], v[0])
    βp = β + α
    ux = np.cos(βp); uz = np.sin(βp)
    if uz < 0:  # 确保光轴指向上方 (+Z)
        βp = β - α
        ux = np.cos(βp); uz = np.sin(βp)
    û = np.array([ux, 0.0, uz])
    û /= np.linalg.norm(û)

    # 口径平面基矢
    ê1 = np.array([û[2], 0.0, -û[0]])  # XZ 平面内 ⟂ û
    ê1 /= np.linalg.norm(ê1)
    ê2 = np.array([0.0, 1.0, 0.0])

    # 口径采样
    u1 = np.linspace(-1, 1, n_uv); v1 = np.linspace(-1, 1, n_uv)
    U, V = np.meshgrid(u1, v1)
    mask = U**2 + V**2 <= 1.0

    # 口径坐标 (米)
    u_apt = U * a
    v_apt = V * b

    # 抛物面: P = F + u·ê1 + v·ê2 + ((u²+v²)/(4f) - f)·û
    uv2 = u_apt**2 + v_apt**2
    w = uv2 / (4 * f) - f

    X = F[0] + u_apt * ê1[0] + w * û[0]
    Y = F[1] + v_apt * ê2[1] + w * û[1]
    Z = F[2] + u_apt * ê1[2] + w * û[2]

    # 整体下移使最低点 Z=0
    z_off = np.min(Z[mask]) if np.any(mask) else 0.0
    Z -= z_off

    # 法向: n = û - u/(2f)·ê1 - v/(2f)·ê2
    nx = û[0] - (u_apt/(2*f)) * ê1[0]
    ny = û[1] - (v_apt/(2*f)) * ê2[1]
    nz = û[2] - (u_apt/(2*f)) * ê1[2]
    n_len = np.sqrt(nx**2 + ny**2 + nz**2)

    du = 2 * a / max(n_uv - 1, 1)
    dv = 2 * b / max(n_uv - 1, 1)
    dS = n_len * du * dv

    return dict(X=X, Y=Y, Z=Z,
                nx=nx/n_len, ny=ny/n_len, nz=nz/n_len,
                dS=dS, mask=mask,
                F=F, û=û, ê1=ê1, ê2=ê2, z_off=z_off)


def pattern_ray_trace(t, p, L, d_ref, dish_a, dish_b, dish_f,
                       feed_dx, feed_dz, shield_D, offset_ang=60, n_uv=18):
    """
    物理光学(PO)光线追踪综合模型 (偏馈抛物面)

    从偶极子出发 -> 反射板(镜像法) -> 偏馈抛物面锅(PO面积分) -> 屏蔽桶(角域滤波)
    最终合成远场方向图. 锅面为偏馈截面, 光轴与馈源→最低点连线成 offset_ang 夹角.

    Parameters
    ----------
    t, p : ndarray — 远场球坐标 θ, φ (可 1D 或 2D)
    L : float — 偶极子长度
    d_ref : float — 反射板距离
    dish_a, dish_b : float — 椭圆锅半轴 (m)
    dish_f : float — 母抛物面焦距 (m)
    feed_dx, feed_dz : float — 馈源偏移 (m)
    shield_D : float — 屏蔽桶直径 (m)
    offset_ang : float — 偏馈角, 光轴与馈源→原点夹角 (°, 默认 60)
    n_uv : int — 锅面采样每轴点数

    Returns
    -------
    E : ndarray — 复数远场方向图
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    _scalar = t.ndim == 0 and p.ndim == 0
    if _scalar:
        t = np.array([t]); p = np.array([p])
    _1d = t.ndim == 1
    if _1d:
        t, p = np.meshgrid(t, np.atleast_1d(p), indexing='ij')

    # === 偏馈锅面几何 ===
    G = _offset_dish_geometry(dish_a, dish_b, dish_f, offset_ang,
                               feed_dx, feed_dz, n_uv)
    X, Y, Z = G['X'], G['Y'], G['Z']
    F = G['F']
    mask = G['mask']

    # === 馈源→锅面各点 ===
    dX = X - F[0]
    dY = Y - F[1]
    dZ = Z - F[2]
    R_feed = np.sqrt(dX**2 + dY**2 + dZ**2)

    # 偶极子方向图 (Y轴)
    cos_theta_y = dY / R_feed
    theta_y = np.arccos(np.clip(cos_theta_y, -1, 1))
    E_dip = dipole_field(theta_y, L)

    # 反射板 (镜像法: 法线沿 X 轴方向)
    cos_phi_x = dX / R_feed
    E_ref = E_dip * (2j * np.sin(BETA * d_ref * cos_phi_x))

    # 半圆柱阴影
    w = 1.0 / (1.0 + np.exp(-cos_phi_x / 0.05))
    E_illum = E_ref * (w + 0.02 * (1 - w))

    # 空间衰减 + 入射相位
    E_illum = E_illum / R_feed
    phase_inc = np.exp(-1j * BETA * R_feed)

    # === PO 电流 ===
    nx, ny, nz = G['nx'], G['ny'], G['nz']
    cos_inc = np.abs(dX*nx + dY*ny + dZ*nz) / R_feed
    J = 2.0 * E_illum * phase_inc * cos_inc
    J_all = J * G['dS'] * mask

    # === 屏蔽桶: 偏馈锅面由 mask (U²+V²≤1) 控制边界 ===
    # 桶边缘角用于溢失辐射截止
    rim_angle = np.arctan2(shield_D/2, dish_f)

    # === 远场积分 (向量化 4D 广播) ===
    n_uv2 = n_uv
    J_4d = J_all.reshape(1, 1, n_uv2, n_uv2).astype(complex)
    X_4d = X.reshape(1, 1, n_uv2, n_uv2)
    Y_4d = Y.reshape(1, 1, n_uv2, n_uv2)
    Z_4d = Z.reshape(1, 1, n_uv2, n_uv2)

    st = np.sin(t)[:, :, None, None]
    ct = np.cos(t)[:, :, None, None]
    sp = np.sin(p)[:, :, None, None]
    cp = np.cos(p)[:, :, None, None]
    ux = st * cp
    uy = st * sp
    uz = ct

    phase_far = np.exp(1j * BETA * (ux*X_4d + uy*Y_4d + uz*Z_4d))
    E_dish = np.sum(J_4d * phase_far, axis=(2, 3))

    # === 溢失辐射 (spillover) ===
    E_spill = pattern_reflector(t, p, L, d_ref)
    sf = 1.0 / (1.0 + np.exp((t - rim_angle) / 0.04))
    E_spill *= sf * 0.15

    E_total = E_dish + E_spill

    Pm = np.max(np.abs(E_total))
    if Pm > 0:
        E_total /= np.sqrt(Pm)

    if _scalar:
        return E_total[0, 0]
    if _1d:
        return E_total[:, 0] if E_total.shape[1] == 1 else E_total
    return E_total


DIPOLE_TYPES = {"半波长 (λ/2)": LAM/2, "全波长 (λ)": LAM}

# ======================== 默认结构参数 ========================
DEF_A = 0.250   # 椭圆锅长轴半长 (m)
DEF_B = 0.235   # 椭圆锅短轴半长 (m)
DEF_F = 0.32    # 焦距 (m)
DEF_ANG = 60    # 偏馈角 (度)
DEF_RR = LAM/4  # 半圆柱反射板半径
DEF_RL = 0.15   # 半圆柱反射板长度 (m)

# ======================== 3D 几何 ========================
def _offset_axis(f, offset_ang, feed_dx, feed_dz):
    """计算偏馈抛物面光轴 û, 口径基矢 ê1,ê2 及 Z 偏移"""
    α = np.radians(offset_ang)
    F = np.array([feed_dx, 0.0, f + feed_dz], dtype=float)
    # 光轴方向: 从锅底指向馈源方向, 向 +X 侧旋转 offset_ang
    v = F / np.linalg.norm(F) if np.linalg.norm(F) > 1e-12 else np.array([0, 0, 1])
    β = np.arctan2(v[2], v[0])
    βp = β + α
    ux = np.cos(βp); uz = np.sin(βp)
    if uz < 0:  # 确保光轴指向上方 (+Z)
        βp = β - α
        ux = np.cos(βp); uz = np.sin(βp)
    û = np.array([ux, 0.0, uz])
    û /= np.linalg.norm(û)
    ê1 = np.array([û[2], 0.0, -û[0]])
    ê1 /= np.linalg.norm(ê1)
    ê2 = np.array([0.0, 1.0, 0.0])
    return û, ê1, ê2, F

def dish_mesh(a, b, f, nr=20, np_=30, offset_ang=60, feed_dx=0, feed_dz=0):
    û, ê1, ê2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    u = np.linspace(0, 2*np.pi, np_); v = np.linspace(1e-6, 1, nr)
    U, V = np.meshgrid(u, v)
    u_apt = V*a*np.cos(U); v_apt = V*b*np.sin(U)
    uv2 = u_apt**2 + v_apt**2; w = uv2/(4*f) - f
    X = F[0] + u_apt*ê1[0] + w*û[0]
    Y = F[1] + v_apt*ê2[1] + w*û[1]
    Z = F[2] + u_apt*ê1[2] + w*û[2]
    Z -= np.min(Z)
    return X, Y, Z

def dish_edge(a, b, f, n=60, offset_ang=60, feed_dx=0, feed_dz=0):
    û, ê1, ê2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    u = np.linspace(0, 2*np.pi, n)
    u_apt = a*np.cos(u); v_apt = b*np.sin(u)
    uv2 = u_apt**2 + v_apt**2; w = uv2/(4*f) - f
    X = F[0] + u_apt*ê1[0] + w*û[0]
    Y = F[1] + v_apt*ê2[1] + w*û[1]
    Z = F[2] + u_apt*ê1[2] + w*û[2]
    Z -= np.min(Z)
    return X, Y, Z

def shield_mesh(a, b, f, offset_ang, feed_dx, feed_dz, H, D=None,
                rot_x=0, rot_y=0, nh=15, np_=30):
    """屏蔽桶: 正圆柱体, 轴线沿入射光轴 û (截面为正圆)"""
    û, ê1, ê2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    u = np.linspace(0, 2*np.pi, np_); v = np.linspace(0, H, nh)
    U, V = np.meshgrid(u, v)
    R = D/2 if D is not None else np.sqrt(a*b)
    cu = R * np.cos(U); cv = R * np.sin(U)
    uv2 = cu**2 + cv**2; w0 = uv2/(4*f) - f
    X0 = F[0] + cu*ê1[0] + cv*ê2[0] + w0*û[0]
    Y0 = F[1] + cu*ê1[1] + cv*ê2[1] + w0*û[1]
    Z0 = F[2] + cu*ê1[2] + cv*ê2[2] + w0*û[2]
    # 沿 +û 方向拉伸
    X = X0 + V*û[0]; Y = Y0 + V*û[1]; Z = Z0 + V*û[2]
    # 三维旋转: 绕锅中心旋转桶指向
    if rot_x != 0 or rot_y != 0:
        cx, cy, cz = F - f*û  # dish center
        Xt = X - cx; Yt = Y - cy; Zt = Z - cz
        if rot_x != 0:
            a_ = np.radians(rot_x); c, s = np.cos(a_), np.sin(a_)
            Yt, Zt = Yt*c - Zt*s, Yt*s + Zt*c
        if rot_y != 0:
            a_ = np.radians(rot_y); c, s = np.cos(a_), np.sin(a_)
            Xt, Zt = Xt*c + Zt*s, -Xt*s + Zt*c
        X = Xt + cx; Y = Yt + cy; Z = Zt + cz
    Z -= np.min(Z)
    return X, Y, Z

def dipole_line(length, center=(0,0,0)):
    cx,cy,cz = center; h=length/2
    return [(cx,cy-h,cz),(cx,cy-h,cz+0.002),(cx,cy+h,cz+0.002),(cx,cy+h,cz)]

def semi_cylinder_mesh(R, length, angle_deg, n_u=25, n_v=20):
    """
    180° 半圆柱反射板
    圆柱轴沿 Y, 开口朝向由 angle_deg 决定
    angle_deg: 半圆柱顶点→偶极子连线与水平面夹角
    """
    # 局域坐标系: 顶点在 +Z 方向, 开口在 -Z 方向
    u = np.linspace(-np.pi/2, np.pi/2, n_u)  # 180°
    v = np.linspace(-length/2, length/2, n_v)
    U, V = np.meshgrid(u, v)

    # 先建立标准半圆柱 (顶点在 +Z 方向)
    Xl = R * np.sin(U)   # 开口方向 (±X)
    Yl = V               # 沿圆柱轴
    Zl = R * np.cos(U)   # 顶点在 Zl = R

    # 旋转: 使顶点-中心连线与水平面成 angle_deg
    ang = np.radians(angle_deg)
    # 绕 Y 轴旋转, 使 Z 轴方向顶点转到 (sin(ang), 0, cos(ang)) 方向
    # 开口方向从 -Z 转到 -(cos(ang), 0, -sin(ang))
    Xg = Xl * np.cos(ang) + Zl * np.sin(ang)
    Zg = -Xl * np.sin(ang) + Zl * np.cos(ang)
    Yg = Yl

    return Xg, Yg, Zg

# ======================== 结构渲染 ========================
def render_structure(ax, dip_len, dish_a, dish_b, dish_f,
                     offset_ang, show_shield, shield_D, shield_H,
                     cyl_R, cyl_len, cyl_ang, feed_dx=0, feed_dz=0,
                     can_rot_x=0, can_rot_y=0):
    f = dish_f
    feed = np.array([feed_dx, 0, f + feed_dz])
    û, ê1, ê2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    # 1. 锅
    Xd, Yd, Zd = dish_mesh(dish_a, dish_b, f, offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
    ax.plot_wireframe(Xd, Yd, Zd, rstride=2, cstride=3,
                      color='gray', alpha=0.2, linewidth=0.4)
    ax.plot_surface(Xd, Yd, Zd, color='silver', alpha=0.12,
                    rstride=2, cstride=2, edgecolor='none')
    xe, ye, ze = dish_edge(dish_a, dish_b, f, offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
    ax.plot(xe, ye, ze, 'k-', lw=1.3, alpha=0.6)

    # 2. 屏蔽桶
    if show_shield:
        Xs, Ys, Zs = shield_mesh(dish_a, dish_b, f, offset_ang, feed_dx, feed_dz, shield_H, D=shield_D,
                                  rot_x=can_rot_x, rot_y=can_rot_y)
        ax.plot_surface(Xs, Ys, Zs, color='darkgray', alpha=0.06,
                        rstride=2, cstride=2, edgecolor='gray',
                        linewidth=0.2, linestyle='--')
        ax.plot(Xs[0,:], Ys[0,:], Zs[0,:], color='gray', lw=0.8, alpha=0.4)

    # 4. 偏馈角标注: 光轴方向 û
    r_ang = 0.08
    u_len = 0.35 * dish_a
    ax.plot([feed[0], feed[0] + u_len*û[0]], [0, 0],
            [feed[2], feed[2] + u_len*û[2]], 'r--', lw=0.8, alpha=0.5)
    ax.plot([feed[0], 0], [0, 0], [feed[2], 0], 'r:', lw=0.5, alpha=0.3)
    v_to_origin = -F / np.linalg.norm(F)
    half_dir = v_to_origin + û
    half_dir = half_dir / np.linalg.norm(half_dir)
    lx = feed[0] + r_ang * half_dir[0]
    lz = feed[2] + r_ang * half_dir[2]
    ax.text(lx, 0, lz, f"{offset_ang}°", color='red', fontsize=8, fontweight='bold')

    # 5. 偶极子 (沿 Y 轴)
    dip = dipole_line(dip_len, center=feed)
    dx = [p[0] for p in dip]
    dy = [p[1] for p in dip]
    dz = [p[2] for p in dip]
    ax.plot(dx, dy, dz, 'b-', lw=4, alpha=0.9, label='偶极子')
    ax.scatter(dx, dy, dz, color='blue', s=20, alpha=0.8, zorder=5)
    ax.text(feed[0]+0.08, 0, feed[2]+0.005, "偶极子∥锅面", fontsize=7, color='blue', ha='center')

    # 6. 半圆柱 180° 反射板
    Xc, Yc, Zc = semi_cylinder_mesh(cyl_R, cyl_len, cyl_ang)
    ang_r = np.radians(cyl_ang)
    apex_vec = np.array([np.sin(ang_r), 0, np.cos(ang_r)])
    offset = feed + cyl_R * apex_vec
    Xc_t = Xc + feed[0]; Yc_t = Yc + feed[1]; Zc_t = Zc + feed[2]
    ax.plot_surface(Xc_t, Yc_t, Zc_t, color='gold', alpha=0.25,
                    rstride=2, cstride=2, edgecolor='orange', linewidth=0.3)
    ax.scatter(*offset, color='orange', s=25, marker='^', zorder=5)
    ax.plot([offset[0], feed[0]], [offset[1], feed[1]],
            [offset[2], feed[2]], 'orange', lw=0.5, alpha=0.4)

    # 7. 馈源标记
    ax.scatter(*feed, color='red', s=30, marker='x', zorder=6)
    ax.text(feed[0], 0, feed[2]+0.01, "馈源", fontsize=7, color='red', ha='center')

    # 8. 尺寸标注
    ax.text(dish_a+0.015, 0, 0, f"a={dish_a*100:.0f}cm", fontsize=7, color='gray')
    ax.text(0, dish_b+0.015, 0, f"b={dish_b*100:.0f}cm", fontsize=7, color='gray')

    # 9. 坐标轴
    md = max(dish_a, dish_b, abs(feed[2]), abs(feed[0]))*1.4
    z_min = -md*0.2 if show_shield else -0.02
    ax.set_xlim(-md, md); ax.set_ylim(-md, md); ax.set_zlim(z_min, md*1.1)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect([1, 1, 0.9])
    for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a_.pane.fill = False; a_.pane.set_edgecolor('lightgray')

def render_overlay(ax, theta, phi, E_pat, dish_a, dish_b, dish_f,
                   offset_ang, show_shield, shield_D, shield_H,
                   cyl_R, cyl_len, cyl_ang, dip_len, feed_dx=0, feed_dz=0,
                   can_rot_x=0, can_rot_y=0):
    """结构与方向图叠加"""
    f = dish_f
    feed = np.array([feed_dx, 0, f + feed_dz])
    Xd, Yd, Zd = dish_mesh(dish_a, dish_b, f, offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
    ax.plot_wireframe(Xd, Yd, Zd, rstride=3, cstride=4,
                      color='gray', alpha=0.12, linewidth=0.3)
    xe, ye, ze = dish_edge(dish_a, dish_b, f, offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
    ax.plot(xe, ye, ze, 'k-', lw=1, alpha=0.4)
    ax.scatter(*feed, color='red', s=20, marker='x', zorder=5)

    # 半圆柱 (简化)
    Xc, Yc, Zc = semi_cylinder_mesh(cyl_R, cyl_len, cyl_ang)
    ax.plot_surface(Xc+feed[0], Yc+feed[1], Zc+feed[2],
                    color='gold', alpha=0.12,
                    rstride=2, cstride=2, edgecolor='orange', linewidth=0.2)

    # 屏蔽桶 (叠加视图)
    if show_shield:
        Xs, Ys, Zs = shield_mesh(dish_a, dish_b, f, offset_ang, feed_dx, feed_dz, shield_H, D=shield_D,
                                  rot_x=can_rot_x, rot_y=can_rot_y)
        ax.plot_surface(Xs, Ys, Zs, color='darkgray', alpha=0.06,
                        rstride=2, cstride=2, edgecolor='gray',
                        linewidth=0.2, linestyle='--')
        ax.plot(Xs[0,:], Ys[0,:], Zs[0,:], color='gray', lw=0.8, alpha=0.4)

    # 方向图叠加
    P = np.abs(E_pat); Pn = P/np.max(P) if np.max(P)>0 else P
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    scale = 0.4
    R = Pn * scale
    Xg = R * np.sin(TH) * np.cos(PH) + feed[0]
    Yg = R * np.sin(TH) * np.sin(PH)
    Zg = R * np.cos(TH) + feed[2]

    # 只显示高于 -15dB 的区域
    P_dB = 10*np.log10(np.clip(Pn, 1e-10, None))
    mask = P_dB >= -15

    if np.any(mask):
        Xm = np.where(mask, Xg, np.nan)
        Ym = np.where(mask, Yg, np.nan)
        Zm = np.where(mask, Zg, np.nan)
        colors = plt.cm.hot(Pn)
        ax.plot_surface(Xm, Ym, Zm, facecolors=colors,
                        rstride=2, cstride=2, alpha=0.5,
                        linewidth=0, antialiased=True)

    md = max(dish_a, dish_b, abs(feed[2]), abs(feed[0]), scale)*1.3
    z_min = -md*0.2 if show_shield else -0.02
    ax.set_xlim(-md, md); ax.set_ylim(-md, md); ax.set_zlim(z_min, md*1.1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect([1, 1, 0.9])
    for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a_.pane.fill = False; a_.pane.set_edgecolor('lightgray')

# ======================== 数值分析 ========================
def compute_metrics(func, length, nt=360, np_=720, **kw):
    t = np.linspace(0, np.pi, nt); p = np.linspace(0, 2*np.pi, np_)
    TH, PH = np.meshgrid(t, p, indexing='ij')
    E = func(TH, PH, length, **kw); P = np.abs(E)**2; Pm = np.max(P)
    if Pm <= 0: return {'D_dBi':0,'HPBW_E':0,'HPBW_H':0,'FB_dB':0}
    with np.errstate(divide='ignore',invalid='ignore'):
        from scipy.integrate import simpson
        try: Pr = simpson(simpson(P*np.sin(TH), t, axis=0), p)
        except: Pr = np.trapz(np.trapz(P*np.sin(TH), t, axis=0), p)
    if Pr <= 0: return {'D_dBi':0,'HPBW_E':0,'HPBW_H':0,'FB_dB':0}
    D = 10*np.log10(4*np.pi*Pm/Pr)

    def _hpbw(arr, coord):
        a = arr/np.max(arr) if np.max(arr)>0 else arr
        ab = a>=0.5
        if not np.any(ab): return 0
        idx = np.where(ab)[0]
        g = np.split(idx, np.where(np.diff(idx)!=1)[0]+1)
        ml = max(g, key=len) if g else idx
        if len(ml) < 2: return 0
        w = np.degrees(coord[ml[-1]]-coord[ml[0]])
        # If lobe clipped at boundary and another lobe exists, double
        if len(ml) < len(idx) and (ml[0] == 0 or ml[-1] == len(coord)-1):
            w *= 2
        return w

    HE = _hpbw(np.abs(func(t, np.pi/2, length, **kw))**2, t)  # E面: φ=90°
    HH = _hpbw(np.abs(func(t, 0, length, **kw))**2, t)        # H面: φ=0° (XZ⊥Y)

    it = np.argmin(np.abs(t-np.pi/2))
    ip = np.argmin(np.abs(p-np.pi))
    Pf = np.abs(func(t[it], p[0], length, **kw))**2
    Pb = np.abs(func(t[it], p[ip], length, **kw))**2
    FB = 10*np.log10(Pf/Pb) if Pb>0 else 40
    return {'D_dBi':D,'HPBW_E':HE,'HPBW_H':HH,'FB_dB':FB}

# ======================== GUI ========================
class App:
    def __init__(self, root):
        self.root = root
        root.title("1420 MHz 天线仿真器 v2.1")
        root.geometry("1480x960")
        root.minsize(1200, 750)

        # 参数变量
        self.dip_type = tk.StringVar(value="半波长 (λ/2)")

        self.var_d = {}  # 存放所有调节参数
        self._make_vars()
        self._last_pat = None

        self._build_ui()
        self._update()

    def _make_vars(self):
        V = self.var_d
        V['ref_dist']  = tk.DoubleVar(value=LAM/4)
        V['can_D']     = tk.DoubleVar(value=2*DEF_A)
        V['can_H']     = tk.DoubleVar(value=0.3)
        V['cyl_R']     = tk.DoubleVar(value=DEF_RR)
        V['cyl_L']     = tk.DoubleVar(value=DEF_RL)
        V['cyl_ang']   = tk.DoubleVar(value=DEF_ANG)
        V['can_rot_x'] = tk.DoubleVar(value=0.0)
        V['can_rot_y'] = tk.DoubleVar(value=0.0)
        V['dish_a']    = tk.DoubleVar(value=DEF_A)
        V['dish_b']    = tk.DoubleVar(value=DEF_B)
        V['dish_f']    = tk.DoubleVar(value=DEF_F)
        V['offset_ang'] = tk.DoubleVar(value=DEF_ANG)
        V['feed_dx']   = tk.DoubleVar(value=0.0)
        V['feed_dz']   = tk.DoubleVar(value=0.0)
        V['show_struct'] = tk.BooleanVar(value=True)
        V['show_shield'] = tk.BooleanVar(value=True)

    def _build_ui(self):
        mp = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        mp.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        cf = ttk.Frame(mp, width=320); mp.add(cf, weight=0)
        self.nb = ttk.Notebook(mp); mp.add(self.nb, weight=1)

        # 控制面板
        r = 0
        ttk.Label(cf, text="天线仿真配置 v2.1",
                  font=("Segoe UI", 14, "bold")).grid(
            row=r, column=0, columnspan=2, pady=(5,10), sticky="w"); r+=1
        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=3); r+=1

        ttk.Label(cf, text=f"频率: {FREQ/1e6:.1f} MHz | λ = {LAM*100:.1f} cm",
                  font=("Segoe UI", 9)).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=2); r+=1
        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=3); r+=1

        # 偶极子类型 (自动刷新)
        ttk.Label(cf, text="偶极子:").grid(row=r, column=0, sticky="w", pady=3)
        dt = ttk.Combobox(cf, textvariable=self.dip_type,
                      values=list(DIPOLE_TYPES.keys()),
                      state="readonly", width=20)
        dt.grid(row=r, column=1, sticky="w", pady=3)
        dt.bind("<<ComboboxSelected>>", lambda _: self._update())
        r += 1

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=5); r+=1

        # 参数折叠框架
        self._build_param_section(cf, r)
        r += 20  # param section fills rows

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=5); r+=1

        # 按钮
        ttk.Button(cf, text="🔄 更新仿真",
                   command=self._update, width=22).grid(
            row=r, column=0, columnspan=2, pady=5); r+=1
        ttk.Button(cf, text="📊 全部对比",
                   command=self._compare, width=22).grid(
            row=r, column=0, columnspan=2, pady=3); r+=1

        # 导入/导出
        btnf = ttk.Frame(cf)
        btnf.grid(row=r, column=0, columnspan=2, pady=3); r+=1
        ttk.Button(btnf, text="📥 导入参数",
                   command=self._import_params, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btnf, text="📤 导出参数",
                   command=self._export_params, width=10).pack(side=tk.LEFT, padx=2)

        ttk.Button(cf, text="📄 导出图表",
                   command=self._export_all, width=22).grid(
            row=r, column=0, columnspan=2, pady=3); r+=1

        ttk.Separator(cf, orient=tk.HORIZONTAL).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=5); r+=1

        # 状态
        self.st = ttk.Label(cf, text="就绪", font=("Segoe UI", 9))
        self.st.grid(row=r, column=0, columnspan=2, sticky="w", pady=2); r+=1

        # 指标
        mf = ttk.LabelFrame(cf, text="仿真指标", padding=6)
        mf.grid(row=r, column=0, columnspan=2, sticky="ew", pady=5); r+=1
        self.mt = tk.StringVar(value="等待计算...")
        ttk.Label(mf, textvariable=self.mt,
                  font=("Consolas", 10), justify=tk.LEFT).pack(anchor="w")

        cf.columnconfigure(1, weight=1)
        cf.rowconfigure(r, weight=1)

        # 标签页
        self._make_tabs()

    def _build_param_section(self, parent, start_row):
        """可调参数 — 数字输入框"""
        groups = [
            ("天线锅参数", ['dish_a','dish_b','dish_f','offset_ang']),
            ("馈源位置偏移", ['feed_dx','feed_dz']),
            ("反射板参数", ['ref_dist','cyl_R','cyl_L','cyl_ang']),
            ("屏蔽桶", ['can_D','can_H','can_rot_x','can_rot_y','show_shield']),
            ("显示", ['show_struct']),
        ]
        labels = {
            'dish_a':'锅长轴 a (m)', 'dish_b':'锅短轴 b (m)',
            'dish_f':'锅焦距 f (m)', 'offset_ang':'偏馈角 (°)',
            'feed_dx':'馈源偏移 X (m)', 'feed_dz':'馈源偏移 Z (m)',
            'ref_dist':'反射板距离 (m)', 'cyl_R':'半圆柱半径 R (m)',
            'cyl_L':'反射板长度 L (m)', 'cyl_ang':'反射板倾斜角 (°)',
            'can_D':'桶直径 (m)', 'can_H':'桶高度 (m)',
            'can_rot_x':'桶旋转 X (°)', 'can_rot_y':'桶旋转 Y (°)',
            'show_shield':'显示屏蔽桶', 'show_struct':'显示天线结构',
        }
        self._param_ranges = {
            'dish_a':(1e-6,1e3),'dish_b':(1e-6,1e3),'dish_f':(1e-6,1e3),
            'offset_ang':(0,360),
            'feed_dx':(-1e3,1e3),'feed_dz':(-1e3,1e3),
            'ref_dist':(1e-6,1e3),'cyl_R':(1e-6,1e3),
            'cyl_L':(1e-6,1e3),'cyl_ang':(0,360),
            'can_D':(1e-6,1e3),'can_H':(1e-6,1e3),
            'can_rot_x':(-360,360),'can_rot_y':(-360,360),
        }
        _fmts = {k:'.3f' for k in self._param_ranges}
        _fmts.update({'offset_ang':'.1f','cyl_ang':'.1f','can_rot_x':'.1f','can_rot_y':'.1f'})
        bools = {'show_shield','show_struct'}

        frame = ttk.Frame(parent)
        frame.grid(row=start_row, column=0, columnspan=2, sticky="nsew")
        self._param_widgets = {}

        for gname, keys in groups:
            if not keys: continue
            lf = ttk.LabelFrame(frame, text=gname, padding=3)
            lf.pack(fill=tk.X, pady=2)
            for k in keys:
                v = self.var_d[k]
                if k in bools:
                    cb = ttk.Checkbutton(lf, text=labels.get(k,k),
                                         variable=v,
                                         command=self._update)
                    cb.pack(anchor="w", pady=1)
                    self._param_widgets[k] = cb
                else:
                    f2 = ttk.Frame(lf); f2.pack(fill=tk.X, pady=1)
                    ttk.Label(f2, text=labels.get(k,k),
                              width=16, anchor="w").pack(side=tk.LEFT)
                    sv = tk.StringVar(value=f"{v.get():{_fmts.get(k,'.3f')}}")
                    ent = ttk.Entry(f2, textvariable=sv,
                                    width=9, font=("Consolas", 9))
                    ent.pack(side=tk.RIGHT, padx=(3,3))
                    ent.bind("<Return>",
                        lambda _, kk=k, vv=v, svv=sv: self._entry_update(kk, vv, svv))
                    ent.bind("<FocusOut>",
                        lambda _, kk=k, vv=v, svv=sv: self._entry_update(kk, vv, svv))
                    self._param_widgets[k+'_ent'] = sv

    def _entry_update(self, k, var, sv):
        try:
            val = float(sv.get())
            lo, hi = self._param_ranges.get(k, (0, 1e6))
            val = max(lo, min(hi, val))
            var.set(val)
            sv.set(f"{val:.3f}" if k not in ('offset_ang','cyl_ang') else f"{val:.1f}")
            self._update()
        except ValueError:
            sv.set(f"{var.get():.3f}" if k not in ('offset_ang','cyl_ang') else f"{var.get():.1f}")

    def _make_tabs(self):
        self.t_pol = ttk.Frame(self.nb); self.nb.add(self.t_pol, text="极坐标方向图")
        self.t_3d = ttk.Frame(self.nb); self.nb.add(self.t_3d, text="3D 方向图")
        self.t_str = ttk.Frame(self.nb); self.nb.add(self.t_str, text="天线结构")
        self.t_ov = ttk.Frame(self.nb); self.nb.add(self.t_ov, text="叠加视图")
        self.t_comp = ttk.Frame(self.nb); self.nb.add(self.t_comp, text="配置对比")

    def _get_cfg(self):
        return self.cfg_name.get(), self.dip_type.get(), self.var_d

    def _update(self):
        self.st.config(text="计算中..."); self.root.update_idletasks()
        threading.Thread(target=self._compute, daemon=True).start()

    def _compute(self):
        try:
            V = self.var_d
            length = DIPOLE_TYPES[self.dip_type.get()]
            a = V['dish_a'].get(); b = V['dish_b'].get(); f = V['dish_f'].get()
            D_eff = 2 * np.sqrt(a * b)
            fD = f / D_eff if D_eff > 0 else 0.4
            d_ref = V['ref_dist'].get()
            feed_dx = V['feed_dx'].get()
            feed_dz = V['feed_dz'].get()
            shield_D = V['can_D'].get()

            # PO 光线追踪闭包
            def ray_trace(t, p, L):
                return pattern_ray_trace(t, p, L, d_ref, a, b, f,
                                          feed_dx, feed_dz, shield_D,
                                          offset_ang=V['offset_ang'].get())

            # === 指标 ===
            # 指向性: 由锅面物理口径估算
            A_phys = np.pi * a * b
            eta_ap = 0.55
            D_dBi = 10 * np.log10(4 * np.pi * eta_ap * A_phys / LAM**2)

            # HPBW: 从 1D 方向图切面精确提取
            t_fine = np.linspace(0, np.pi, 720)
            E_e = np.abs(ray_trace(t_fine, np.pi/2, length))
            E_h = np.abs(ray_trace(t_fine, np.pi, length))
            P_e = E_e / np.max(E_e) if np.max(E_e) > 0 else E_e
            P_h = E_h / np.max(E_h) if np.max(E_h) > 0 else E_h

            def _hpbw_1d(theta, Pn):
                ab = Pn >= 0.5
                if not np.any(ab): return 0.0
                idx = np.where(ab)[0]
                g = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
                ml = max(g, key=len) if g else idx
                if len(ml) < 2: return 0.0
                w = np.degrees(theta[ml[-1]] - theta[ml[0]])
                if len(ml) < len(idx) and (ml[0] == 0 or ml[-1] == len(theta) - 1):
                    w *= 2
                return w

            hpbw_e = _hpbw_1d(t_fine, P_e**2)
            hpbw_h = _hpbw_1d(t_fine, P_h**2)

            # 前后比 (主瓣方向 θ=60°, φ=π 与背瓣)
            bs_t = np.radians(60)
            Pf = np.abs(ray_trace(bs_t, np.pi, length))**2
            Pb = np.abs(ray_trace(np.pi - bs_t, 0.0, length))**2
            fb = 10 * np.log10(Pf/Pb) if Pb > 0 else 40.0

            txt = (f"指向性 D: {D_dBi:.2f} dBi (PO 物理光学)\n"
                   f"E面 HPBW: {hpbw_e:.1f}°\n"
                   f"H面 HPBW: {hpbw_h:.1f}°\n"
                   f"前后比 F/B: {fb:.1f} dB"
                   f"\n有效口径 D={D_eff:.3f}m  f/D={fD:.3f}"
                   f"\n反射板距离 d={d_ref*100:.1f}cm")
            self.root.after(0, lambda: self.mt.set(txt))
            self.root.after(0, lambda: self.st.config(text="✓ 完成"))

            m = {'D_dBi': D_dBi, 'HPBW_E': hpbw_e, 'HPBW_H': hpbw_h, 'FB_dB': fb}

            # 缓存方向图 (粗网格, 用于叠加渲染)
            t = np.linspace(0, np.pi, 60); p_ = np.linspace(0, 2*np.pi, 90)
            TH, PH = np.meshgrid(t, p_, indexing='ij')
            Ep = ray_trace(TH, PH, length)
            self._last_pat = (t, p_, Ep)

            self._plot_polar(length, ray_trace, m)
            self._plot_3d(length, ray_trace)
            self._plot_struct(length)
            self._plot_overlay(length)
        except Exception as e:
            self.root.after(0, lambda: self.st.config(text=f"❌ {e}"))
            import traceback; traceback.print_exc()

    def _plot_polar(self, L, func, m):
        fig = Figure(figsize=(7,6), dpi=100)
        fig.suptitle(f"{self.dip_type.get()} — 全金属模型",
                     fontsize=12, fontweight="bold")
        th = np.linspace(0, 2*np.pi, 720)
        for ax, tt, nm in [
                (fig.add_subplot(1,2,1,projection='polar'), np.pi/2, "E 面 (φ=90°)"),
                (fig.add_subplot(1,2,2,projection='polar'), np.pi, "H 面 (φ=180° 主瓣)")]:
            E = np.abs(func(th, tt, L))
            P = E/np.max(E) if np.max(E)>0 else E
            r = 10**(np.clip(10*np.log10(np.clip(P,1e-10,None)),-30,0)/20)
            ax.plot(th, r, 'b-', lw=1.5); ax.set_title(nm, va="bottom", fontsize=10)
            ax.set_ylim(0,1.05); ax.grid(True, alpha=0.3)
        h = m.get('HPBW_E',0)
        if h>0:
            hh=np.radians(h/2)
            fig.axes[0].plot([np.pi-hh, np.pi+hh],[0.707,0.707],'r--',lw=1.5,alpha=0.7)
            fig.axes[0].annotate(f"HPBW={h:.1f}°",xy=(np.pi,0.85),fontsize=9,color='red',ha='center')
        h=m.get('HPBW_H',0)
        if 10<h<170:
            hh=np.radians(h/2)
            fig.axes[1].plot([np.pi-hh, np.pi+hh],[0.707,0.707],'r--',lw=1.5,alpha=0.7)
            fig.axes[1].annotate(f"HPBW={h:.1f}°",xy=(np.pi,0.85),fontsize=9,color='red',ha='center')
        fig.tight_layout(); self._embed(fig, self.t_pol)

    def _plot_3d(self, L, func):
        fig = Figure(figsize=(8,7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(0, np.pi, 60); ph = np.linspace(0, 2*np.pi, 90)
        TH, PH = np.meshgrid(t, ph, indexing='ij')
        E = np.abs(func(TH, PH, L)); P = E/np.max(E) if np.max(E)>0 else E
        Pd = np.clip(10*np.log10(np.clip(P,1e-10,None)), -20, 0)
        n = plt.Normalize(-20,0); c = plt.cm.viridis(n(Pd))
        R = P; X=R*np.sin(TH)*np.cos(PH); Y=R*np.sin(TH)*np.sin(PH); Z=R*np.cos(TH)
        ax.plot_surface(X,Y,Z, facecolors=c, rstride=2, cstride=2, alpha=0.9, lw=0)
        u,v=np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
        ax.plot_wireframe(0.3*np.sin(u)*np.cos(v),0.3*np.sin(u)*np.sin(v),0.3*np.cos(u),
                          color='gray', alpha=0.06, lw=0.3)
        ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_zlim(-1.1,1.1)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title("{} — 全金属模型".format(self.dip_type.get()),
                     fontsize=11, fontweight="bold")
        for a_ in [ax.xaxis,ax.yaxis,ax.zaxis]: a_.pane.fill=False
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        m=plt.cm.ScalarMappable(norm=n,cmap=plt.cm.viridis); m.set_array([])
        fig.colorbar(m, ax=ax, shrink=0.6, pad=0.1).set_label('归一化方向图 (dB)', fontsize=9)
        fig.tight_layout(); self._embed(fig, self.t_3d)

    def _plot_struct(self, length):
        if not self.var_d['show_struct'].get(): return
        fig = Figure(figsize=(8,7), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        V = self.var_d
        render_structure(ax, length,
                         V['dish_a'].get(), V['dish_b'].get(), V['dish_f'].get(),
                         V['offset_ang'].get(),
                         V['show_shield'].get(), V['can_D'].get(), V['can_H'].get(),
                         V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                         V['feed_dx'].get(), V['feed_dz'].get(),
                         V['can_rot_x'].get(), V['can_rot_y'].get())
        ax.set_title("天线结构: 椭圆锅 + 半圆柱反射板 + 偶极子∥锅面",
                     fontsize=11, fontweight="bold")
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([0],[0],color='blue',lw=3,label='偶极子'),
            Line2D([0],[0],color='gold',lw=6,alpha=0.4,label='半圆柱反射板'),
            Line2D([0],[0],color='gray',lw=1,alpha=0.4,label='屏蔽桶'),
        ], loc='upper right', fontsize=7)
        fig.tight_layout(); self._embed(fig, self.t_str)

    def _plot_overlay(self, length):
        if self._last_pat is None: return
        t, p_, Ep = self._last_pat
        fig = Figure(figsize=(9,8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        V = self.var_d
        render_overlay(ax, t, p_, Ep,
                       V['dish_a'].get(), V['dish_b'].get(), V['dish_f'].get(),
                       V['offset_ang'].get(),
                       V['show_shield'].get(), V['can_D'].get(), V['can_H'].get(),
                       V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                       length, V['feed_dx'].get(), V['feed_dz'].get(),
                       V['can_rot_x'].get(), V['can_rot_y'].get())
        ax.set_title("结构 + 方向图叠加 (颜色=增益强度)", fontsize=11, fontweight="bold")
        Pn = np.abs(Ep)/np.max(np.abs(Ep)) if np.max(np.abs(Ep))>0 else np.abs(Ep)
        n_ = plt.Normalize(0,1); m_=plt.cm.ScalarMappable(norm=n_,cmap=plt.cm.hot); m_.set_array([])
        fig.colorbar(m_, ax=ax, shrink=0.5, pad=0.1).set_label('归一化增益', fontsize=9)
        fig.tight_layout(); self._embed(fig, self.t_ov)

    def _embed(self, fig, parent):
        def e():
            for w in parent.winfo_children(): w.destroy()
            c = FigureCanvasTkAgg(fig, master=parent); c.draw()
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            t_ = NavigationToolbar2Tk(c, parent, pack_toolbar=False)
            t_.update(); t_.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.after(0, e)

    def _compare(self):
        self.st.config(text="计算对比..."); self.root.update_idletasks()
        threading.Thread(target=self._comp_run, daemon=True).start()

    def _comp_run(self):
        try:
            V = self.var_d
            L = DIPOLE_TYPES[self.dip_type.get()]
            a = V['dish_a'].get(); b = V['dish_b'].get(); f = V['dish_f'].get()
            D0 = 2 * np.sqrt(a * b)
            d0 = V['ref_dist'].get()

            # 对比几种口径
            scales = [0.8, 1.0, 1.2]
            Ds = [D0 * s for s in scales]
            nm = [f"D={D:.2f}m" for D in Ds]
            feed_dx = V['feed_dx'].get()
            feed_dz = V['feed_dz'].get()
            shield_D = V['can_D'].get()
            oa = V['offset_ang'].get()

            def _hpbw_1d(theta, Pn):
                ab = Pn >= 0.5
                if not np.any(ab): return 0.0
                idx = np.where(ab)[0]
                g = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
                ml = max(g, key=len) if g else idx
                if len(ml) < 2: return 0.0
                w = np.degrees(theta[ml[-1]] - theta[ml[0]])
                if len(ml) < len(idx) and (ml[0] == 0 or ml[-1] == len(theta) - 1):
                    w *= 2
                return w

            mm = []
            tf = np.linspace(0, np.pi, 360)
            for s in scales:
                aa = a * s; bb = b * s
                rt = lambda t,p,L,dd=aa,ee=bb: pattern_ray_trace(t,p,L,d0,dd,ee,f,feed_dx,feed_dz,shield_D,offset_ang=oa)
                D_dBi = 10 * np.log10(4*np.pi*0.55*np.pi*aa*bb / LAM**2)
                Ee = np.abs(rt(tf, np.pi/2, L))
                Eh = np.abs(rt(tf, np.pi, L))
                pe = Ee/np.max(Ee) if np.max(Ee)>0 else Ee
                ph = Eh/np.max(Eh) if np.max(Eh)>0 else Eh
                he = _hpbw_1d(tf, pe**2)
                hh = _hpbw_1d(tf, ph**2)
                bs_t = np.radians(60)
                Pf = np.abs(rt(bs_t, np.pi, L))**2
                Pb = np.abs(rt(np.pi - bs_t, 0.0, L))**2
                fb = 10*np.log10(Pf/Pb) if Pb>0 else 40.0
                mm.append({'D_dBi':D_dBi, 'HPBW_E':he, 'HPBW_H':hh, 'FB_dB':fb})

            fig=Figure(figsize=(12,10),dpi=100)
            gs=GridSpec(2,3,figure=fig,hspace=0.35,wspace=0.3)
            fig.suptitle("1420 MHz PO 光线追踪 — 口径对比",fontsize=14,fontweight="bold")
            cl=['blue','green','red']; th=np.linspace(0,2*np.pi,720)
            ax_e=fig.add_subplot(gs[0,0],projection='polar')
            for i,(s,c) in enumerate(zip(scales,cl)):
                aa = a*s; bb = b*s
                def _po(t,p,dd=aa,ee=bb): return pattern_ray_trace(t,p,L,d0,dd,ee,f,feed_dx,feed_dz,shield_D,offset_ang=oa)
                E=np.abs(_po(th,np.pi/2)); P=E/np.max(E) if np.max(E)>0 else E
                r=10**(np.clip(10*np.log10(np.clip(P,1e-10,None)),-30,0)/20)
                ax_e.plot(th,r,color=c,lw=1.2,label=nm[i])
            ax_e.set_title("E 面 (φ=90°)",fontsize=10); ax_e.set_ylim(0,1.05); ax_e.legend(fontsize=7)
            ax_h=fig.add_subplot(gs[0,1],projection='polar')
            for i,(s,c) in enumerate(zip(scales,cl)):
                aa = a*s; bb = b*s
                def _po(t,p,dd=aa,ee=bb): return pattern_ray_trace(t,p,L,d0,dd,ee,f,feed_dx,feed_dz,shield_D,offset_ang=oa)
                E=np.abs(_po(th,0)); P=E/np.max(E) if np.max(E)>0 else E
                r=10**(np.clip(10*np.log10(np.clip(P,1e-10,None)),-30,0)/20)
                ax_h.plot(th,r,color=c,lw=1.2,label=nm[i])
            ax_h.set_title("H 面 (φ=0)",fontsize=10); ax_h.set_ylim(0,1.05); ax_h.legend(fontsize=7)
            ax_t=fig.add_subplot(gs[0,2]); ax_t.axis('off')
            d_=[[nm[i],f"{mm[i]['D_dBi']:.1f}",f"{mm[i]['HPBW_E']:.0f}°",f"{mm[i]['HPBW_H']:.0f}°",f"{mm[i]['FB_dB']:.1f}"] for i in range(3)]
            t_=ax_t.table(cellText=d_,colLabels=["口径","D(dBi)","HPBW_E","HPBW_H","F/B(dB)"],loc='center',cellLoc='center',colWidths=[0.22,0.16,0.16,0.16,0.16])
            t_.auto_set_font_size(False); t_.set_fontsize(9); t_.scale(1,1.8)
            ax_t.set_title("指标对比",fontsize=11,fontweight="bold",pad=20)
            ax_g=fig.add_subplot(gs[1,:])
            Dr=np.linspace(0.3,1.0,20)
            gs_plot = []
            for dD in Dr:
                gs_plot.append(10*np.log10(4*np.pi*0.55*np.pi*(dD/2)**2 / LAM**2))
            ax_g.plot(Dr,gs_plot,'r-',lw=2); ax_g.set_xlabel("有效口径 D (m)",fontsize=10)
            ax_g.set_ylabel("指向性 (dBi)",fontsize=10)
            ax_g.set_title("PO 模型增益 vs 口径 (理论极限)",fontsize=11,fontweight="bold")
            ax_g.grid(True,alpha=0.3)
            fig.tight_layout(); self._embed(fig, self.t_comp)
            self.root.after(0, lambda: self.nb.select(4))
            self.root.after(0, lambda: self.st.config(text="✓ 对比完成"))
        except Exception as e:
            self.root.after(0, lambda: self.st.config(text=f"❌ {e}"))
            import traceback; traceback.print_exc()

    def _export_params(self):
        """导出参数到 JSON 文件"""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            title="导出参数")
        if not path:
            return
        try:
            data = {k: v.get() for k, v in self.var_d.items()}
            data['dip_type'] = self.dip_type.get()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.st.config(text=f"✓ 已导出: {os.path.basename(path)}")
        except Exception as e:
            self.st.config(text=f"❌ 导出失败: {e}")

    def _import_params(self):
        """从 JSON 文件导入参数"""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            title="导入参数")
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 更新 dip_type
            if 'dip_type' in data and data['dip_type'] in DIPOLE_TYPES:
                self.dip_type.set(data['dip_type'])
            # 更新数值参数
            _fmts = {k: '.3f' for k in self._param_ranges}
            _fmts.update({'offset_ang': '.1f', 'cyl_ang': '.1f'})
            for k, var in self.var_d.items():
                if k in data:
                    try:
                        val = float(data[k])
                        lo, hi = self._param_ranges.get(k, (0, 1e6))
                        val = max(lo, min(hi, val))
                        var.set(val)
                        # 同步更新输入框显示
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
        """一键导出所有图表到 PNG/PDF"""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("PDF 文档", "*.pdf"), ("所有文件", "*.*")],
            title="导出图表")
        if not path:
            return
        self.st.config(text="渲染中..."); self.root.update_idletasks()
        try:
            V = self.var_d
            L = DIPOLE_TYPES[self.dip_type.get()]
            a = V['dish_a'].get(); b = V['dish_b'].get(); f = V['dish_f'].get()
            d_ref = V['ref_dist'].get()
            feed_dx = V['feed_dx'].get(); feed_dz = V['feed_dz'].get()
            shield_D = V['can_D'].get(); shield_H = V['can_H'].get()
            oa = V['offset_ang'].get()
            rx = V['can_rot_x'].get(); ry = V['can_rot_y'].get()

            def rt(t, p, L):
                return pattern_ray_trace(t, p, L, d_ref, a, b, f,
                                         feed_dx, feed_dz, shield_D, offset_ang=oa)

            th = np.linspace(0, np.pi, 720)
            Ee = np.abs(rt(th, np.pi/2, L))
            Eh = np.abs(rt(th, np.pi, L))
            Pe = Ee / np.max(Ee) if np.max(Ee) > 0 else Ee
            Ph = Eh / np.max(Eh) if np.max(Eh) > 0 else Eh

            # 指向性
            A_phys = np.pi * a * b; eta_ap = 0.55
            D_dBi = 10 * np.log10(4 * np.pi * eta_ap * A_phys / LAM**2)

            def _hpbw(theta, Pn):
                ab = Pn >= 0.5
                if not np.any(ab): return 0.0
                idx = np.where(ab)[0]; g = np.split(idx, np.where(np.diff(idx)!=1)[0]+1)
                ml = max(g, key=len) if g else idx
                if len(ml) < 2: return 0.0
                w = np.degrees(theta[ml[-1]] - theta[ml[0]])
                if len(ml) < len(idx) and (ml[0] == 0 or ml[-1] == len(theta)-1): w *= 2
                return w

            hpbw_e = _hpbw(th, Pe**2); hpbw_h = _hpbw(th, Ph**2)
            bs_t = np.radians(60)
            Pf = np.abs(rt(bs_t, np.pi, L))**2
            Pb = np.abs(rt(np.pi-bs_t, 0.0, L))**2
            fb = 10*np.log10(Pf/Pb) if Pb>0 else 40.0

            # 大图排版
            fig = Figure(figsize=(16, 20), dpi=120)
            gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1.4])

            # ---- (0,0) 极坐标方向图 ----
            a0 = fig.add_subplot(gs[0, 0], projection='polar')
            r_e = 10**(np.clip(10*np.log10(np.clip(Pe,1e-10,None)), -30, 0)/20)
            r_h = 10**(np.clip(10*np.log10(np.clip(Ph,1e-10,None)), -30, 0)/20)
            a0.plot(th, r_e, 'b-', lw=1.2, label='E 面 (φ=90°)')
            a0.plot(th, r_h, 'r-', lw=1.2, label='H 面 (φ=180° 主瓣)')
            a0.set_ylim(0, 1.05); a0.grid(True, alpha=0.3)
            a0.set_title("极坐标方向图", fontsize=12, fontweight="bold", pad=12)
            a0.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.3, 1.0))
            # HPBW 标注
            if hpbw_e > 0:
                hh = np.radians(hpbw_e/2)
                a0.plot([np.pi-hh, np.pi+hh], [0.707]*2, 'b--', lw=1.5, alpha=0.5)
                a0.annotate(f"E HPBW={hpbw_e:.1f}°", xy=(np.pi, 0.8),
                           fontsize=8, color='blue', ha='center')
            if hpbw_h > 0:
                hh = np.radians(hpbw_h/2)
                a0.plot([np.pi-hh, np.pi+hh], [0.707]*2, 'r--', lw=1.5, alpha=0.5)
                a0.annotate(f"H HPBW={hpbw_h:.1f}°", xy=(np.pi, 0.7),
                           fontsize=8, color='red', ha='center')

            # ---- (0,1) 3D 方向图 ----
            a1 = fig.add_subplot(gs[0, 1], projection='3d')
            t3 = np.linspace(0, np.pi, 60); p3 = np.linspace(0, 2*np.pi, 90)
            T3, P3 = np.meshgrid(t3, p3, indexing='ij')
            E3 = np.abs(rt(T3, P3, L)); P3d = E3/np.max(E3) if np.max(E3)>0 else E3
            Pd = np.clip(10*np.log10(np.clip(P3d,1e-10,None)), -20, 0)
            n_ = plt.Normalize(-20, 0); c_ = plt.cm.viridis(n_(Pd))
            R3 = P3d
            X3 = R3*np.sin(T3)*np.cos(P3); Y3 = R3*np.sin(T3)*np.sin(P3); Z3 = R3*np.cos(T3)
            a1.plot_surface(X3, Y3, Z3, facecolors=c_, rstride=2, cstride=2, alpha=0.9, lw=0)
            a1.set_xlim(-1.1, 1.1); a1.set_ylim(-1.1, 1.1); a1.set_zlim(-1.1, 1.1)
            a1.set_title("3D 方向图", fontsize=12, fontweight="bold")
            for ax_ in [a1.xaxis, a1.yaxis, a1.zaxis]: ax_.pane.fill = False
            a1.set_xticks([]); a1.set_yticks([]); a1.set_zticks([])

            # ---- (1,0) 天线结构 ----
            a2 = fig.add_subplot(gs[1, 0], projection='3d')
            render_structure(a2, L, a, b, f, oa,
                             V['show_shield'].get(), shield_D, shield_H,
                             V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                             feed_dx, feed_dz, rx, ry)
            a2.set_title("天线结构", fontsize=12, fontweight="bold")

            # ---- (1,1) 叠加视图 ----
            a3 = fig.add_subplot(gs[1, 1], projection='3d')
            t_ = np.linspace(0, np.pi, 60); p_ = np.linspace(0, 2*np.pi, 90)
            TT, PP = np.meshgrid(t_, p_, indexing='ij')
            Ep = rt(TT, PP, L)
            render_overlay(a3, t_, p_, Ep, a, b, f, oa,
                           V['show_shield'].get(), shield_D, shield_H,
                           V['cyl_R'].get(), V['cyl_L'].get(), V['cyl_ang'].get(),
                           L, feed_dx, feed_dz, rx, ry)
            a3.set_title("结构 + 方向图叠加", fontsize=12, fontweight="bold")

            # ---- (2,0:2) 指标汇总 ----
            a4 = fig.add_subplot(gs[2, 0]); a4.axis('off')
            txt = (f"频率: {FREQ/1e6:.1f} MHz  λ={LAM*100:.1f} cm\n"
                   f"偶极子: {self.dip_type.get()}  L={L*1000:.1f} mm\n"
                   f"锅: a={a*100:.1f}cm b={b*100:.1f}cm  f={f*100:.1f}cm\n"
                   f"偏馈角: {oa:.1f}°  反射板距离: {d_ref*100:.1f}cm\n"
                   f"桶: D={shield_D*100:.1f}cm  H={shield_H*100:.1f}cm\n"
                   f"旋转: X={rx:.1f}°  Y={ry:.1f}°\n\n"
                   f"✦ 指向性: {D_dBi:.2f} dBi\n"
                   f"✦ E面 HPBW: {hpbw_e:.1f}°\n"
                   f"✦ H面 HPBW: {hpbw_h:.1f}°\n"
                   f"✦ 前后比 F/B: {fb:.1f} dB\n"
                   f"✦ 有效口径: {2*np.sqrt(a*b):.3f}m  f/D={(f/(2*np.sqrt(a*b)) if a*b>0 else 0):.3f}")
            a4.text(0.05, 0.95, txt, transform=a4.transAxes,
                   fontsize=10, fontfamily='monospace', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

            # 增益 vs 口径图
            a5 = fig.add_subplot(gs[2, 1])
            Dr = np.linspace(0.3, 1.0, 50)
            G = [10*np.log10(4*np.pi*0.55*np.pi*(d/2)**2 / LAM**2) for d in Dr]
            a5.plot(Dr, G, 'r-', lw=2)
            if a*b > 0:
                D_eff = 2*np.sqrt(a*b)
                G_pt = 10*np.log10(4*np.pi*0.55*np.pi*(D_eff/2)**2 / LAM**2)
                a5.plot(D_eff, G_pt, 'bo', markersize=8)
                a5.annotate(f"当前 D={D_eff:.2f}m\nG={G_pt:.1f}dBi",
                           xy=(D_eff, G_pt), fontsize=9, color='blue',
                           xytext=(5, 5), textcoords='offset points')
            a5.set_xlabel("有效口径 D (m)", fontsize=10)
            a5.set_ylabel("指向性 (dBi)", fontsize=10)
            a5.set_title("增益 vs 口径 (理论极限)", fontsize=11, fontweight="bold")
            a5.grid(True, alpha=0.3)

            fig.suptitle(f"1420 MHz 天线仿真报告 — {self.dip_type.get()} 全金属模型",
                        fontsize=16, fontweight="bold", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(path, dpi=200, bbox_inches='tight')
            self.st.config(text=f"✓ 已导出: {os.path.basename(path)}")
        except Exception as e:
            self.st.config(text=f"❌ 导出失败: {e}")
            import traceback; traceback.print_exc()

    # ======================== 主入口 ========================
def main():
    import traceback
    try:
        for fn in ['SimHei','Microsoft YaHei','WenQuanYi Micro Hei','Noto Sans CJK SC','DejaVu Sans']:
            try:
                plt.rcParams['font.family']=fn
                f_,a_=plt.subplots(figsize=(0.01,0.01)); a_.set_title("测试"); plt.close(f_); break
            except: plt.rcParams['font.family']='sans-serif'; continue
        plt.rcParams['font.size']=10; plt.rcParams['axes.unicode_minus']=False
        root=tk.Tk(); App(root); root.mainloop()
    except Exception as e:
        lp=os.path.join(os.path.dirname(os.path.abspath(__file__)),'error_log.txt')
        with open(lp,'w',encoding='utf-8') as f: f.write(f"错误: {e}\n\n"); traceback.print_exc(file=f)
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0,f"程序启动失败:\n\n{e}\n\n详情: {lp}","天线仿真器 - 错误",0x10)
        except: pass
        raise

if __name__=="__main__":
    main()
