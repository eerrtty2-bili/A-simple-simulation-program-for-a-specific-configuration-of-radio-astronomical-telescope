"""3D 几何网格: 抛物面、屏蔽桶、偶极子、半圆柱反射板"""

import numpy as np


def _offset_axis(f, offset_ang, feed_dx, feed_dz):
    """计算偏馈抛物面光轴 u_hat, 口径基矢 e1,e2 及馈源位 F"""
    alpha = np.radians(offset_ang)
    F = np.array([feed_dx, 0.0, f + feed_dz], dtype=float)
    v = F / np.linalg.norm(F) if np.linalg.norm(F) > 1e-12 else np.array([0, 0, 1])
    beta = np.arctan2(v[2], v[0])
    beta_p = beta + alpha
    ux = np.cos(beta_p)
    uz = np.sin(beta_p)
    if uz < 0:
        beta_p = beta - alpha
        ux = np.cos(beta_p)
        uz = np.sin(beta_p)
    u_hat = np.array([ux, 0.0, uz])
    u_hat /= np.linalg.norm(u_hat)
    e1 = np.array([u_hat[2], 0.0, -u_hat[0]])
    e1 /= np.linalg.norm(e1)
    e2 = np.array([0.0, 1.0, 0.0])
    return u_hat, e1, e2, F


def dish_mesh(a, b, f, nr=20, np_=30, offset_ang=60, feed_dx=0, feed_dz=0):
    """抛物面锅网格"""
    u_hat, e1, e2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    u = np.linspace(0, 2 * np.pi, np_)
    v = np.linspace(1e-6, 1, nr)
    U, V = np.meshgrid(u, v)
    u_apt = V * a * np.cos(U)
    v_apt = V * b * np.sin(U)
    uv2 = u_apt**2 + v_apt**2
    w = uv2 / (4 * f) - f
    X = F[0] + u_apt * e1[0] + w * u_hat[0]
    Y = F[1] + v_apt * e2[1] + w * u_hat[1]
    Z = F[2] + u_apt * e1[2] + w * u_hat[2]
    Z -= np.min(Z)
    return X, Y, Z


def dish_edge(a, b, f, n=60, offset_ang=60, feed_dx=0, feed_dz=0):
    """抛物面锅边缘曲线"""
    u_hat, e1, e2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    u = np.linspace(0, 2 * np.pi, n)
    u_apt = a * np.cos(u)
    v_apt = b * np.sin(u)
    uv2 = u_apt**2 + v_apt**2
    w = uv2 / (4 * f) - f
    X = F[0] + u_apt * e1[0] + w * u_hat[0]
    Y = F[1] + v_apt * e2[1] + w * u_hat[1]
    Z = F[2] + u_apt * e1[2] + w * u_hat[2]
    Z -= np.min(Z)
    return X, Y, Z


def shield_mesh(a, b, f, offset_ang, feed_dx, feed_dz, H, D=None,
                rot_x=0, rot_y=0, nh=15, np_=30):
    """屏蔽桶: 正圆柱体, 轴线沿入射光轴"""
    u_hat, e1, e2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)
    u = np.linspace(0, 2 * np.pi, np_)
    v = np.linspace(0, H, nh)
    U, V = np.meshgrid(u, v)
    R = D / 2 if D is not None else np.sqrt(a * b)
    cu = R * np.cos(U)
    cv = R * np.sin(U)
    uv2 = cu**2 + cv**2
    w0 = uv2 / (4 * f) - f
    X0 = F[0] + cu * e1[0] + cv * e2[0] + w0 * u_hat[0]
    Y0 = F[1] + cu * e1[1] + cv * e2[1] + w0 * u_hat[1]
    Z0 = F[2] + cu * e1[2] + cv * e2[2] + w0 * u_hat[2]
    # 沿 +u_hat 方向拉伸
    X = X0 + V * u_hat[0]
    Y = Y0 + V * u_hat[1]
    Z = Z0 + V * u_hat[2]
    # 三维旋转
    if rot_x != 0 or rot_y != 0:
        cx, cy, cz = F - f * u_hat
        Xt = X - cx
        Yt = Y - cy
        Zt = Z - cz
        if rot_x != 0:
            a_ = np.radians(rot_x)
            c, s = np.cos(a_), np.sin(a_)
            Yt, Zt = Yt * c - Zt * s, Yt * s + Zt * c
        if rot_y != 0:
            a_ = np.radians(rot_y)
            c, s = np.cos(a_), np.sin(a_)
            Xt, Zt = Xt * c + Zt * s, -Xt * s + Zt * c
        X = Xt + cx
        Y = Yt + cy
        Z = Zt + cz
    Z -= np.min(Z)
    return X, Y, Z


def dipole_line(length, center=(0, 0, 0)):
    """偶极子线段 (沿 Y 轴)"""
    cx, cy, cz = center
    h = length / 2
    return [(cx, cy - h, cz),
            (cx, cy - h, cz + 0.002),
            (cx, cy + h, cz + 0.002),
            (cx, cy + h, cz)]


def semi_cylinder_mesh(R, length, angle_deg, n_u=25, n_v=20):
    """180° 半圆柱反射板网格, 圆柱轴沿 Y"""
    u = np.linspace(-np.pi / 2, np.pi / 2, n_u)
    v = np.linspace(-length / 2, length / 2, n_v)
    U, V = np.meshgrid(u, v)

    Xl = R * np.sin(U)
    Yl = V
    Zl = R * np.cos(U)

    ang = np.radians(angle_deg)
    Xg = Xl * np.cos(ang) + Zl * np.sin(ang)
    Zg = -Xl * np.sin(ang) + Zl * np.cos(ang)
    Yg = Yl

    return Xg, Yg, Zg
