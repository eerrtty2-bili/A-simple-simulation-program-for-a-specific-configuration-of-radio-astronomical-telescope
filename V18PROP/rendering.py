"""matplotlib 3D 渲染: 天线结构 / 方向图叠加"""

import numpy as np
import matplotlib.pyplot as plt
from geometry import (_offset_axis, dish_mesh, dish_edge,
                      shield_mesh, dipole_line, semi_cylinder_mesh)


def render_structure(ax, dip_len, dish_a, dish_b, dish_f,
                     offset_ang, show_shield, shield_D, shield_H,
                     cyl_R, cyl_len, cyl_ang, feed_dx=0, feed_dz=0,
                     can_rot_x=0, can_rot_y=0,
                     nr=30, np_=48, cyl_nu=36, cyl_nv=28,
                     show_dish=True, show_reflector=True):
    """在 3D 坐标轴上绘制天线结构"""
    f = dish_f
    feed = np.array([feed_dx, 0, f + feed_dz])
    u_hat, e1, e2, F = _offset_axis(f, offset_ang, feed_dx, feed_dz)

    # 1. 抛物面锅
    if show_dish:
        Xd, Yd, Zd = dish_mesh(dish_a, dish_b, f, nr=nr, np_=np_,
                               offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
        rstride = max(1, nr // 15)
        cstride = max(1, np_ // 24)
        ax.plot_wireframe(Xd, Yd, Zd, rstride=rstride, cstride=cstride,
                          color='gray', alpha=0.2, linewidth=0.4)
        ax.plot_surface(Xd, Yd, Zd, color='silver', alpha=0.12,
                        rstride=2, cstride=2, edgecolor='none')
        xe, ye, ze = dish_edge(dish_a, dish_b, f,
                               offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
        ax.plot(xe, ye, ze, 'k-', lw=1.3, alpha=0.6)

    # 2. 屏蔽桶
    if show_shield:
        Xs, Ys, Zs = shield_mesh(dish_a, dish_b, f, offset_ang, feed_dx, feed_dz, shield_H,
                                  D=shield_D, rot_x=can_rot_x, rot_y=can_rot_y,
                                  nh=nr // 2, np_=np_)
        ax.plot_surface(Xs, Ys, Zs, color='darkgray', alpha=0.06,
                        rstride=2, cstride=2, edgecolor='gray',
                        linewidth=0.2, linestyle='--')
        ax.plot(Xs[0, :], Ys[0, :], Zs[0, :], color='gray', lw=0.8, alpha=0.4)

    # 3. 偏馈角标注
    r_ang = 0.08
    u_len = 0.35 * dish_a
    ax.plot([feed[0], feed[0] + u_len * u_hat[0]], [0, 0],
            [feed[2], feed[2] + u_len * u_hat[2]], 'r--', lw=0.8, alpha=0.5)
    ax.plot([feed[0], 0], [0, 0], [feed[2], 0], 'r:', lw=0.5, alpha=0.3)
    v_to_origin = -F / np.linalg.norm(F)
    half_dir = v_to_origin + u_hat
    half_dir = half_dir / np.linalg.norm(half_dir)
    lx = feed[0] + r_ang * half_dir[0]
    lz = feed[2] + r_ang * half_dir[2]
    ax.text(lx, 0, lz, f"{offset_ang}°", color='red', fontsize=8, fontweight='bold')

    # 4. 偶极子
    dip = dipole_line(dip_len, center=feed)
    dx = [p[0] for p in dip]
    dy = [p[1] for p in dip]
    dz = [p[2] for p in dip]
    ax.plot(dx, dy, dz, 'b-', lw=4, alpha=0.9, label='偶极子')
    ax.scatter(dx, dy, dz, color='blue', s=20, alpha=0.8, zorder=5)
    ax.text(feed[0] + 0.08, 0, feed[2] + 0.005, "偶极子∥锅面",
            fontsize=7, color='blue', ha='center')

    # 5. 半圆柱反射板
    if show_reflector:
        Xc, Yc, Zc = semi_cylinder_mesh(cyl_R, cyl_len, cyl_ang,
                                         n_u=cyl_nu, n_v=cyl_nv)
        ang_r = np.radians(cyl_ang)
        apex_vec = np.array([np.sin(ang_r), 0, np.cos(ang_r)])
        offset = feed + cyl_R * apex_vec
        Xc_t = Xc + feed[0]
        Yc_t = Yc + feed[1]
        Zc_t = Zc + feed[2]
        ax.plot_surface(Xc_t, Yc_t, Zc_t, color='gold', alpha=0.25,
                        rstride=2, cstride=2, edgecolor='orange', linewidth=0.3)
        ax.scatter(*offset, color='orange', s=25, marker='^', zorder=5)
        ax.plot([offset[0], feed[0]], [offset[1], feed[1]],
                [offset[2], feed[2]], 'orange', lw=0.5, alpha=0.4)

    # 6. 馈源标记
    ax.scatter(*feed, color='red', s=30, marker='x', zorder=6)
    ax.text(feed[0], 0, feed[2] + 0.01, "馈源", fontsize=7, color='red', ha='center')

    # 7. 尺寸标注
    ax.text(dish_a + 0.015, 0, 0, f"a={dish_a * 100:.0f}cm", fontsize=7, color='gray')
    ax.text(0, dish_b + 0.015, 0, f"b={dish_b * 100:.0f}cm", fontsize=7, color='gray')

    # 8. 坐标轴
    md = max(dish_a, dish_b, abs(feed[2]), abs(feed[0])) * 1.4
    z_min = -md * 0.2 if show_shield else -0.02
    ax.set_xlim(-md, md)
    ax.set_ylim(-md, md)
    ax.set_zlim(z_min, md * 1.1)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect([1, 1, 0.9])
    for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a_.pane.fill = False
        a_.pane.set_edgecolor('lightgray')


def render_overlay(ax, theta, phi, E_pat, dish_a, dish_b, dish_f,
                   offset_ang, show_shield, shield_D, shield_H,
                   cyl_R, cyl_len, cyl_ang, dip_len, feed_dx=0, feed_dz=0,
                   can_rot_x=0, can_rot_y=0,
                   show_dish=True, show_reflector=True):
    """结构与方向图叠加渲染"""
    f = dish_f
    feed = np.array([feed_dx, 0, f + feed_dz])

    if show_dish:
        Xd, Yd, Zd = dish_mesh(dish_a, dish_b, f,
                               offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
        ax.plot_wireframe(Xd, Yd, Zd, rstride=3, cstride=4,
                          color='gray', alpha=0.12, linewidth=0.3)
        xe, ye, ze = dish_edge(dish_a, dish_b, f,
                               offset_ang=offset_ang, feed_dx=feed_dx, feed_dz=feed_dz)
        ax.plot(xe, ye, ze, 'k-', lw=1, alpha=0.4)
    ax.scatter(*feed, color='red', s=20, marker='x', zorder=5)

    if show_reflector:
        Xc, Yc, Zc = semi_cylinder_mesh(cyl_R, cyl_len, cyl_ang)
        ax.plot_surface(Xc + feed[0], Yc + feed[1], Zc + feed[2],
                        color='gold', alpha=0.12,
                        rstride=2, cstride=2, edgecolor='orange', linewidth=0.2)

    # 屏蔽桶 (叠加视图)
    if show_shield:
        Xs, Ys, Zs = shield_mesh(dish_a, dish_b, f, offset_ang, feed_dx, feed_dz,
                                  shield_H, D=shield_D,
                                  rot_x=can_rot_x, rot_y=can_rot_y)
        ax.plot_surface(Xs, Ys, Zs, color='darkgray', alpha=0.06,
                        rstride=2, cstride=2, edgecolor='gray',
                        linewidth=0.2, linestyle='--')
        ax.plot(Xs[0, :], Ys[0, :], Zs[0, :], color='gray', lw=0.8, alpha=0.4)

    # 方向图叠加
    P = np.abs(E_pat)
    Pn = P / np.max(P) if np.max(P) > 0 else P
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    scale = 0.4
    R = Pn * scale
    Xg = R * np.sin(TH) * np.cos(PH) + feed[0]
    Yg = R * np.sin(TH) * np.sin(PH)
    Zg = R * np.cos(TH) + feed[2]

    P_dB = 10 * np.log10(np.clip(Pn, 1e-10, None))
    mask = P_dB >= -15

    if np.any(mask):
        Xm = np.where(mask, Xg, np.nan)
        Ym = np.where(mask, Yg, np.nan)
        Zm = np.where(mask, Zg, np.nan)
        colors = plt.cm.hot(Pn)
        ax.plot_surface(Xm, Ym, Zm, facecolors=colors,
                        rstride=2, cstride=2, alpha=0.5,
                        linewidth=0, antialiased=True)

    md = max(dish_a, dish_b, abs(feed[2]), abs(feed[0]), scale) * 1.3
    z_min = -md * 0.2 if show_shield else -0.02
    ax.set_xlim(-md, md)
    ax.set_ylim(-md, md)
    ax.set_zlim(z_min, md * 1.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=22, azim=-65)
    ax.set_box_aspect([1, 1, 0.9])
    for a_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
        a_.pane.fill = False
        a_.pane.set_edgecolor('lightgray')
