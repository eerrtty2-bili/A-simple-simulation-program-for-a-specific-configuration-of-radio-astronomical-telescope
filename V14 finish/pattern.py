"""方向图计算: 偶极子 / 反射板 / 抛物面 / 屏蔽桶 PO 光线追踪"""

import numpy as np
import constants as C  # 引用 mutable 常量 (C.LAM, C.BETA)


# ======================== 方向图 ========================
def dipole_field(theta, length):
    kL2 = C.BETA * length / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        E = (np.cos(kL2 * np.cos(theta)) - np.cos(kL2)) / np.sin(theta)
    if np.any(l := np.isnan(E) | np.isinf(E)):
        E = np.where(l, 0.0, E)
    return E


def _bcast(r, phi):
    if np.ndim(r) == 0 and np.ndim(phi) > 0:
        return np.broadcast_to(r, np.shape(phi))
    return r


def _angle_from_y(t, p):
    """球坐标 (θ,φ) 中与 Y 轴的夹角"""
    return np.arccos(np.clip(np.sin(t) * np.sin(p), -1, 1))


def pattern_bare(t, p, L):
    theta_axis = _angle_from_y(t, p)
    return _bcast(dipole_field(theta_axis, L), p)


def pattern_reflector(t, p, L, d_ref=None):
    if d_ref is None:
        d_ref = C.LAM / 4
    theta_axis = _angle_from_y(t, p)
    E = dipole_field(theta_axis, L) * 2j * np.sin(C.BETA * d_ref * np.sin(t) * np.cos(p))
    x = np.sin(t) * np.cos(p)
    if np.ndim(x) == 0:
        if x < 0:
            E *= 0.1
    else:
        xa = np.asarray(x)
        ea = np.asarray(E, dtype=complex)
        w = (1 + np.tanh(xa / 0.12)) / 2
        ea = ea * (w + 0.02 * (1 - w))
        E = ea
    return E


def pattern_dish(t, p, L, D=None, fD=None):
    if D is None:
        D = 1.0
    if fD is None:
        fD = 0.4
    u = C.BETA * D / 2 * np.sin(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        E = 2 * bessel_j1(u) / u
        if np.ndim(E) == 0:
            if np.isnan(E) | np.isinf(E):
                E = 0.0
            elif np.abs(u) < 1e-10:
                E = 1.0
        else:
            E = np.asarray(E, dtype=float)
            E[np.isnan(E) | np.isinf(E)] = 0.0
            E = np.where(np.abs(u) < 1e-10, 1.0, E)
    return _bcast(E * np.sqrt(0.55), p)


# ======================== Bessel J1 (local copy) ========================
def bessel_j1(x):
    x = np.asarray(x, dtype=float)
    is_scalar = x.ndim == 0
    x = np.atleast_1d(x)
    r = np.zeros_like(x)
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
    if np.any(~m):
        xl = x[~m]
        r[~m] = np.sqrt(2 / (np.pi * xl)) * np.cos(xl - 3 * np.pi / 4)
    return r.item() if is_scalar else r


# ======================== 偏馈锅面几何 (PO 积分用) ========================
def _offset_dish_geometry(a, b, f, offset_deg, feed_dx, feed_dz, n_uv):
    """生成偏馈抛物面锅几何数据"""
    alpha = np.radians(offset_deg)
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

    u1 = np.linspace(-1, 1, n_uv)
    U, V = np.meshgrid(u1, u1)
    mask = U**2 + V**2 <= 1.0

    u_apt = U * a
    v_apt = V * b
    uv2 = u_apt**2 + v_apt**2
    w = uv2 / (4 * f) - f

    X = F[0] + u_apt * e1[0] + w * u_hat[0]
    Y = F[1] + v_apt * e2[1] + w * u_hat[1]
    Z = F[2] + u_apt * e1[2] + w * u_hat[2]

    z_off = np.min(Z[mask]) if np.any(mask) else 0.0
    Z -= z_off

    nx = u_hat[0] - (u_apt / (2 * f)) * e1[0]
    ny = u_hat[1] - (v_apt / (2 * f)) * e2[1]
    nz = u_hat[2] - (u_apt / (2 * f)) * e1[2]
    n_len = np.sqrt(nx**2 + ny**2 + nz**2)

    du = 2 * a / max(n_uv - 1, 1)
    dv = 2 * b / max(n_uv - 1, 1)
    dS = n_len * du * dv

    return dict(X=X, Y=Y, Z=Z,
                nx=nx / n_len, ny=ny / n_len, nz=nz / n_len,
                dS=dS, mask=mask,
                F=F, u_hat=u_hat, e1=e1, e2=e2, z_off=z_off)


# ======================== PO 光线追踪 ========================
def pattern_ray_trace(t, p, L, d_ref, dish_a, dish_b, dish_f,
                       feed_dx, feed_dz, shield_D, shield_H=0.0,
                       offset_ang=60, n_uv=18,
                       use_reflector=True, use_dish=True, use_shield=True):
    """物理光学 (PO) 光线追踪综合模型 (偏馈抛物面)"""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    _scalar = t.ndim == 0 and p.ndim == 0
    if _scalar:
        t = np.array([t])
        p = np.array([p])
    _1d = t.ndim == 1
    if _1d:
        t, p = np.meshgrid(t, np.atleast_1d(p), indexing='ij')

    # 无锅模式
    if not use_dish:
        if use_reflector:
            E = pattern_reflector(t, p, L, d_ref)
        else:
            E = pattern_bare(t, p, L)
        Pm = np.max(np.abs(E))
        if Pm > 0:
            E /= np.sqrt(Pm)
        if _scalar:
            return E[0, 0]
        if _1d:
            return E[:, 0] if E.shape[1] == 1 else E
        return E

    # 偏馈锅面几何
    G = _offset_dish_geometry(dish_a, dish_b, dish_f, offset_ang,
                               feed_dx, feed_dz, n_uv)
    X, Y, Z = G['X'], G['Y'], G['Z']
    F = G['F']
    mask = G['mask']

    # 馈源 → 锅面各点
    dX = X - F[0]
    dY = Y - F[1]
    dZ = Z - F[2]
    R_feed = np.sqrt(dX**2 + dY**2 + dZ**2)

    cos_theta_y = dY / R_feed
    theta_y = np.arccos(np.clip(cos_theta_y, -1, 1))
    E_dip = dipole_field(theta_y, L)

    cos_phi_x = dX / R_feed
    if use_reflector:
        E_ref = E_dip * (2j * np.sin(C.BETA * d_ref * cos_phi_x))
        w = 1.0 / (1.0 + np.exp(-cos_phi_x / 0.05))
        E_illum = E_ref * (w + 0.02 * (1 - w))
    else:
        E_illum = E_dip

    E_illum = E_illum / R_feed
    phase_inc = np.exp(-1j * C.BETA * R_feed)

    # PO 电流
    nx, ny, nz = G['nx'], G['ny'], G['nz']
    cos_inc = np.abs(dX * nx + dY * ny + dZ * nz) / R_feed
    J = 2.0 * E_illum * phase_inc * cos_inc
    J_all = J * G['dS'] * mask

    rim_angle = np.arctan2(shield_D / 2, dish_f + shield_H)

    # 远场积分
    st = np.sin(t)
    ct = np.cos(t)
    sp = np.sin(p)
    cp = np.cos(p)
    ux = st * cp
    uy = st * sp
    uz = ct

    J_flat = np.asarray(J_all * mask, dtype=complex).ravel()
    X_f = X.ravel()
    Y_f = Y.ravel()
    Z_f = Z.ravel()
    nz_idx = np.where(np.abs(J_flat) > 1e-30)[0]
    N_far = ux.size
    N_dish = n_uv * n_uv

    use_gpu = C._GPU_AVAIL and N_far > 5000 and N_dish > 200

    if use_gpu:
        import cupy as cp
        ux_g = cp.asarray(ux)
        uy_g = cp.asarray(uy)
        uz_g = cp.asarray(uz)
        X_g = cp.asarray(X_f)
        Y_g = cp.asarray(Y_f)
        Z_g = cp.asarray(Z_f)
        J_g = cp.asarray(J_flat)
        k_hat = cp.column_stack([ux_g.ravel(), uy_g.ravel(), uz_g.ravel()])
        r_vec = cp.column_stack([X_g, Y_g, Z_g])
        phase = cp.exp(1j * C.BETA * (k_hat @ r_vec.T))
        E_dish = (phase @ J_g).reshape(ux.shape)
    elif N_far * N_dish < 10_000_000:
        k_hat = np.column_stack([ux.ravel(), uy.ravel(), uz.ravel()])
        r_vec = np.column_stack([X_f, Y_f, Z_f])
        phase = np.exp(1j * C.BETA * (k_hat @ r_vec.T))
        E_dish = (phase @ J_flat).reshape(ux.shape)
    else:
        E_dish = np.zeros_like(ux, dtype=complex)
        for k in nz_idx:
            E_dish += J_flat[k] * np.exp(1j * C.BETA * (
                ux * X_f[k] + uy * Y_f[k] + uz * Z_f[k]))

    if use_gpu:
        E_dish = C._to_np(E_dish)

    # 溢失辐射 (spillover) — 桶高不足 λ/20 时等效于无桶
    if use_shield and shield_H > C.LAM / 20:
        E_spill = pattern_reflector(t, p, L, d_ref)
        sf = 1.0 / (1.0 + np.exp((t - rim_angle) / 0.04))
        E_spill *= sf * 0.15
        E_total = E_dish + E_spill
    else:
        E_total = E_dish

    Pm = np.max(np.abs(E_total))
    if Pm > 0:
        E_total /= np.sqrt(Pm)

    if _scalar:
        return E_total[0, 0]
    if _1d:
        return E_total[:, 0] if E_total.shape[1] == 1 else E_total
    return E_total
