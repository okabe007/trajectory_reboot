import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from tools.derived_constants import get_limits
from tools.derived_constants import _egg_position  # 必要ならtoolsに移動しておく
import sys
print("[DEBUG] sys.path =", sys.path)
# =======================
# 🔧 共通ヘルパー関数
# =======================
def _make_save_path(prefix="trajectory", ext="png", dir="Figs_and_Movies"):
    os.makedirs(dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dir}/{prefix}_{timestamp}.{ext}"
def draw_medium(ax, constants: dict):
    shape = constants.get("shape", "cube").lower()

    if shape == "spot":
        R = constants.get("spot_r", 1.0)
        z_base = constants.get("spot_bottom_height", 0.0)

        # 球冠: z ≥ z_base → cos(θ) ≤ z_base/R → θ ≥ acos(z_base/R)
        cos_theta_min = z_base / R
        theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))
        theta = np.linspace(0, theta_min, 30)
        phi = np.linspace(0, 2 * np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")

    elif shape == "drop":
        R = constants.get("drop_r", 1.0)
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]  # ✅ 球体全体を描く
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")

    elif shape == "cube":
        x_min, x_max = constants["x_min"], constants["x_max"]
        y_min, y_max = constants["y_min"], constants["y_max"]
        z_min, z_max = constants["z_min"], constants["z_max"]
        for s, e in zip(
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min),
             (x_min, y_max, z_min), (x_min, y_min, z_max), (x_max, y_min, z_max),
             (x_max, y_max, z_max), (x_min, y_max, z_max)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),
             (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max),
             (x_min, y_max, z_max), (x_min, y_min, z_max)]
        ):
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color="gray", alpha=0.5)

def _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')  # ✅ 数値の増分 = 実長さに一致

def draw_egg(ax, pos, radius, *, color="yellow", alpha=0.6):
    """Draw the egg as a sphere.

    Parameters
    ----------
    ax : matplotlib axes
        3D axes object to draw on.
    pos : tuple[float, float, float]
        (x, y, z) position of the egg centre.
    radius : float
        Radius of the egg.
    color : str, optional
        Surface colour (default ``"yellow"``).
    alpha : float, optional
        Surface transparency (default ``0.6``).
    """

    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = radius * np.cos(u) * np.sin(v) + pos[0]
    y = radius * np.sin(u) * np.sin(v) + pos[1]
    z = radius * np.cos(v) + pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")
# =======================
# 🟨 2D軌跡プロット
# =======================
from matplotlib.patches import Circle

def plot_2d_trajectories(trajs, constants, save_path=None, show=True, max_sperm=None):
    from tools.derived_constants import get_limits, _egg_position

    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    shape = constants.get('shape', 'cube').lower()
    drop_r = float(constants.get('drop_r', 0.0))
    spot_r = float(constants.get('spot_r', 0.0))
    egg_r = float(constants.get("gamete_r", 0.05))
    egg_pos = _egg_position(constants)

    if max_sperm is None:
        max_sperm = trajs.shape[0]

    # 軌跡描画
    for s in range(min(max_sperm, trajs.shape[0])):
        axs[0].plot(trajs[s, :, 0], trajs[s, :, 1], linewidth=0.6)  # XY
        axs[1].plot(trajs[s, :, 0], trajs[s, :, 2], linewidth=0.6)  # XZ
        axs[2].plot(trajs[s, :, 1], trajs[s, :, 2], linewidth=0.6)  # YZ


    # 背景（メディウム）描画
    if shape == "drop":
        medium_r = drop_r
        center = (0.0, 0.0)
        axs[0].add_patch(Circle(center, medium_r, color="pink", alpha=0.3))  # XY
        axs[1].add_patch(Circle((0, 0), medium_r, color="pink", alpha=0.3))  # XZ
        axs[2].add_patch(Circle((0, 0), medium_r, color="pink", alpha=0.3))  # YZ

    elif shape == "spot":
        R = spot_r
        bottom_r = constants.get("spot_bottom_r", R)
        bottom_h = constants.get("spot_bottom_height", 0.0)

        # XY 平面: 底面半径で塗りつぶす
        axs[0].add_patch(Circle((0, 0), bottom_r, color="pink", alpha=0.3))

        # XZ / YZ 平面: 球面断面を塗りつぶし
        x = np.linspace(-bottom_r, bottom_r, 200)
        z = np.sqrt(np.clip(R ** 2 - x**2, 0.0, None))
        axs[1].fill_between(x, bottom_h, z, color="pink", alpha=0.3)
        axs[2].fill_between(x, bottom_h, z, color="pink", alpha=0.3)

    # 卵子描画（円）
    egg_kw = dict(color="yellow", alpha=0.6, edgecolor="gray", linewidth=0.5)
    axs[0].add_patch(Circle(
        (egg_pos[0], egg_pos[1]), egg_r,
        facecolor="yellow", alpha=0.6,
        edgecolor="gray", linewidth=1.0
    ))

    axs[1].add_patch(Circle(
        (egg_pos[0], egg_pos[2]), egg_r,
        facecolor="yellow", alpha=0.6,
        edgecolor="gray", linewidth=1.0
    ))

    axs[2].add_patch(Circle(
        (egg_pos[1], egg_pos[2]), egg_r,
        facecolor="yellow", alpha=0.6,
        edgecolor="gray", linewidth=1.0
    ))

    # 軸とアスペクト比
    axs[0].set_title("XY")
    axs[1].set_title("XZ")
    axs[2].set_title("YZ")

    _set_common_2d_ax(axs[0], (x_min, x_max), (y_min, y_max), "X", "Y")
    _set_common_2d_ax(axs[1], (x_min, x_max), (z_min, z_max), "X", "Z")
    _set_common_2d_ax(axs[2], (y_min, y_max), (z_min, z_max), "Y", "Z")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()


# =======================
# 🟨 3D用の補助描画関数
# =======================
def draw_egg_3d(ax, egg_pos, radius, *, color="yellow", alpha=0.6):
    """3Dの卵子球を描画する関数（egg_pos, radius を外部から渡す構造）"""
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + egg_pos[0]
    y = radius * np.sin(u) * np.sin(v) + egg_pos[1]
    z = radius * np.cos(v) + egg_pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")




def draw_motion_area_3d(ax: plt.Axes, constants: dict) -> None:
    shape = constants.get("shape", "cube").lower()

    if shape == "spot":
        R = constants.get("spot_R", 1.0)
        angle_deg = constants.get("spot_angle", 60.0)
        theta = np.linspace(0, np.deg2rad(angle_deg), 30)
        phi = np.linspace(0, 2 * np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        ax.plot_surface(x, y, z, color="red", alpha=0.3, edgecolor="none")

    elif shape == "drop":
        R = constants.get("drop_r", 1.0)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi / 2, 30)
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color="blue", alpha=0.3, edgecolor="none")

    elif shape == "cube":
        x_min, x_max = constants["x_min"], constants["x_max"]
        y_min, y_max = constants["y_min"], constants["y_max"]
        z_min, z_max = constants["z_min"], constants["z_max"]
        for s, e in zip(
            [(x_min, y_min, z_min), (x_max, y_min, z_min), (x_max, y_max, z_min),
             (x_min, y_max, z_min), (x_min, y_min, z_max), (x_max, y_min, z_max),
             (x_max, y_max, z_max), (x_min, y_max, z_max)],
            [(x_max, y_min, z_min), (x_max, y_max, z_min), (x_min, y_max, z_min),
             (x_min, y_min, z_max), (x_max, y_min, z_max), (x_max, y_max, z_max),
             (x_min, y_max, z_max), (x_min, y_min, z_max)]
        ):
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color="gray", alpha=0.2)


# =======================
# 🟨 3Dムービー描画
# =======================
from tools.derived_constants import _egg_position

def plot_3d_movie_trajectories(trajs: np.ndarray, constants: dict, save_path=None, show=True, intersection_records=None):
    """3D軌跡ムービーの作成（卵子とメディウムも毎フレーム描画）"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xlim, ylim, zlim = constants["xlim"], constants["ylim"], constants["zlim"]

    def set_ax_3D(ax):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Sperm Trajectories")
        ax.set_box_aspect([1, 1, 1])  # ← ★ これが超重要（軸スケールを固定）

    

    def update(frame):
        ax.cla()  # ← これがある限り、再描画が必須！
        set_ax_3D(ax)
        draw_medium(ax, medium_radius)
        draw_egg(ax, egg_pos, egg_radius)
        for s in range(num_sperm):
            ax.plot(trajs[s, :frame+1, 0],
                    trajs[s, :frame+1, 1],
                    trajs[s, :frame+1, 2],
                    lw=0.7)

    draw_egg(ax, pos, radius)
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100)

    if save_path:
        ani.save(save_path, writer='ffmpeg')
    if show:
        plt.show()


############
def plot_trajectories(trajectories, constants, max_sperm=5, save_path=None):
    """
    軌跡データ（精子×ステップ×3）をXY/XZ/YZで可視化
    """
    if trajectories is None or len(trajectories) == 0:
        print("[WARNING] 軌跡データがありません。")
        return

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    all_mins = [constants["x_min"], constants["y_min"], constants["z_min"]]
    all_maxs = [constants["x_max"], constants["y_max"], constants["z_max"]]
    global_min = min(all_mins)
    global_max = max(all_maxs)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    ax_xy, ax_xz, ax_yz = axes
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    n_sperm = min(len(trajectories), max_sperm)

    from tools.derived_constants import _egg_position
    egg_x, egg_y, egg_z = _egg_position(constants)
    for ax, (x, y) in zip(axes, [(egg_x, egg_y), (egg_x, egg_z), (egg_y, egg_z)]):
        ax.add_patch(Circle((x, y), radius=constants.get("gamete_r", 0),
                            facecolor="yellow", alpha=0.8, ec="gray", linewidth=0.5))

    for i in range(n_sperm):
        ax_xy.plot(trajectories[i][:,0], trajectories[i][:,1], color=colors[i % len(colors)])
        ax_xz.plot(trajectories[i][:,0], trajectories[i][:,2], color=colors[i % len(colors)])
        ax_yz.plot(trajectories[i][:,1], trajectories[i][:,2], color=colors[i % len(colors)])

    for ax, (label_x, label_y) in zip(axes, [("X", "Y"), ("X", "Z"), ("Y", "Z")]):
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.set_aspect("equal")
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)

    param_summary = ', '.join(f"{k}={constants.get(k)}" for k in [
        "shape", "vol", "sperm_conc", "vsl", "deviation"
    ])
    param_summary2 = ', '.join(f"{k}={constants.get(k)}" for k in [
        "surface_time", "egg_localization", "gamete_r", "sim_min", "sample_rate_hz", "sim_repeat"
    ])
    fig.suptitle(f"{param_summary}\n{param_summary2}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] 保存しました: {save_path}")
    else:
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_2d_trajectories(trajectories, constants):
    fig, ax = plt.subplots()

    for trajectory in trajectories:
        valid = ~np.isnan(trajectory[:, 0])
        if valid.any():
            ax.plot(trajectory[valid, 0], trajectory[valid, 1], '-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
