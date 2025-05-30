import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib import patches
from tools.derived_constants import get_limits  # スマートなimport

def _make_save_path(prefix="trajectory", ext="png", dir="Figs_and_Movies"):
    os.makedirs(dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dir}/{prefix}_{timestamp}.{ext}"

def _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel, equal=False):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal:
        ax.set_aspect('equal', adjustable='box')

def plot_2d_trajectories(trajs, constants, save_path=None, show=True, max_sperm=None):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    shape = str(constants.get('shape', '')).lower()
    drop_r = float(constants.get('drop_r', 0.0))
    spot_r = float(constants.get('spot_r', 0.0))
    spot_bottom_height = float(constants.get('spot_bottom_height', 0.0))
    spot_bottom_r = float(constants.get('spot_bottom_r', spot_r))

    if shape == 'spot' and spot_r > 0:
        axis_configs = [
            (axs[0], (-spot_r, spot_r), (-spot_r, spot_r), "X", "Y", "XY-projection"),
            (axs[1], (-spot_r, spot_r), (spot_bottom_height, spot_r), "X", "Z", "XZ-projection"),
            (axs[2], (-spot_r, spot_r), (spot_bottom_height, spot_r), "Y", "Z", "YZ-projection"),
        ]
    else:
        axis_configs = [
            (axs[0], (x_min, x_max), (y_min, y_max), "X", "Y", "XY-projection"),
            (axs[1], (x_min, x_max), (z_min, z_max), "X", "Z", "XZ-projection"),
            (axs[2], (y_min, y_max), (z_min, z_max), "Y", "Z", "YZ-projection"),
        ]


    for ax, xlim, ylim, xlabel, ylabel, title in axis_configs:
<<<<<<< ours
<<<<<<< ours
        equal = shape in ('drop', 'cube')
=======
        equal = shape in ('drop', 'cube', 'spot')
>>>>>>> theirs
=======
        equal = shape in ('drop', 'cube', 'spot')
>>>>>>> theirs
        _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel, equal)
        ax.set_title(title)
        if shape == 'drop' and drop_r > 0:
            ax.add_patch(
                patches.Circle((0, 0), drop_r, ec='none', facecolor='red', alpha=0.1)
            )
        elif shape == 'cube':
            width = xlim[1] - xlim[0]
            height = ylim[1] - ylim[0]
            ax.add_patch(
                patches.Rectangle((xlim[0], ylim[0]), width, height,
                                  ec='none', facecolor='red', alpha=0.1)
            )
<<<<<<< ours
<<<<<<< ours
=======
=======
>>>>>>> theirs
        elif shape == 'spot' and spot_r > 0:
            if xlabel == 'X' and ylabel == 'Y':
                ax.add_patch(
                    patches.Circle((0, 0), spot_bottom_r, ec='none', facecolor='red', alpha=0.1)
                )
            else:
                ax.add_patch(
                    patches.Circle((0, 0), spot_r, ec='none', facecolor='red', alpha=0.1)
                )
                ax.axhline(spot_bottom_height, color='gray', linestyle='--', linewidth=0.8)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs

    n_sperm = min(trajs.shape[0], max_sperm or trajs.shape[0])
    for s in range(n_sperm):
        axs[0].plot(trajs[s, :, 0], trajs[s, :, 1])
        axs[1].plot(trajs[s, :, 0], trajs[s, :, 2])
        axs[2].plot(trajs[s, :, 1], trajs[s, :, 2])

    fig.suptitle(
        f"shape={constants.get('shape')}, vol={constants.get('volume')}, "
        f"sperm_conc={constants.get('sperm_conc')}, vsl={constants.get('vsl')}, "
        f"sim_min={constants.get('sim_min')}, sim_repeat={constants.get('sim_repeat')}",
        fontsize=10
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if not save_path:
        save_path = _make_save_path("trajectory_2d", "png")
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] 2D図を保存しました: {save_path}")
    if show:
        plt.show()

def plot_3d_trajectories(traj: np.ndarray, constants: dict, max_sperm: int = 5, show=True):
    shape = str(constants.get('shape', '')).lower()

    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    if shape == 'spot':
        spot_r = float(constants.get('spot_r', 0.0))
        spot_bottom_height = float(constants.get('spot_bottom_height', 0.0))
        if spot_r > 0:
            x_min, x_max = -spot_r, spot_r
            y_min, y_max = -spot_r, spot_r
            z_min, z_max = spot_bottom_height, spot_r

    n_sperm = min(traj.shape[0], max_sperm)

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')
    for s in range(n_sperm):
        ax.plot(traj[s, :, 0], traj[s, :, 1], traj[s, :, 2], label=f"Sperm {s}" if n_sperm <= 5 else None)

    drop_r = float(constants.get('drop_r', 0.0))
    if shape == 'drop' and drop_r > 0:
        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        sx = drop_r * np.outer(np.sin(v), np.cos(u))
        sy = drop_r * np.outer(np.sin(v), np.sin(u))
        sz = drop_r * np.outer(np.cos(v), np.ones_like(u))
        ax.plot_surface(sx, sy, sz, color='red', alpha=0.1)
    elif shape == 'cube':
        xs = [x_min, x_max]
        ys = [y_min, y_max]
        zs = [z_min, z_max]
        # 6 faces of the cube
        for z in zs:
            X, Y = np.meshgrid(xs, ys)
            Z = np.full_like(X, z)
            ax.plot_surface(X, Y, Z, color='red', alpha=0.1)
        for x in xs:
            Y, Z = np.meshgrid(ys, zs)
            X = np.full_like(Y, x)
            ax.plot_surface(X, Y, Z, color='red', alpha=0.1)
        for y in ys:
            X, Z = np.meshgrid(xs, zs)
            Y = np.full_like(X, y)
            ax.plot_surface(X, Y, Z, color='red', alpha=0.1)
<<<<<<< ours
<<<<<<< ours
=======
=======
>>>>>>> theirs
    elif shape == 'spot':
        spot_r = float(constants.get('spot_r', 0.0))
        spot_angle = float(constants.get('spot_angle', 0.0))
        if spot_r > 0 and spot_angle > 0:
            u = np.linspace(0, 2 * np.pi, 60)
            v = np.linspace(0, np.deg2rad(spot_angle), 60)
            sx = spot_r * np.outer(np.sin(v), np.cos(u))
            sy = spot_r * np.outer(np.sin(v), np.sin(u))
            sz = spot_r * np.outer(np.cos(v), np.ones_like(u))
            ax.plot_surface(sx, sy, sz, color='red', alpha=0.1)
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ])
    ax.set_title("3D Trajectory")
    if n_sperm <= 5:
        ax.legend()

    fig.suptitle(
        f"shape={constants.get('shape')}, vol={constants.get('volume')}, "
        f"sperm_conc={constants.get('sperm_conc')}, vsl={constants.get('vsl')}, "
        f"sim_min={constants.get('sim_min')}, sim_repeat={constants.get('sim_repeat')}",
        fontsize=10
    )
    save_path = _make_save_path("trajectory_3d", "png")
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] 3D図を保存しました: {save_path}")
    if show:
        plt.show()
