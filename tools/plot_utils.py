import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from tools.derived_constants import get_limits  # スマートなimport

def _make_save_path(prefix="trajectory", ext="png", dir="Figs_and_Movies"):
    os.makedirs(dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dir}/{prefix}_{timestamp}.{ext}"

def _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_2d_trajectories(trajs, constants, save_path=None, show=True, max_sperm=None):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axis_configs = [
        (axs[0], (x_min, x_max), (y_min, y_max), "X", "Y", "XY-projection"),
        (axs[1], (x_min, x_max), (z_min, z_max), "X", "Z", "XZ-projection"),
        (axs[2], (y_min, y_max), (z_min, z_max), "Y", "Z", "YZ-projection"),
    ]

    for ax, xlim, ylim, xlabel, ylabel, title in axis_configs:
        _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel)
        ax.set_title(title)

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
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    n_sperm = min(traj.shape[0], max_sperm)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for s in range(n_sperm):
        ax.plot(traj[s, :, 0], traj[s, :, 1], traj[s, :, 2], label=f"Sperm {s}" if n_sperm <= 5 else None)

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
