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

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

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
        equal = shape in ('drop', 'cube', 'spot')
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
