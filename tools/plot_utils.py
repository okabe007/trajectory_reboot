import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib import patches
from tools.derived_constants import get_limits  # スマートなimport
import math

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


def _egg_position(constants):
    """Return (egg_x, egg_y, egg_z) according to shape and localization."""
    shape = str(constants.get("shape", "cube")).lower()
    loc = constants.get("egg_localization", "center")
    r = constants.get("gamete_r", 0)

    if shape == "cube":
        positions = {
            "center": (0, 0, 0),
            "bottom_center": (0, 0, constants.get("z_min", 0) + r),
            "bottom_side": (
                constants.get("x_min", 0) / 2 + r,
                constants.get("y_min", 0) / 2 + r,
                constants.get("z_min", 0) + r,
            ),
            "bottom_corner": (
                constants.get("x_min", 0) + r,
                constants.get("y_min", 0) + r,
                constants.get("z_min", 0) + r,
            ),
        }
    elif shape == "drop":
        drop_r = constants.get("drop_r", 0)
        positions = {
            "center": (0, 0, 0),
            "bottom_center": (0, 0, -drop_r + r),
        }
    elif shape == "spot":
        spot_r = constants.get("spot_r", 0)
        spot_bottom_height = constants.get("spot_bottom_height", 0)
        positions = {
            "center": (0, 0, (spot_bottom_height + spot_r) / 2),
            "bottom_center": (0, 0, spot_bottom_height + r),
            "bottom_edge": (
                math.sqrt(max((spot_r - r) ** 2 - (spot_bottom_height + r) ** 2, 0)),
                0,
                spot_bottom_height + r,
            ),
        }
    elif shape == "ceros":
        cx = (constants.get("x_min", 0) + constants.get("x_max", 0)) / 2
        cy = (constants.get("y_min", 0) + constants.get("y_max", 0)) / 2
        cz = (constants.get("z_min", 0) + constants.get("z_max", 0)) / 2
        positions = {"center": (cx, cy, cz), "bottom_center": (cx, cy, cz), "bottom_edge": (cx, cy, cz)}
    else:
        raise RuntimeError(f"Unknown shape '{shape}'")

    if loc not in positions:
        raise RuntimeError(f"Invalid egg_localization '{loc}' for shape '{shape}'")

    return positions[loc]

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
            (axs[0], (-spot_bottom_r, spot_bottom_r), (-spot_bottom_r, spot_bottom_r),
             "X", "Y", "XY-projection"),
            (axs[1], (-spot_bottom_r, spot_bottom_r), (spot_bottom_height, spot_r),
             "X", "Z", "XZ-projection"),
            (axs[2], (-spot_bottom_r, spot_bottom_r), (spot_bottom_height, spot_r),
             "Y", "Z", "YZ-projection"),
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

    if shape != 'ceros':
        egg_x, egg_y, egg_z = _egg_position(constants)
        for ax, (x, y) in zip(axs, [(egg_x, egg_y), (egg_x, egg_z), (egg_y, egg_z)]):
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    radius=constants.get('gamete_r', 0),
                    facecolor='yellow',
                    alpha=0.8,
                    ec='gray',
                    linewidth=1.0,
                )
            )

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
