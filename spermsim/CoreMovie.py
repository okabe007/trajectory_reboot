import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from tools.derived_constants import _egg_position
from tools.plot_utils import draw_medium, draw_egg_3d


def render_3d_movie(trajs: np.ndarray, vectors: np.ndarray, constants: dict, save_path=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 軸設定
    xlim = constants["x_min"], constants["x_max"]
    ylim = constants["y_min"], constants["y_max"]
    zlim = constants["z_min"], constants["z_max"]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        zlim[1] - zlim[0]
    ])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Sperm Vectors (Fixed Length)")

    # 卵子・メディウム
    draw_medium(ax, constants)
    egg_pos = _egg_position(constants)
    egg_radius = constants.get("gamete_r", 0.05)
    draw_egg_3d(ax, egg_pos, egg_radius)

    num_sperm = trajs.shape[0]
    num_frames = trajs.shape[1]

    # 色（赤抜き19色）
    full_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    red_index = 3
    colors = [full_colors[i] for i in range(20) if i != red_index]

    # 初期矢印
    quivers = [
        ax.quiver(0, 0, 0, 0, 0, 0, length=0.1, normalize=True,
                  arrow_length_ratio=0.9, linewidth=2.5, color=colors[i % 19])
        for i in range(num_sperm)
    ]

    def update(frame):
        for i in range(num_sperm):
            x, y, z = trajs[i, frame]
            u, v, w = vectors[i, frame]
            quivers[i].remove()
            quivers[i] = ax.quiver(
                x, y, z, u, v, w,
                length=0.1,             # ✅ 常に0.1mmで固定
                normalize=True,         # ✅ 単位ベクトル化
                arrow_length_ratio=0.7, # ✅ 太く見せる
                linewidth=2,          # ✅ 軸を太く（環境依存）
                color=colors[i % 19]
            )
        return quivers

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    if save_path:
        ani.save(save_path, fps=10)
    elif show:
        plt.show()
