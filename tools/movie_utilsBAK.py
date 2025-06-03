
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools.plot_utils import draw_medium, draw_egg_3d
from tools.derived_constants import _egg_position
from datetime import datetime

def render_3d_movie(trajs: np.ndarray, vectors: np.ndarray, constants: dict, save_path=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xlim = constants["x_min"], constants["x_max"]
    ylim = constants["y_min"], constants["y_max"]
    zlim = constants["z_min"], constants["z_max"]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Sperm Vectors (Fixed Length)")

    draw_medium(ax, constants)
    egg_pos = _egg_position(constants)
    egg_radius = constants.get("gamete_r", 0.05)
    draw_egg_3d(ax, egg_pos, egg_radius)

    num_sperm = trajs.shape[0]
    num_frames = trajs.shape[1]

    full_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = [full_colors[i] for i in range(20) if i != 3]

    quivers = [
        ax.quiver(0, 0, 0, 0, 0, 0, length=0.1, normalize=True,
                  arrow_length_ratio=0.9, linewidth=2.5, color=colors[i % len(colors)])
        for i in range(num_sperm)
    ]

    def update(frame):
        for i in range(num_sperm):
            x, y, z = trajs[i, frame]
            u, v, w = vectors[i, frame]
            quivers[i].remove()
            quivers[i] = ax.quiver(
                x, y, z, u, v, w,
                length=0.1,
                normalize=True,
                arrow_length_ratio=0.7,
                linewidth=2,
                color=colors[i % len(colors)]
            )
        return quivers

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    if not save_path:
        base_dir = os.path.join(os.path.dirname(__file__), "figs_and_movies")
        os.makedirs(base_dir, exist_ok=True)
        dtstr = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(base_dir, f"movie_{dtstr}.mp4")

    try:
        ani.save(save_path, fps=10)
        print(f"[INFO] 動画を保存しました: {save_path}")
    except Exception as e:
        print(f"[WARN] ffmpeg保存失敗 ({e}) → pillow writerで再試行")
        try:
            ani.save(save_path, writer='pillow', fps=10)
            print(f"[INFO] pillow writerで動画を保存しました: {save_path}")
        except Exception as e2:
            print(f"[ERROR] pillow writerでも保存失敗: {e2}")
            return

    if show:
        plt.show()

    plt.close(fig)
