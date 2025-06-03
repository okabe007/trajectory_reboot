
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools.plot_utils import draw_medium, draw_egg_3d
from tools.derived_constants import _egg_position
from datetime import datetime

def render_3d_movie(trajs: np.ndarray, vectors: np.ndarray, constants: dict,
                    save_path=None, show=True, format="mp4"):
    """
    3D軌跡＋ベクトルをアニメーションで表示・保存（MP4またはGIF対応）

    Parameters
    ----------
    trajs : np.ndarray
        (n_sperm, n_steps, 3) 軌跡配列
    vectors : np.ndarray
        (n_sperm, n_steps, 3) ベクトル配列
    constants : dict
        定数群（x_min, y_max, gamete_r など）
    save_path : str or None
        保存先ファイルパス（指定しなければ自動命名）
    show : bool
        表示するかどうか（Trueならplt.show()）
    format : str
        "mp4" または "gif"
    """
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
        ext = "gif" if format == "gif" else "mp4"
        save_path = os.path.join(base_dir, f"movie_{dtstr}.{ext}")

    try:
        if format == "gif":
            ani.save(save_path, writer="pillow", fps=10)
        else:
            ani.save(save_path, fps=10)  # ffmpeg
        print(f"[INFO] 動画を保存しました: {save_path}")
    except Exception as e:
        print(f"[ERROR] 保存失敗: {e}")

    if show:
        plt.show()
    plt.close(fig)
