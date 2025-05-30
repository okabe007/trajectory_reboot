import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tools.derived_constants import _egg_position
from tools.plot_utils import draw_egg
from tools.plot_utils import draw_medium


def render_3d_movie(trajs: np.ndarray, constants: dict, save_path=None, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # === 軸範囲設定（constantsから）===
    xlim = constants["x_min"], constants["x_max"]
    ylim = constants["y_min"], constants["y_max"]
    zlim = constants["z_min"], constants["z_max"]
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect([x_range, y_range, z_range])  # ✅ 絶対長さ比で軸を固定
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Sperm Trajectories")

    # === 背景（メディウム、卵子）を初期描画（1回だけ）===
    draw_medium(ax, constants)
    draw_egg(ax, _egg_position(constants), constants.get("gamete_r", 0.05))

    # === 軌跡表示オブジェクトを初期化 ===
    num_sperm = trajs.shape[0]
    num_frames = trajs.shape[1]
    lines = [ax.plot([], [], [], lw=1)[0] for _ in range(num_sperm)]

    # === アニメーション更新関数 ===
    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(trajs[i, :frame+1, 0], trajs[i, :frame+1, 1])
            line.set_3d_properties(trajs[i, :frame+1, 2])
        return lines

    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

    if save_path:
        ani.save(save_path, fps=10)
    elif show:
        plt.show()

