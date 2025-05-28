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

# def draw_medium(ax, constants: dict):
#     shape = constants.get("shape", "cube").lower()

#     if shape == "spot":
#         R = constants.get("spot_r", 1.0)
#         z_base = constants.get("spot_bottom_height", 0.0)

#         # θ: 球の上部からの角度範囲を計算（z_base以上）
#         cos_theta_min = z_base / R
#         theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))  # θ ∈ [0, π]
#         theta = np.linspace(0, theta_min, 30)
#         phi = np.linspace(0, 2 * np.pi, 30)
#         theta, phi = np.meshgrid(theta, phi)

#         x = R * np.sin(theta) * np.cos(phi)
#         y = R * np.sin(theta) * np.sin(phi)
#         z = R * np.cos(theta)
#         ax.plot_surface(x, y, z, color="pink", alpha=0.1, edgecolor="none")

#     else:
#         # 通常の球体（drop や cube など）
#         R = constants.get("radius", 1.0)
#         u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
#         x = R * np.cos(u) * np.sin(v)
#         y = R * np.sin(u) * np.sin(v)
#         z = R * np.cos(v)
#         ax.plot_surface(x, y, z, color='pink', alpha=0.1, edgecolor='none')

# def draw_egg(ax, pos, radius):
#     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     x = radius * np.cos(u) * np.sin(v) + pos[0]
#     y = radius * np.sin(u) * np.sin(v) + pos[1]
#     z = radius * np.cos(v) + pos[2]
#     ax.plot_surface(x, y, z, color='red', alpha=0.3, edgecolor='none')
