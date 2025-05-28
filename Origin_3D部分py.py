import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def draw_egg_3d(ax: plt.Axes, constants: dict) -> None:
    egg_x, egg_y, egg_z = _egg_position(constants)
    r = constants.get("gamete_r", 0.15)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = egg_x + r * np.outer(np.cos(u), np.sin(v))
    y = egg_y + r * np.outer(np.sin(u), np.sin(v))
    z = egg_z + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='yellow', alpha=0.6, edgecolor='none')


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
        ax.plot_surface(x, y, z, color="red", alpha=0.1, edgecolor="none")

    elif shape == "drop":
        R = constants.get("drop_r", 1.0)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi / 2, 30)
        x = R * np.outer(np.cos(u), np.sin(v))
        y = R * np.outer(np.sin(u), np.sin(v))
        z = R * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color="blue", alpha=0.1, edgecolor="none")

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


def plot_3d_movie_trajectories(trajs: np.ndarray, constants: dict, save_path: str = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 軸の範囲を constants から取得して固定（仕様通り）
    x_min, x_max = constants["x_min"], constants["x_max"]
    y_min, y_max = constants["y_min"], constants["y_max"]
    z_min, z_max = constants["z_min"], constants["z_max"]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Sperm Trajectories")

    # 背景（メディウム）と卵子を描画
    draw_motion_area_3d(ax, constants)
    draw_egg_3d(ax, constants)

    # 軌跡表示用オブジェクトを準備
    num_sperm = trajs.shape[0]
    lines = [ax.plot([], [], [], lw=1)[0] for _ in range(num_sperm)]

    def update(frame: int):
        for i, line in enumerate(lines):
            line.set_data(trajs[i, :frame + 1, 0], trajs[i, :frame + 1, 1])
            line.set_3d_properties(trajs[i, :frame + 1, 2])
        return lines

    ani = FuncAnimation(fig, update, frames=trajs.shape[1], interval=100, blit=False)

    if save_path:
        ani.save(save_path, fps=10)
    else:
        plt.show()
