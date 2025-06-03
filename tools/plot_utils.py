
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tools.derived_constants import get_limits

def plot_2d_trajectories(trajs_um, constants, save_path=None, show=True, max_sperm=None):
    """
    ● 前提：
        - trajs_um : μm 単位 (np.ndarray) の精子軌跡データ
          → 関数内で一度だけ μm→mm に変換します。
        - constants は calculate_derived_constants(raw_constants) が返した辞書で、
          以下のキーがすべて“mm 単位”になっています：
            * constants["shape"]
            * constants["drop_r"]
            * constants["spot_r"], constants["spot_bottom_r"], constants["spot_bottom_height"]
            * constants["gamete_r"], constants["egg_center"]（array([x_mm, y_mm, z_mm])）
            * get_limits(constants) が返す (x_min, x_max, y_min, y_max, z_min, z_max)
    """

    # ① get_limits で “mm 単位” の描画範囲を取得
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)

    # ② trajs_um が μm 単位なら mm 単位へ変換
    trajs_mm = trajs_um.astype(float) / 1000.0

    # ③ サブプロット作成 (XY, XZ, YZ)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # ④ 背景 (drop or spot) を mm 単位で描画
    shape = constants.get('shape', 'cube').lower()

    if shape == "drop":
        r = constants["drop_r"]  # mm
        axs[0].add_patch(Circle((0.0, 0.0), r, color="pink", alpha=0.3))  # XY
        axs[1].add_patch(Circle((0.0, 0.0), r, color="pink", alpha=0.3))  # XZ
        axs[2].add_patch(Circle((0.0, 0.0), r, color="pink", alpha=0.3))  # YZ

    elif shape == "spot":
        R   = constants["spot_r"]              # mm
        b_r = constants["spot_bottom_r"]       # mm
        b_h = constants["spot_bottom_height"]  # mm

        # XY: 底面円 (mm)
        axs[0].add_patch(Circle((0.0, 0.0), b_r, color="pink", alpha=0.3))

        # XZ／YZ: 球面断面 (mm)
        x_vals = np.linspace(-b_r, b_r, 200)
        z_vals = np.sqrt(np.clip(R**2 - x_vals**2, 0.0, None))
        axs[1].fill_between(x_vals, b_h, z_vals, color="pink", alpha=0.3)  # XZ
        axs[2].fill_between(x_vals, b_h, z_vals, color="pink", alpha=0.3)  # YZ

    else:
        # cube などその他形状は背景を描かない
        pass

    # ⑤ 卵子を mm 単位で描画
    egg_center = constants["egg_center"]  # array([x_mm, y_mm, z_mm])
    egg_r      = constants["gamete_r"]   # mm

    axs[0].add_patch(
        Circle(
            (egg_center[0], egg_center[1]),  # XY 中心 (mm)
            egg_r,
            facecolor="yellow", alpha=0.6,
            edgecolor="gray", linewidth=1.0
        )
    )
    axs[1].add_patch(
        Circle(
            (egg_center[0], egg_center[2]),  # XZ 中心 (mm)
            egg_r,
            facecolor="yellow", alpha=0.6,
            edgecolor="gray", linewidth=1.0
        )
    )
    axs[2].add_patch(
        Circle(
            (egg_center[1], egg_center[2]),  # YZ 中心 (mm)
            egg_r,
            facecolor="yellow", alpha=0.6,
            edgecolor="gray", linewidth=1.0
        )
    )

    # ⑥ 精子軌跡を mm 単位で描画
    if max_sperm is None:
        max_sperm = trajs_mm.shape[0]

    for s in range(min(max_sperm, trajs_mm.shape[0])):
        axs[0].plot(trajs_mm[s, :, 0], trajs_mm[s, :, 1], linewidth=0.6)  # XY
        axs[1].plot(trajs_mm[s, :, 0], trajs_mm[s, :, 2], linewidth=0.6)  # XZ
        axs[2].plot(trajs_mm[s, :, 1], trajs_mm[s, :, 2], linewidth=0.6)  # YZ

    # ⑦ 軸範囲とラベルを mm 単位で設定
    axs[0].set_title("XY (mm)")
    axs[1].set_title("XZ (mm)")
    axs[2].set_title("YZ (mm)")

    _set_common_2d_ax(
        axs[0],
        (x_min, x_max),
        (y_min, y_max),
        "X (mm)",
        "Y (mm)"
    )
    _set_common_2d_ax(
        axs[1],
        (x_min, x_max),
        (z_min, z_max),
        "X (mm)",
        "Z (mm)"
    )
    _set_common_2d_ax(
        axs[2],
        (y_min, y_max),
        (z_min, z_max),
        "Y (mm)",
        "Z (mm)"
    )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()


def _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel):
    """
    軸範囲とラベルを設定するヘルパー関数
    xlim   : (xmin, xmax) in mm
    ylim   : (ymin, ymax) in mm
    xlabel : 例 "X (mm)"
    ylabel : 例 "Y (mm)"
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)



def _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel):
    """
    軸範囲とラベルを設定するヘルパー関数
    xlim   : (xmin, xmax) in mm
    ylim   : (ymin, ymax) in mm
    xlabel : 例 "X (mm)"
    ylabel : 例 "Y (mm)"
    """
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)



def draw_medium(ax, constants: dict):
    shape = constants.get("shape", "cube").lower()

    if shape == "spot":
        R = constants.get("spot_r", 1.0)
        z_base = constants.get("spot_bottom_height", 0.0)

        # 球冠: z ≥ z_base → cos(θ) ≤ z_base/R → θ ≥ acos(z_base/R)
        cos_theta_min = z_base / R
        theta_min = np.arccos(np.clip(cos_theta_min, -1.0, 1.0))
        theta = np.linspace(0, theta_min, 30)
        phi = np.linspace(0, 2 * np.pi, 30)
        theta, phi = np.meshgrid(theta, phi)
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")

    elif shape == "drop":
        R = constants.get("drop_r", 1.0)
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]  # ✅ 球体全体を描く
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_surface(x, y, z, color="pink", alpha=0.3, edgecolor="none")

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
            ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], color="gray", alpha=0.5)


def _set_common_2d_ax(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

def draw_egg_3d(ax, egg_pos, radius, *, color="yellow", alpha=0.6):
    """3Dの卵子球を描画する関数（egg_pos, radius を外部から渡す構造）"""
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + egg_pos[0]
    y = radius * np.sin(u) * np.sin(v) + egg_pos[1]
    z = radius * np.cos(v) + egg_pos[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none")



def plot_3d_movie_trajectories(trajs: np.ndarray, constants: dict, save_path=None, show=True, intersection_records=None):
    """3D軌跡ムービーの作成（卵子とメディウムも毎フレーム描画）"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xlim, ylim, zlim = constants["xlim"], constants["ylim"], constants["zlim"]

    def set_ax_3D(ax):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Sperm Trajectories")
        ax.set_box_aspect([1, 1, 1])  # ← ★ これが超重要（軸スケールを固定）

    

    def update(frame):
        ax.cla()  # ← これがある限り、再描画が必須！
        set_ax_3D(ax)
        draw_medium(ax, medium_radius)
        draw_egg(ax, egg_pos, egg_radius)
        for s in range(num_sperm):
            ax.plot(trajs[s, :frame+1, 0],
                    trajs[s, :frame+1, 1],
                    trajs[s, :frame+1, 2],
                    lw=0.7)

    draw_egg(ax, pos, radius)
    
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100)

    if save_path:
        ani.save(save_path, writer='ffmpeg')
    if show:
        plt.show()
