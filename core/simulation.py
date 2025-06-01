from io_status import IOStatus
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
import math
import os
from matplotlib.animation import FuncAnimation
from typing import Tuple
from numpy import linalg as LA


from io_status import IOStatus

from tools.derived_constants import _egg_position, calculate_derived_constants
from tools.plot_utils import plot_2d_trajectories, plot_3d_movie_trajectories

from tools.geometry import CubeShape, DropShape, SpotShape, CerosShape, _handle_drop_outside
from core.simulation_core import SpermSimulation
from core.simulation_core import SpotIO, _spot_status_check
from tools.io_checks import IO_check_spot
from tools.io_checks import IO_check_drop
from tools.io_checks import IO_check_cube
# def IO_check_spot(base_position: np.ndarray, temp_position: np.ndarray, constants: dict, IO_status: str) -> str:
#     radius   = constants['radius']
#     bottom_z = constants['spot_bottom_height']
#     bottom_R = constants['spot_bottom_r']  # ← 小文字に注意
#     limit    = constants['limit']

#     z_tip = temp_position[2]
#     r_tip = LA.norm(temp_position)
#     xy_dist = np.sqrt(temp_position[0]**2 + temp_position[1]**2)

#     if z_tip > bottom_z + limit:
#         if r_tip > radius + limit:
#             return "sphere_out"







#     INSIDE = "inside"
#     BORDER = "border"
#     SPHERE_OUT = "sphere_out"
#     BOTTOM_OUT = "bottom_out"
#     SPOT_EDGE_OUT = "spot_edge_out"
#     POLYGON_MODE = "polygon_mode"
#     SPOT_BOTTOM = "spot_bottom"


#     radius = constants.get("spot_r", constants.get("radius", 0.0))
#     bottom_z = constants.get("spot_bottom_height", 0.0)
#     bottom_r = constants.get("spot_bottom_r", 0.0)
#     limit = constants.get("limit", 1e-9)

#     z_tip = temp_position[2]
#     r_tip = np.linalg.norm(temp_position)
#     xy_dist = np.linalg.norm(temp_position[:2])

#     if z_tip > bottom_z + limit:
#         if r_tip > radius + limit:
#             return SpotIO.SPHERE_OUT
#         if r_tip < radius - limit:
#             return SpotIO.POLYGON_MODE if stick_status > 0 else IOStatus.INSIDE
#         return SpotIO.BORDER

#     if z_tip < bottom_z - limit:
#         denom = temp_position[2] - base_position[2]
#         if abs(denom) < limit:
#             return SpotIO.SPHERE_OUT
#         t = (bottom_z - base_position[2]) / denom
#         if t < 0 or t > 1:
#             return SpotIO.SPHERE_OUT
#         intersect_xy = base_position[:2] + t * (temp_position[:2] - base_position[:2])
#         dist_xy = np.linalg.norm(intersect_xy)
#         if dist_xy < bottom_r + limit:
#             return SpotIO.BOTTOM_OUT
#         return SpotIO.SPHERE_OUT

#     if bottom_z - limit < z_tip < bottom_z + limit:
#         if xy_dist > bottom_r + limit:
#             return SpotIO.SPOT_EDGE_OUT
#         if abs(xy_dist - bottom_r) <= limit:
#             return SpotIO.BORDER
#         if xy_dist < bottom_r - limit:
#             if prev_stat in (SpotIO.SPOT_EDGE_OUT, SpotIO.POLYGON_MODE) or stick_status > 0:
#                 return SpotIO.POLYGON_MODE
#             return SpotIO.SPOT_BOTTOM

#     return IOStatus.INSIDE


def _io_check_spot(
    base_position: np.ndarray,
    temp_position: np.ndarray,
    constants: dict,
    prev_stat: str = "inside",
    stick_status: int = 0,
) -> tuple[np.ndarray, str, bool]:
    """Return corrected position, final status and bottom hit flag.

    The candidate position is repeatedly adjusted whenever it exits the
    spherical cap so that the returned position lies inside.  If the
    trajectory hits the bottom plane during this process ``bottom_hit`` is
    set to ``True``.
    """
    candidate = temp_position.copy()
    pos = base_position.copy()
    status = _spot_status_check(pos, candidate, constants, prev_stat, stick_status)
    bottom_hit = False
    step_vec = candidate - pos
    step_len = np.linalg.norm(step_vec)
    if step_len < 1e-12:
        return candidate, status, bottom_hit
    vec = step_vec / step_len

    for _ in range(10):
        if status in (
            IOStatus.INSIDE,
            SpotIO.BORDER,
            SpotIO.SPOT_BOTTOM,
            SpotIO.POLYGON_MODE,
        ):
            break

        if status in (SpotIO.SPHERE_OUT, SpotIO.SPOT_EDGE_OUT):
            intersect, remain = _line_sphere_intersection(pos, candidate, constants["spot_r"])
            normal = intersect / (np.linalg.norm(intersect) + 1e-12)
            vec = _reflect(vec, normal)
            pos = intersect
            step_len = remain
            candidate = pos + vec * step_len

        elif status == SpotIO.BOTTOM_OUT:
            bottom_hit = True
            bottom_z = constants.get("spot_bottom_height", 0.0)
            step_vec = candidate - pos
            if abs(step_vec[2]) < 1e-12:
                proj = step_vec.copy()
                proj[2] = 0.0
                norm = np.linalg.norm(proj)
                if norm < 1e-12:
                    proj = np.array([1.0, 0.0, 0.0]) * step_len
                    norm = step_len
                vec = proj / norm
                candidate = pos + vec * step_len
            else:
                t = (bottom_z - pos[2]) / step_vec[2]
                t = max(0.0, min(1.0, t))
                intersect = pos + step_vec * t
                remain = step_len * (1.0 - t)
                proj_dir = vec.copy()
                proj_dir[2] = 0.0
                norm = np.linalg.norm(proj_dir)
                if norm < 1e-12:
                    proj_dir = np.array([1.0, 0.0, 0.0])
                    norm = 1.0
                proj_dir /= norm
                pos = intersect
                vec = proj_dir
                step_len = remain
                candidate = pos + vec * step_len

        status = _spot_status_check(pos, candidate, constants, prev_stat, stick_status)

    return candidate, status, bottom_hit


def _line_sphere_intersection(p0: np.ndarray, p1: np.ndarray, r: float) -> tuple[np.ndarray, float]:
    """Return intersection point and remaining distance after hitting the sphere."""
    d = p1 - p0
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-12:
        return p0.copy(), 0.0
    d_unit = d / d_norm
    f = p0
    a = 1.0
    b = 2.0 * float(f @ d_unit)
    c = float(f @ f) - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return p0.copy(), 0.0
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    # Only intersections that lie on the segment between ``p0`` and ``p1``
    # are valid.  Without this check ``t`` may point to an intersection far
    # away from the current step which leads to an excessively long vector.
    t_candidates = [t for t in (t1, t2) if 0 <= t <= d_norm]
    if not t_candidates:
        return p0.copy(), 0.0
    t = min(t_candidates)
    intersection = p0 + d_unit * t
    remaining = max(d_norm - t, 0.0)
    return intersection, remaining


def _reflect(vec: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect ``vec`` on plane defined by ``normal``."""
    return vec - 2.0 * np.dot(vec, normal) * normal


def polygon_mode(
    current_pos: np.ndarray,
    polygon_idx: int,
    spot_r: float,
    stick_count: int,
    deviation: float,
    spot_angle_rad: float,
) -> np.ndarray:
    """Return next vector when tracing the bottom edge polygon.

    When ``stick_count`` falls below 1, ``detach_edge_mode`` is invoked to
    generate a vector that points inside the spherical cap.
    """

    if stick_count >= 1:
        theta_step = 2 * np.pi / 70
        next_polygon_idx = (polygon_idx + 1) % 70
        next_theta = next_polygon_idx * theta_step

        x = spot_r * np.sin(spot_angle_rad) * np.cos(next_theta)
        y = spot_r * np.sin(spot_angle_rad) * np.sin(next_theta)
        z = spot_r * np.cos(spot_angle_rad)

        next_pos = np.array([x, y, z])
        next_vector = next_pos - current_pos
    else:
        next_vector = detach_edge_mode(
            current_pos, spot_r, deviation, spot_angle_rad
        )

    return next_vector


def detach_edge_mode(
    current_pos: np.ndarray,
    spot_r: float,
    deviation: float,
    spot_angle_rad: float,
) -> np.ndarray:
    """Return a vector pointing inside the spherical cap with limited spread."""

    sphere_center = np.array([0.0, 0.0, 0.0])
    base_center = np.array([0.0, 0.0, spot_r * np.cos(spot_angle_rad)])

    local_z = sphere_center - current_pos
    local_z /= np.linalg.norm(local_z) + 1e-12

    local_x = base_center - current_pos
    local_x = local_x - np.dot(local_x, local_z) * local_z
    local_x /= np.linalg.norm(local_x) + 1e-12

    local_y = np.cross(local_z, local_x)

    theta = abs(np.random.normal(0.0, deviation))
    phi_max = spot_angle_rad - (2 * np.pi / 70)
    phi = np.random.uniform(0.0, phi_max)

    local_vec = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )

    rotation = np.column_stack([local_x, local_y, local_z])
    global_vec = rotation @ local_vec
    global_vec /= np.linalg.norm(global_vec) + 1e-12

    return global_vec


class SpermSimulation:
    def __init__(self, constants):
        self.constants = constants

        # --- 型安全化：数値パラメータはfloat/intに変換 ---
        float_keys = [
            "spot_angle", "vol", "sperm_conc", "vsl", "deviation", "surface_time",
            "gamete_r", "sim_min", "sample_rate_hz"
        ]
        int_keys = [
            "sim_repeat"
        ]
        for key in float_keys:
            if key in self.constants and not isinstance(self.constants[key], float):
                try:
                    self.constants[key] = float(self.constants[key])
                except Exception:
                    print(f"[WARNING] {key} = {self.constants[key]} をfloat変換できませんでした")
        for key in int_keys:
            if key in self.constants and not isinstance(self.constants[key], int):
                try:
                    self.constants[key] = int(float(self.constants[key]))
                except Exception:
                    print(f"[WARNING] {key} = {self.constants[key]} をint変換できませんでした")
        # shape, egg_localization, などはstr型のままでOK
        self.constants = calculate_derived_constants(self.constants)
        

    
        # ------------------------------------------------------------------
    def drop_polygon_move(self, base_position, last_vec, stick_status, constants):
        """
        drop球面に張り付いた状態で這うように動く。
        stick_statusに応じて deviation の自由度を段階的に変化させる：
            - stick_status > 1  : 球面上のポリゴン面内のみを這う
            - stick_status == 1: ポリゴン面 + 球体内側方向にやや浮かせる
            - stick_status == 0: 自由運動（POLYGON_MODE終了）

        Parameters:
            base_position: 現在の位置（球面上）
            last_vec: 前回の進行ベクトル
            stick_status: 現在の貼り付き残りステップ数（実数）
            constants: 定数辞書

        Returns:
            new_temp_position: 次の位置
            new_last_vec: 次のベクトル
            new_stick_status: 更新された貼り付き時間
            next_state: IOStatus（POLYGON_MODEまたはINSIDE）
        """

        step_len = constants['step_length']
        dev_mag = constants['deviation']
        limit = constants['limit']

        # --- 法線ベクトル（球中心からの放射方向） ---
        normal = base_position / (LA.norm(base_position) + 1e-12)

        # --- ポリゴン面ベクトルの初期化（法線に直交する方向） ---
        vec_norm = LA.norm(last_vec)
        if vec_norm < limit:
            # 進行方向が小さすぎるとき → 仮の接線を定義
            if abs(normal[0]) < 0.9:
                tangent = np.cross(normal, [1.0, 0.0, 0.0])
            else:
                tangent = np.cross(normal, [0.0, 1.0, 0.0])
            v_base = tangent / (LA.norm(tangent) + 1e-12)
        else:
            # 接線方向に正規化
            v_base = last_vec / vec_norm
            v_base = v_base - np.dot(v_base, normal) * normal
            base_norm = LA.norm(v_base)
            if base_norm < limit:
                if abs(normal[0]) < 0.9:
                    tangent = np.cross(normal, [1.0, 0.0, 0.0])
                else:
                    tangent = np.cross(normal, [0.0, 1.0, 0.0])
                v_base = tangent / (LA.norm(tangent) + 1e-12)
            else:
                v_base /= base_norm

        # --- ポリゴン面（normalに直交）上の2軸 (u, v) ---
        u = v_base
        v = np.cross(normal, u)
        v /= LA.norm(v) + 1e-12

        # --- stick_status に応じた deviation の生成 ---
        if stick_status > 1:
            # ポリゴン面内で左右にぶれるだけ
            theta = np.random.uniform(-np.pi, np.pi)
            deviation_vec = dev_mag * (np.cos(theta) * u + np.sin(theta) * v)

        elif stick_status == 1:
            # 最後だけ球面内方向にもぶれられるように（半コーン）
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            deviation_vec = dev_mag * (np.cos(theta) * normal + np.sin(theta) * v_base)

        else:
            # POLYGON_MODE 終了 → 自由運動
            rand_vec = np.random.normal(0, 1, 3)
            rand_vec /= LA.norm(rand_vec) + 1e-12
            deviation_vec = dev_mag * rand_vec

        # --- 新しい進行方向ベクトル ---
        final_dir = v_base + deviation_vec
        final_dir /= LA.norm(final_dir) + 1e-12

        new_last_vec = final_dir * step_len
        new_temp_position = base_position + new_last_vec

        # --- 貼り付き残り時間の更新と状態切り替え ---
        new_stick_status = stick_status - 1 if stick_status > 0 else 0
        if new_stick_status <= 0:
            next_state = IOStatus.INSIDE
        else:
            next_state = IOStatus.POLYGON_MODE

        return new_temp_position, new_last_vec, new_stick_status, next_state

    def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
        vector = temp_position - base_position
        # if LA.norm(vector) < 1e-9:
        #     raise RuntimeError("zzz")
        distance_base = LA.norm(base_position - egg_center)
        distance_tip = LA.norm(temp_position - egg_center)
        if distance_base <= gamete_r or distance_tip <= gamete_r:
            return True
        f = base_position - egg_center
        a = vector @ vector
        b = 2 * (f @ vector)
        c = f @ f - gamete_r**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return False
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2*a)
        t2 = (-b + sqrt_discriminant) / (2*a)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        return False
    
    from tools.geometry import DropShape, CubeShape, SpotShape  # cerosは内部処理で対応
    from tools.io_checks import IO_check_cube, IO_check_drop, IO_check_spot  # 必要に応じて

    def run(self, constants, result_dir, save_name, save_flag):
        """
        全てのshape（cube, drop, spot, ceros）に対応した精子運動シミュレーション実行関数。
        self.trajectory および self.vectors を更新する。
        """
        self.constants = constants
        shape = constants["shape"]
        step_len = constants["step_length"]
        vsl = constants["vsl"]
        hz = constants["sample_rate_hz"]
        deviation = constants["deviation"]
        seed = int(constants.get("seed_number", 0))

        self.number_of_sperm = int(constants["sperm_conc"] * constants["vol"] * 1e-3)
        self.number_of_steps = int(constants["sim_min"] * hz * 60)

        rng = np.random.default_rng(seed)

        # === 初期位置と形状オブジェクト ===
        if shape == "drop":
            shape_obj = DropShape(constants)
        elif shape == "cube":
            shape_obj = CubeShape(constants)
        elif shape == "spot":
            shape_obj = SpotShape(constants)
        elif shape == "ceros":
            shape_obj = None
        else:
            raise ValueError(f"Unknown shape: {shape}")

        if shape == "ceros":
            self.initial_position = np.full((self.number_of_sperm, 3), np.inf)
        else:
            self.initial_position = np.zeros((self.number_of_sperm, 3))
            for j in range(self.number_of_sperm):
                self.initial_position[j] = shape_obj.initial_position()

        # === 初期ベクトル（ランダム方向） ===
        self.initial_vectors = np.zeros((self.number_of_sperm, 3))
        for j in range(self.number_of_sperm):
            vec = rng.normal(0, 1, 3)
            vec /= np.linalg.norm(vec) + 1e-12
            self.initial_vectors[j] = vec

        # === 配列初期化 ===
        self.trajectory = np.zeros((self.number_of_sperm, self.number_of_steps, 3))
        self.vectors = np.zeros((self.number_of_sperm, self.number_of_steps, 3))

        # === メインループ ===
        for j in range(self.number_of_sperm):
            pos = self.initial_position[j].copy()
            vec = self.initial_vectors[j].copy()
            stick_status = 0
            prev_stat = "inside"

            self.trajectory[j, 0] = pos
            self.vectors[j, 0] = vec

            for i in range(1, self.number_of_steps):
                vec += rng.normal(0, deviation, 3)
                vec /= np.linalg.norm(vec) + 1e-12
                candidate = pos + vec * step_len

                # === IO 判定 ===
                if shape == "cube":
                    status, _ = IO_check_cube(candidate, constants)
                elif shape == "drop":
                    status, stick_status = IO_check_drop(candidate, stick_status, constants)
                elif shape == "spot":
                    status = IO_check_spot(pos, candidate, constants, prev_stat, stick_status)
                    prev_stat = status
                elif shape == "ceros":
                    status = IOStatus.INSIDE
                else:
                    raise ValueError(f"Unknown shape: {shape}")

                # === ステータスごとの挙動 ===
                if status in [IOStatus.INSIDE, IOStatus.INSIDE]:
                    pos = candidate
                elif status == IOStatus.POLYGON_MODE:
                    candidate, vec, stick_status, status = self.drop_polygon_move(pos, vec, stick_status, self.constants)


                elif status in [IOStatus.REFLECT, SpotIO.REFLECT]:
                    vec *= -1

                elif status in [IOStatus.STICK, SpotIO.STICK, SpotIO.POLYGON_MODE]:
                    stick_status = int(constants["surface_time"] / hz)

                elif status in [IOStatus.BORDER, SpotIO.BORDER, SpotIO.BOTTOM_OUT]:
                    pass  # 境界付近で停止
                elif status in [IOStatus.BORDER, IOStatus.BOTTOM_OUT, IOStatus.SPOT_EDGE_OUT]:
                    pass  # 停止や跳ね返り処理なし（その場維持）


                else:
                    print(f"[WARNING] Unexpected status: {status}")
                    pass

                self.trajectory[j, i] = pos
                self.vectors[j, i] = vec

        print("[DEBUG] 初期位置数:", len(self.initial_position))
        print("[DEBUG] 精子数:", self.number_of_sperm)
        self.trajectories = self.trajectory  # 外部用に公開


    import matplotlib.pyplot as plt

    def plot_trajectories(self, max_sperm=5, save_path=None):
        """
        インスタンスのself.trajectory（リスト of N×3 配列）を可視化
        max_sperm: 表示する精子軌跡の最大本数
        save_path: Noneなら画面表示のみ、パス指定で保存
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        trajectories = np.array(self.trajectory)
        constants = self.constants

        if trajectories is None or len(trajectories) == 0:
            print("[WARNING] 軌跡データがありません。run()実行後にplot_trajectoriesしてください。")
            return

        # --- 軸幅の統一 ---
        all_mins = [constants["x_min"], constants["y_min"], constants["z_min"]]
        all_maxs = [constants["x_max"], constants["y_max"], constants["z_max"]]
        global_min = min(all_mins)
        global_max = max(all_maxs)

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        ax_xy, ax_xz, ax_yz = axes
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        n_sperm = min(len(trajectories), max_sperm)

        # --- draw egg position ---
        egg_x, egg_y, egg_z = _egg_position(constants)
        for ax, (x, y) in zip(axes, [(egg_x, egg_y), (egg_x, egg_z), (egg_y, egg_z)]):
            ax.add_patch(
                patches.Circle(
                    (x, y),
                    radius=constants.get("gamete_r", 0),
                    facecolor="yellow",
                    alpha=0.8,
                    ec="gray",
                    linewidth=0.5,
                )
            )

        # XY投影
        for i in range(n_sperm):
            ax_xy.plot(trajectories[i][:,0], trajectories[i][:,1], color=colors[i % len(colors)])
        ax_xy.set_xlim(global_min, global_max)
        ax_xy.set_ylim(global_min, global_max)
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("XY projection")

        # XZ投影
        for i in range(n_sperm):
            ax_xz.plot(trajectories[i][:,0], trajectories[i][:,2], color=colors[i % len(colors)])
        ax_xz.set_xlim(global_min, global_max)
        ax_xz.set_ylim(global_min, global_max)
        ax_xz.set_aspect('equal')
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_title("XZ projection")

        # YZ投影
        for i in range(n_sperm):
            ax_yz.plot(trajectories[i][:,1], trajectories[i][:,2], color=colors[i % len(colors)])
        ax_yz.set_xlim(global_min, global_max)
        ax_yz.set_ylim(global_min, global_max)
        ax_yz.set_aspect('equal')
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_title("YZ projection")

        param_summary = ', '.join(
            f"{k}={constants.get(k)}" for k in [
                "shape",
                "vol",
                "sperm_conc",
                "vsl",
                "deviation",
            ]
        )
        param_summary2 = ', '.join(
            f"{k}={constants.get(k)}" for k in [
                "surface_time",
                "egg_localization",
                "gamete_r",
                "sim_min",
                "sample_rate_hz",
                "sim_repeat",
            ]
        )
        fig.suptitle(f"{param_summary}\n{param_summary2}", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


    # def plot_trajectories(self, max_sperm=5, save_path=None):
        """
        シミュレーションで記録したself.trajectory（リスト of N×3 配列）を可視化します。
        max_sperm: 表示する精子軌跡の最大本数
        save_path: Noneならこのスクリプトと同じ階層のFigs_and_Moviesに自動保存
        """
        trajectories = self.trajectory  # run()でセットされているはず
        if not trajectories or len(trajectories) == 0:
            print("[WARNING] 軌跡データがありません。run()実行後にplot_trajectoriesしてください。")
            return

        # 描画本数を制限
        n_plot = min(max_sperm, len(trajectories))
        perc_shown = n_plot / len(trajectories) * 100

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))  # XY, XZ, YZ
        ax_labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
        idxs = [(0, 1), (0, 2), (1, 2)]

        for ax, (label_x, label_y), (i, j) in zip(axes, ax_labels, idxs):
            for t in trajectories[:n_plot]:
                ax.plot(t[:, i], t[:, j], alpha=0.7)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.set_aspect('equal')
            ax.set_title(f"{label_x}{label_y} projection")

        # suptitleにパラメータ2行分
        param_summary = ', '.join(f"{k}={self.constants.get(k)}" for k in ["shape", "vol", "sperm_conc", "vsl", "deviation"])
        param_summary2 = ', '.join(f"{k}={self.constants.get(k)}" for k in ["surface_time", "egg_localization", "gamete_r", "sim_min", "sample_rate_hz", "sim_repeat"])
        fig.suptitle(f"{param_summary}\n{param_summary2}", fontsize=12)

        # 本数注釈
        fig.text(0.99, 0.01, f"※ 表示は全体の{perc_shown:.1f}%（{n_plot}本/{len(trajectories)}本）", ha='right', fontsize=10, color="gray")

        fig.tight_layout(rect=[0, 0.03, 1, 0.92])

        # --- 保存先パスをスクリプトの場所基準で作る ---
        import datetime
        dtstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # このスクリプトがあるディレクトリを基準にする
        base_dir = os.path.dirname(__file__)
        figs_dir = os.path.join(base_dir, "figs_and_movies")
        os.makedirs(figs_dir, exist_ok=True)

        if save_path is None:
            # 自動で日付入りファイル名
            save_path = os.path.join(figs_dir, f"trajectory_{dtstr}.png")
        else:
            # ファイル名だけ渡された場合もFigs_and_Moviesに入れる
            filename = os.path.basename(save_path)
            save_path = os.path.join(figs_dir, filename)

        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] 軌跡画像を保存しました: {save_path}")

    def plot_movie_trajectories(self, save_path=None, fps: int = 5):
        """Animate recorded trajectories and save to a movie file."""
        import numpy as np
        from matplotlib.animation import FuncAnimation

        trajectories = np.array(self.trajectory)
        if trajectories is None or len(trajectories) == 0:
            print("[WARNING] 軌跡データがありません。run()実行後にplot_movie_trajectoriesしてください。")
            return None

        n_sperm, n_frames, _ = trajectories.shape

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        const = self.constants
        ax.set_xlim(const["x_min"], const["x_max"])
        ax.set_ylim(const["y_min"], const["y_max"])
        ax.set_zlim(const["z_min"], const["z_max"])
        ax.set_box_aspect([
            const["x_max"] - const["x_min"],
            const["y_max"] - const["y_min"],
            const["z_max"] - const["z_min"],
        ])

        lines = [ax.plot([], [], [], lw=0.7)[0] for _ in range(n_sperm)]

        def init():
            for ln in lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
            return lines

        def update(frame):
            for i, ln in enumerate(lines):
                ln.set_data(trajectories[i, : frame + 1, 0], trajectories[i, : frame + 1, 1])
                ln.set_3d_properties(trajectories[i, : frame + 1, 2])
            return lines

        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=False)

        base_dir = os.path.dirname(__file__)
        mov_dir = os.path.join(base_dir, "figs_and_movies")
        os.makedirs(mov_dir, exist_ok=True)
        if save_path is None:
            dtstr = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(mov_dir, f"trajectory_{dtstr}.mp4")
        else:
            filename = os.path.basename(save_path)
            save_path = os.path.join(mov_dir, filename)

        try:
            anim.save(save_path, writer="ffmpeg", fps=fps)
        except Exception as e:
            print(f"[WARN] ffmpeg保存失敗 ({e}) → pillow writerで再試行")
            try:
                anim.save(save_path, writer="pillow", fps=fps)
            except Exception as e2:
                print(f"[ERROR] pillow writerでも保存に失敗: {e2}")
                return None

        plt.close(fig)
        print(f"[INFO] 動画を保存しました: {save_path}")
        return save_path

        return np.array(traj), intersection_records