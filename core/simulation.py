
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from datetime import datetime
import math
import os
from tools.plot_utils import plot_2d_trajectories
from core.geometry import CubeShape, DropShape, SpotShape, CerosShape
from tools.derived_constants import calculate_derived_constants


def _make_local_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthogonal to ``forward``."""
    f = forward / (np.linalg.norm(forward) + 1e-12)
    if abs(f[0]) < 0.9:
        base = np.array([1.0, 0.0, 0.0])
    else:
        base = np.array([0.0, 1.0, 0.0])
    y = np.cross(f, base)
    y /= np.linalg.norm(y) + 1e-12
    x = np.cross(y, f)
    return x, y


def _perturb_direction(prev: np.ndarray, deviation: float, rng: np.random.Generator) -> np.ndarray:
    """Return a unit vector deviated from ``prev``."""
    lx, ly = _make_local_basis(prev)
    theta = rng.normal(0.0, deviation)
    phi = rng.uniform(-np.pi, np.pi)
    new_dir = (
        np.cos(theta) * prev
        + np.sin(theta) * (np.cos(phi) * lx + np.sin(phi) * ly)
    )
    new_dir /= np.linalg.norm(new_dir) + 1e-12
    return new_dir


def _egg_position(constants):
    """Return (egg_x, egg_y, egg_z) according to shape and localization."""
    shape = constants.get("shape", "cube").lower()
    loc = constants.get("egg_localization", "center")
    r = constants["gamete_r"]

    if shape == "cube":
        positions = {
            "center": (0, 0, 0),
            "bottom_center": (0, 0, constants["z_min"] + r),
            "bottom_side": (
                constants["x_min"] / 2 + r,
                constants["y_min"] / 2 + r,
                constants["z_min"] + r,
            ),
            "bottom_corner": (
                constants["x_min"] + r,
                constants["y_min"] + r,
                constants["z_min"] + r,
            ),
        }
    elif shape == "drop":
        drop_r = constants["drop_r"]
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
        cx = (constants["x_min"] + constants["x_max"]) / 2
        cy = (constants["y_min"] + constants["y_max"]) / 2
        cz = (constants["z_min"] + constants["z_max"]) / 2
        positions = {"center": (cx, cy, cz), "bottom_center": (cx, cy, cz), "bottom_edge": (cx, cy, cz)}
    else:
        raise RuntimeError(f"Unknown shape '{shape}'")

    if loc not in positions:
        raise RuntimeError(f"Invalid egg_localization '{loc}' for shape '{shape}'")

    return positions[loc]


def _io_check_drop(
    position: np.ndarray, constants: dict, base_position: np.ndarray
) -> str:
    """Return in/out status for drop shape with extended boundary check."""

    r = constants.get("drop_r", 0.0)
    limit = constants.get("limit", 1e-9)

    dist = np.linalg.norm(position)

    if dist > r + limit:
        return "outside"
    if dist < r - limit:
        return "inside"

    # Near the border, extend the vector from ``base_position``
    # to ``position`` and re-evaluate.
    vector = position - base_position
    extended_position = base_position + vector * 1.2
    extended_dist = np.linalg.norm(extended_position)

    if extended_dist <= r:
        return "inside"
    return "outside"


class SpotIO:
    """Status constants for spot geometry."""

    INSIDE = "inside"
    BORDER = "border"
    SPHERE_OUT = "sphere_out"
    BOTTOM_OUT = "bottom_out"
    SPOT_EDGE_OUT = "spot_edge_out"
    POLYGON_MODE = "polygon_mode"
    SPOT_BOTTOM = "spot_bottom"


def _spot_status_check(
    base_position: np.ndarray,
    temp_position: np.ndarray,
    constants: dict,
    prev_stat: str = "inside",
    stick_status: int = 0,
) -> str:
    """Return IO status for spot shape without modifying the position."""

    radius = constants.get("spot_r", constants.get("radius", 0.0))
    bottom_z = constants.get("spot_bottom_height", 0.0)
    bottom_r = constants.get("spot_bottom_r", 0.0)
    limit = constants.get("limit", 1e-9)

    z_tip = temp_position[2]
    r_tip = np.linalg.norm(temp_position)
    xy_dist = np.linalg.norm(temp_position[:2])

    if z_tip > bottom_z + limit:
        if r_tip > radius + limit:
            return SpotIO.SPHERE_OUT
        if r_tip < radius - limit:
            return SpotIO.POLYGON_MODE if stick_status > 0 else SpotIO.INSIDE
        return SpotIO.BORDER

    if z_tip < bottom_z - limit:
        denom = temp_position[2] - base_position[2]
        if abs(denom) < limit:
            return SpotIO.SPHERE_OUT
        t = (bottom_z - base_position[2]) / denom
        if t < 0 or t > 1:
            return SpotIO.SPHERE_OUT
        intersect_xy = base_position[:2] + t * (temp_position[:2] - base_position[:2])
        dist_xy = np.linalg.norm(intersect_xy)
        if dist_xy < bottom_r + limit:
            return SpotIO.BOTTOM_OUT
        return SpotIO.SPHERE_OUT

    if bottom_z - limit < z_tip < bottom_z + limit:
        if xy_dist > bottom_r + limit:
            return SpotIO.SPOT_EDGE_OUT
        if abs(xy_dist - bottom_r) <= limit:
            return SpotIO.BORDER
        if xy_dist < bottom_r - limit:
            if prev_stat in (SpotIO.SPOT_EDGE_OUT, SpotIO.POLYGON_MODE) or stick_status > 0:
                return SpotIO.POLYGON_MODE
            return SpotIO.SPOT_BOTTOM

    return SpotIO.INSIDE


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
            SpotIO.INSIDE,
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
            "gamete_r", "sim_min", "sampl_rate_hz"
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
    # ここから 全面置換  ※クラス定義はそのまま、run() だけ差し替え
    # ------------------------------------------------------------------
       # ------------------------------------------------------------------
    # ★★★ ここから run() 全面置換 ★★★
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # ★★★ ここから run() 全面置換 ★★★
    # ------------------------------------------------------------------
    def run(self, sim_repeat: int = 1):
        """
        sim_repeat 回シミュレーションを実行して
        self.trajectory に各精子の N×3 軌跡配列を格納する。
        座標・距離の単位は **mm** で統一。
        """
        # ---- デバッグ：派生値確認 -------------------------------------
        print("[DEBUG] SpermSimulation パラメータ:", self.constants)

        # ---- 形状オブジェクト生成 -------------------------------------
        shape = self.constants.get("shape", "cube")
        if shape == "cube":
            shape_obj = CubeShape(self.constants)
        elif shape == "spot":
            shape_obj = SpotShape(self.constants)
        elif shape == "drop":
            shape_obj = DropShape(self.constants)
        elif shape == "ceros":
            shape_obj = CerosShape(self.constants)
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        # ---- シミュレーション設定 -------------------------------------
        number_of_sperm  = int(self.constants.get("number_of_sperm", 10))
        number_of_steps  = int(self.constants.get("number_of_steps", 10))
        step_len         = self.constants["step_length"]     # ← mm / step
        seed_val         = self.constants.get("seed_number")
        if seed_val is not None and str(seed_val).lower() != "none":
            try:
                seed_int = int(seed_val)
                # --- 全ての乱数生成を同じシードで制御するため ---
                np.random.seed(seed_int)
                rng = np.random.default_rng(seed_int)
            except Exception:
                rng = np.random.default_rng()
        else:
            rng = np.random.default_rng()

        self.trajectory = []   # ← 毎 run() でリセット
        prev_states = [SpotIO.INSIDE for _ in range(number_of_sperm)]
        bottom_modes = [False for _ in range(number_of_sperm)]
        stick_statuses = [0 for _ in range(number_of_sperm)]

        # ---- ループ ---------------------------------------------------
        for rep in range(int(sim_repeat)):
            for i in range(number_of_sperm):

                pos = shape_obj.initial_position()     # mm
                traj = [pos.copy()]
                
                # 初期方向
                vec = rng.normal(size=3)
                vec /= np.linalg.norm(vec) + 1e-12

                for j in range(number_of_steps):
                    if j > 0:
                        vec = _perturb_direction(vec, self.constants["deviation"], rng)
                    if shape == "spot" and bottom_modes[i] and stick_statuses[i] > 0:
                        vec[2] = 0.0
                        vec /= np.linalg.norm(vec) + 1e-12
                    candidate = pos + vec * step_len
                    if shape == "drop":
                        base_pos = pos
                        move_len = step_len
                        status = _io_check_drop(candidate, self.constants, base_pos)
                        if status == "bottom":
                            step_vec = vec * move_len
                            if abs(step_vec[2]) < 1e-12:
                                normal = np.array([0.0, 0.0, 1.0])
                                vec = _reflect(vec, normal)
                                candidate = base_pos + vec * move_len
                            else:
                                t = (0.0 - base_pos[2]) / step_vec[2]
                                t = max(0.0, min(1.0, t))
                                intersect = base_pos + step_vec * t
                                remain = move_len * (1.0 - t)
                                normal = np.array([0.0, 0.0, 1.0])
                                vec = _reflect(vec, normal)
                                base_pos = intersect
                                candidate = base_pos + vec * remain
                            status = _io_check_drop(candidate, self.constants, base_pos)
                        if status == "outside":
                            intersect, remain = _line_sphere_intersection(
                                base_pos, candidate, self.constants["drop_r"]
                            )
                            normal = intersect / (np.linalg.norm(intersect) + 1e-12)
                            vec = _reflect(vec, normal)
                            candidate = intersect + vec * remain
                    elif shape == "spot":
                        prev = prev_states[i]
                        candidate, status, bottom_hit = _io_check_spot(
                            pos, candidate, self.constants, prev, stick_statuses[i]
                        )
                        vec = (candidate - pos) / step_len
                        if bottom_hit:
                            bottom_modes[i] = True
                            if stick_statuses[i] == 0:
                                stick_statuses[i] = int(
                                    self.constants["surface_time"]
                                    / self.constants["sampl_rate_hz"]
                                )

                    disp_len = np.linalg.norm(candidate - pos)
                    max_len = step_len
                    if disp_len > max_len * 1.05:
                        print(
                            f"[ERROR] displacement {disp_len} mm exceeds "
                            f"step_length {max_len} mm at rep={rep}, sperm={i}, step={j}"
                        )
                        print(f"pos={pos}, candidate={candidate}, vec={vec}")
                        raise RuntimeError("step length exceeded")

                    pos = candidate
                    traj.append(pos.copy())
                    if shape == "spot":
                        prev_states[i] = status
                        if bottom_modes[i]:
                            if stick_statuses[i] > 0:
                                stick_statuses[i] -= 1
                            if stick_statuses[i] == 0:
                                bottom_modes[i] = False

                    if rep == 0 and i == 0 and j == 0:
                        print(f"[DEBUG] 1step_disp(mm) = {np.linalg.norm(vec*step_len):.5f}")

                self.trajectory.append(np.vstack(traj))

        print(f"[DEBUG] run完了: sperm={len(self.trajectory)}, steps={number_of_steps}, "
              f"step_len={step_len} mm")



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
                    linewidth=1.0,
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
                "sampl_rate_hz",
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
        param_summary2 = ', '.join(f"{k}={self.constants.get(k)}" for k in ["surface_time", "egg_localization", "gamete_r", "sim_min", "sampl_rate_hz", "sim_repeat"])
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

        lines = [ax.plot([], [], [], lw=1)[0] for _ in range(n_sperm)]

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
