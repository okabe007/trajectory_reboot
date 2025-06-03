import numpy as np
from numpy import linalg as LA
from tools.derived_constants import calculate_derived_constants
from simcore.motion_modes.polygon_drop import drop_polygon_move
from tools.enums import IOStatus, SpotIO


class SimulationEngine:
    def __init__(self, constants: dict):
        self.constants = constants
        # 必要に応じて self.motion_mode などの初期化もここで行う
        # 例: self.motion_mode = self._select_mode()

        """
        constants: GUIやconfigファイルから与えられたパラメータ辞書
        """
        self.constants = constants.copy()  # 外部改変防止

        # --- 型安全化：数値パラメータはfloat/intに変換 ---
        float_keys = [
            "spot_angle", "vol", "sperm_conc", "vsl", "deviation", "surface_time",
            "gamete_r", "sim_min", "sample_rate_hz"
        ]
        int_keys = ["sim_repeat"]

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

        # --- 派生変数の計算 ---
        self.constants = calculate_derived_constants(self.constants)

        # --- 各種パラメータ抽出 ---
        self.shape = self.constants.get("shape", "cube")
        self.step_length = self.constants["step_length"]
        self.vsl = self.constants["vsl"]
        self.hz = self.constants["sample_rate_hz"]
        self.deviation = self.constants["deviation"]
        self.seed = int(self.constants.get("seed_number", 0))

        # --- シミュレーション数設定 ---
        self.number_of_sperm = int(self.constants["sperm_conc"] * self.constants["vol"] * 1e-3)
        self.number_of_steps = int(self.constants["sim_min"] * self.hz * 60)

        # --- 乱数生成器 ---
        self.rng = np.random.default_rng(self.seed)

        # --- プレースホルダ（軌跡・ベクトルなど） ---
        self.initial_position = None
        self.initial_vectors = None
        self.trajectory = None
        self.vectors = None
        self.trajectories = None  # 外部公開用
from tools.geometry import DropShape, CubeShape, SpotShape
# from tools.enums import IOStatus, SpotIO
from tools.io_checks import IO_check_cube, IO_check_drop, IO_check_spot

class SimulationEngine:
    ...
    def simulate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        シミュレーション本体処理。全精子の軌跡（trajectory）と方向ベクトル（vectors）を生成。
        """
        constants = self.constants
        shape = self.shape
        rng = self.rng

        # --- 初期位置と形状オブジェクトの生成 ---
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

        # --- 初期位置 ---
        if shape == "ceros":
            self.initial_position = np.full((self.number_of_sperm, 3), np.inf)
        else:
            self.initial_position = np.zeros((self.number_of_sperm, 3))
            for j in range(self.number_of_sperm):
                self.initial_position[j] = shape_obj.initial_position()

        # --- 初期ベクトル ---
        self.initial_vectors = np.zeros((self.number_of_sperm, 3))
        for j in range(self.number_of_sperm):
            vec = rng.normal(0, 1, 3)
            vec /= np.linalg.norm(vec) + 1e-12
            self.initial_vectors[j] = vec

        # --- 配列初期化 ---
        self.trajectory = np.zeros((self.number_of_sperm, self.number_of_steps, 3))
        self.vectors = np.zeros((self.number_of_sperm, self.number_of_steps, 3))

        # --- メインループ ---
        for j in range(self.number_of_sperm):
            pos = self.initial_position[j].copy()
            vec = self.initial_vectors[j].copy()
            stick_status = 0
            prev_stat = "inside"

            self.trajectory[j, 0] = pos
            self.vectors[j, 0] = vec

            for i in range(1, self.number_of_steps):
                vec += rng.normal(0, self.deviation, 3)
                vec /= np.linalg.norm(vec) + 1e-12
                candidate = pos + vec * self.step_length

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

                # === 状態に応じた処理 ===
                if status == IOStatus.INSIDE:
                    pos = candidate
                elif status == IOStatus.POLYGON_MODE:
                    from simcore.motion_modes.polygon_drop import drop_polygon_move
                    candidate, vec, stick_status, status = drop_polygon_move(pos, vec, stick_status, constants)
                    pos = candidate
                elif status in [IOStatus.REFLECT, SpotIO.REFLECT]:
                    vec *= -1
                elif status in [IOStatus.STICK, SpotIO.STICK, SpotIO.POLYGON_MODE]:
                    stick_status = int(constants["surface_time"] / self.hz)
                elif status in [IOStatus.BORDER, SpotIO.BORDER, SpotIO.BOTTOM_OUT, IOStatus.SPOT_EDGE_OUT]:
                    pass  # 現在維持
                else:
                    print(f"[WARNING] Unexpected status: {status}")
                    pass

                self.trajectory[j, i] = pos
                self.vectors[j, i] = vec

        print("[DEBUG] 初期位置数:", len(self.initial_position))
        print("[DEBUG] 精子数:", self.number_of_sperm)

        self.trajectories = self.trajectory  # 外部公開用
        return self.trajectory, self.vectors
