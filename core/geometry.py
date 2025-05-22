import numpy as np
from numpy import linalg as LA
from typing import Dict      # ★ この行を追加


class IOStatus:
    INSIDE = "inside"
    TEMP_ON_SURFACE = "temp_on_surface"
    TEMP_ON_EDGE = "temp_on_edge"
    OUTSIDE = "outside"

class BaseShape:
    def __init__(self, constants):
        self.constants = constants

    def get_limits(self):
        # ここで一括管理（すべてのShapeはこのまま継承）
        keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        return tuple(float(self.constants[k]) for k in keys)

    def io_check(self, *args, **kwargs):
        raise NotImplementedError

    def initial_position(self):
        raise NotImplementedError

# class CubeShape(BaseShape):
#     def initial_position(self):
#         x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
#         return np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])

#     def io_check(self, point):
#         # get_limitsを必ず使う
#         x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
#         eps = 1e-9
#         inside = (x_min < point[0] < x_max) and (y_min < point[1] < y_max) and (z_min < point[2] < z_max)
#         if inside:
#             return IOStatus.INSIDE, None
#         on_edge = (
#             np.isclose([point[0]], [x_min, x_max], atol=eps).any() or
#             np.isclose([point[1]], [y_min, y_max], atol=eps).any() or
#             np.isclose([point[2]], [z_min, z_max], atol=eps).any()
#         )
#         if on_edge:
#             return IOStatus.TEMP_ON_EDGE, None
#         return IOStatus.OUTSIDE, None
class CubeShape(BaseShape):
    """
    立方体形状
    - constants に
        • "vol"      : 体積 [µL]          (従来)
        • "vol_um3"  : 体積 [µm³] ←★追加
      のどちらかが入っていれば初期化できます。
    - 計算した一辺長 edge_um, limits は self.constants に追記。
    """

    def __init__(self, constants: Dict[str, float]):
        super().__init__(constants)

        # ------------------ 追加：µm³ 指定をサポート ------------------
        if "vol_um3" in self.constants and "vol" not in self.constants:
            # µm³  →  µL  (1 µm³ = 1e-9 µL)
            self.constants["vol"] = float(self.constants["vol_um3"]) * 1e-9
        # -------------------------------------------------------------

        if "vol" not in self.constants:
            raise ValueError("CubeShape: constants に 'vol' か 'vol_um3' が必要です")

        # ------ 一辺長 edge_um と limits を派生変数として計算 ------
        vol_um3 = self.constants["vol"] * 1e9        # µm³
        edge_um = vol_um3 ** (1.0 / 3.0)             # 一辺 [µm]
        half = edge_um / 2.0

        # 派生値を constants に保存（他モジュールでも共通利用可）
        self.constants.update({
            "edge_um": edge_um,
            "x_min": -half, "x_max": half,
            "y_min": -half, "y_max": half,
            "z_min": -half, "z_max": half,
        })
        # -------------------------------------------------------------

    # ------------- 以降は Masaru さんの元コードをそのまま残す -------------
    def initial_position(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        return np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])

    def io_check(self, point):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        eps = 1e-9
        inside = (x_min < point[0] < x_max) and (y_min < point[1] < y_max) and (z_min < point[2] < z_max)
        if inside:
            return IOStatus.INSIDE, None
        on_edge = (
            np.isclose([point[0]], [x_min, x_max], atol=eps).any() or
            np.isclose([point[1]], [y_min, y_max], atol=eps).any() or
            np.isclose([point[2]], [z_min, z_max], atol=eps).any()
        )
        if on_edge:
            return IOStatus.TEMP_ON_EDGE, None
        return IOStatus.OUTSIDE, None

class DropShape(BaseShape):
    def initial_position(self):
        R = float(self.constants['drop_r'])
        theta = np.arccos(2 * np.random.rand() - 1)
        phi = np.random.uniform(-np.pi, np.pi)
        s = R * np.random.rand() ** (1/3)
        x = s * np.sin(theta) * np.cos(phi)
        y = s * np.sin(theta) * np.sin(phi)
        z = s * np.cos(theta)
        return np.array([x, y, z])

    def io_check(self, point, stick_status):
        R = float(self.constants["drop_r"])
        if point[2] < 0:
            return IOStatus.OUTSIDE
        norm = np.linalg.norm(point)
        if norm < R:
            return IOStatus.INSIDE
        if np.isclose(norm, R, atol=1e-9):
            return IOStatus.TEMP_ON_SURFACE
        return IOStatus.OUTSIDE

class SpotShape(BaseShape):
    def initial_position(self):
        radius = float(self.constants['radius'])
        spot_angle_rad = np.deg2rad(float(self.constants['spot_angle']))
        while True:
            theta = np.random.uniform(0, spot_angle_rad)
            phi = np.random.uniform(-np.pi, np.pi)
            r = radius * (np.random.rand() ** (1/3))
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            if z >= radius * np.cos(spot_angle_rad):
                break
        return np.array([x, y, z])

    def io_check(self, base_point, temp_point=None):
        R = float(self.constants["radius"])
        if temp_point is None:
            return IOStatus.OUTSIDE
        norm = np.linalg.norm(temp_point)
        if norm < R:
            return IOStatus.INSIDE
        if np.isclose(norm, R, atol=1e-9):
            return IOStatus.TEMP_ON_SURFACE
        return IOStatus.OUTSIDE

class CerosShape(BaseShape):
    def initial_position(self):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        return np.random.uniform([x_min, y_min, z_min], [x_max, y_max, z_max])

    def io_check(self, point):
        # cerosもcubeと同じ判定
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_limits()
        eps = 1e-9
        inside = (x_min < point[0] < x_max) and (y_min < point[1] < y_max) and (z_min < point[2] < z_max)
        if inside:
            return IOStatus.INSIDE, None
        on_edge = (
            np.isclose([point[0]], [x_min, x_max], atol=eps).any() or
            np.isclose([point[1]], [y_min, y_max], atol=eps).any() or
            np.isclose([point[2]], [z_min, z_max], atol=eps).any()
        )
        if on_edge:
            return IOStatus.TEMP_ON_EDGE, None
        return IOStatus.OUTSIDE, None
