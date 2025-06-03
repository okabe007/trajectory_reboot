import numpy as np
from numpy import linalg as LA
from typing import Tuple
from tools.enums import IOStatus
import numpy as np

def IO_check_drop(
    temp_position: np.ndarray,
    stick_status: float,
    constants: dict
    ) -> Tuple["IOStatus", float]:
    """
    Drop型空間におけるIO判定および吸着状態の管理。

    Parameters:
        temp_position : np.ndarray (3,) - 精子先端の現在位置
        stick_status  : float            - 現在の貼りつき残りステップ数
        constants     : dict             - 'drop_r', 'limit', 'surface_time', 'sample_rate_hz' を含む

    Returns:
        IOStatus: POLYGON_MODE / INSIDE
        new_stick_status: 更新された貼りつき時間（step単位）
    """

    radius = constants['drop_r']
    limit = constants['limit']
    distance = LA.norm(temp_position)

    # ① 貼りつき中の処理（stick_status > 0）
    if stick_status > 0:
        if distance < radius - limit:
            # → 先端がdrop内部に戻った → 吸着時間を1減らす
            new_stick_status = max(0, stick_status - 1)
            if new_stick_status == 0:
                # 吸着終了 → 自由運動へ
                return IOStatus.INSIDE, 0
            else:
                # まだ吸着中
                return IOStatus.POLYGON_MODE, new_stick_status
        else:
            # 外にいる・接触中 → 吸着継続
            return IOStatus.POLYGON_MODE, stick_status

    # ② 吸着していない（stick_status == 0）場合
    else:
        if distance > radius + limit:
            # dropの外に出た → 吸着を開始
            new_stick_status = constants["surface_time"] / constants["sample_rate_hz"]
            return IOStatus.POLYGON_MODE, new_stick_status
        else:
            # 中にとどまっている or 境界 → 自由運動継続
            return IOStatus.INSIDE, 0

def IO_check_cube(temp_position, constants):
    """
    Cube 型空間における IO 判定。min/max を直接 constants から参照。

    Parameters:
        temp_position : np.ndarray - 判定対象の位置
        constants     : dict       - 'x_min' などを含む設定辞書

    Returns:
        IOStatus, vertex_coords (if applicable)
    """
    x_min = constants['x_min']
    x_max = constants['x_max']
    y_min = constants['y_min']
    y_max = constants['y_max']
    z_min = constants['z_min']
    z_max = constants['z_max']
    limit = constants['limit']

    def classify_dimension(pos, min_val, max_val):
        if pos < min_val - limit:
            return IOStatus.OUTSIDE
        elif pos > max_val + limit:
            return IOStatus.OUTSIDE
        elif abs(pos - min_val) <= limit or abs(pos - max_val) <= limit:
            return IOStatus.SURFACE
        else:
            return IOStatus.INSIDE

    x_class = classify_dimension(temp_position[0], x_min, x_max)
    y_class = classify_dimension(temp_position[1], y_min, y_max)
    z_class = classify_dimension(temp_position[2], z_min, z_max)

    classifications = [x_class, y_class, z_class]
    inside_count  = classifications.count(IOStatus.INSIDE)
    surface_count = classifications.count(IOStatus.SURFACE)
    outside_count = classifications.count(IOStatus.OUTSIDE)

    if inside_count == 3:
        return IOStatus.INSIDE, None
    elif inside_count == 2 and surface_count == 1:
        return IOStatus.TEMP_ON_SURFACE, None
    elif inside_count == 1 and surface_count == 2:
        return IOStatus.TEMP_ON_EDGE, None
    elif inside_count == 2 and outside_count == 1:
        return IOStatus.SURFACE_OUT, None
    elif inside_count == 1 and outside_count == 2:
        return IOStatus.SURFACE_OUT, None
    elif inside_count == 0 and surface_count == 2 and outside_count == 1:
        # VERTEX_OUT の場合、頂点座標も返す
        vx, vy, vz = None, None, None
        x, y, z = temp_position
        if x_class == IOStatus.SURFACE:
            vx = x_min if abs(x - x_min) <= limit else x_max
        elif x_class == IOStatus.OUTSIDE:
            vx = x_min if x < x_min - limit else x_max
        if y_class == IOStatus.SURFACE:
            vy = y_min if abs(y - y_min) <= limit else y_max
        elif y_class == IOStatus.OUTSIDE:
            vy = y_min if y < y_min - limit else y_max
        if z_class == IOStatus.SURFACE:
            vz = z_min if abs(z - z_min) <= limit else z_max
        elif z_class == IOStatus.OUTSIDE:
            vz = z_min if z < z_min - limit else z_max
        vertex_coords = np.array([vx, vy, vz], dtype=float)
        return IOStatus.VERTEX_OUT, vertex_coords
    elif (
        (inside_count == 1 and surface_count == 1 and outside_count == 1) or
        (inside_count == 0 and surface_count == 1 and outside_count == 2)
    ):
        return IOStatus.EDGE_OUT, None
    elif inside_count == 0 and surface_count == 0 and outside_count == 3:
        return IOStatus.SURFACE_OUT, None
    elif inside_count == 0 and surface_count == 3 and outside_count == 0:
        return IOStatus.BORDER, None
    else:
        raise ValueError("Unknown inside/surface/outside combination")

def IO_check_spot(base_position, temp_position, constants, IO_status, stick_status=0, _depth=0):

    """Spot 形状における IO 判定。``_depth`` は再帰回数の制御用。"""

    radius   = constants['radius']
    bottom_z = constants['spot_bottom_height']
    bottom_r = constants['spot_bottom_r']
    limit    = constants['limit']

    z_tip = temp_position[2]
    r_tip = LA.norm(temp_position)
    xy_dist = np.sqrt(temp_position[0] ** 2 + temp_position[1] ** 2)

    if z_tip > bottom_z + limit:
        if r_tip > radius + limit:
            return IOStatus.SPHERE_OUT
        if r_tip < radius - limit:
            return IOStatus.TEMP_ON_POLYGON if stick_status > 0 else IOStatus.INSIDE

        if _depth == 0:
            scaled = base_position + (temp_position - base_position) * 1.2
            return IO_check_spot(base_position, scaled, constants, IO_status, stick_status, _depth=1)
        return IOStatus.INSIDE

    if z_tip < bottom_z - limit:
        denom = temp_position[2] - base_position[2]
        t = (bottom_z - base_position[2]) / denom
        if t < 0 or t > 1:
            return IOStatus.SPHERE_OUT
        intersect_xy = base_position[:2] + t * (temp_position[:2] - base_position[:2])
        dist_xy = np.sqrt(intersect_xy[0] ** 2 + intersect_xy[1] ** 2)
        if dist_xy < bottom_r + limit:
            return IOStatus.BOTTOM_OUT
        return IOStatus.SPHERE_OUT

    if bottom_z - limit < z_tip < bottom_z + limit:
        base_xy = np.sqrt(base_position[0] ** 2 + base_position[1] ** 2)
        start_on_border = (
            abs(base_position[2] - bottom_z) <= limit and abs(base_xy - bottom_r) <= limit
        )
        if start_on_border:
            if IO_status == IOStatus.POLYGON_MODE:
                return IOStatus.POLYGON_MODE
            if xy_dist < bottom_r - limit:
                return IOStatus.SPOT_BOTTOM
            if xy_dist > bottom_r + limit:
                return IOStatus.SPOT_EDGE_OUT
            return IOStatus.SPOT_BOTTOM if xy_dist < bottom_r else IOStatus.SPOT_EDGE_OUT
        else:
            if _depth == 0:
                scaled = base_position + (temp_position - base_position) * 1.2
                return IO_check_spot(base_position, scaled, constants, IO_status, stick_status, _depth=1)
            return IOStatus.INSIDE

    return IOStatus.INSIDE

