
import numpy as np
from numpy import linalg as LA
from tools.enums import IOStatus

class SpotMode:
    def __init__(self, constants: dict):
        self.constants = constants

    def drop_polygon_move(
        base_position: np.ndarray,
        last_vec: np.ndarray,
        stick_status: int,
        constants: dict,
        ) -> tuple[np.ndarray, np.ndarray, int, IOStatus]:
            

        step_len = constants['step_length']
        dev_mag = constants['deviation']
        limit = constants['limit']

        # 法線ベクトル（球中心からの放射方向）
        normal = base_position / (LA.norm(base_position) + 1e-12)

        # 接線ベクトルの準備
        vec_norm = LA.norm(last_vec)
        if vec_norm < limit:
            if abs(normal[0]) < 0.9:
                tangent = np.cross(normal, [1.0, 0.0, 0.0])
            else:
                tangent = np.cross(normal, [0.0, 1.0, 0.0])
            v_base = tangent / (LA.norm(tangent) + 1e-12)
        else:
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

        # ポリゴン面上の2軸 (u, v)
        u = v_base
        v = np.cross(normal, u)
        v /= LA.norm(v) + 1e-12

        # 貼り付き状態に応じた deviation の生成
        if stick_status > 1:
            theta = np.random.uniform(-np.pi, np.pi)
            deviation_vec = dev_mag * (np.cos(theta) * u + np.sin(theta) * v)
        elif stick_status == 1:
            theta = np.random.uniform(-np.pi / 2, np.pi / 2)
            deviation_vec = dev_mag * (np.cos(theta) * normal + np.sin(theta) * v_base)
        else:
            rand_vec = np.random.normal(0, 1, 3)
            rand_vec /= LA.norm(rand_vec) + 1e-12
            deviation_vec = dev_mag * rand_vec

        final_dir = v_base + deviation_vec
        final_dir /= LA.norm(final_dir) + 1e-12
        new_last_vec = final_dir * step_len
        new_temp_position = base_position + new_last_vec

        new_stick_status = stick_status - 1 if stick_status > 0 else 0
        next_state = IOStatus.INSIDE if new_stick_status <= 0 else IOStatus.POLYGON_MODE

        return new_temp_position, new_last_vec, new_stick_status, next_state
