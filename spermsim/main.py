import ast
import configparser
import os
import random
import sqlite3
import sys
import time
from datetime import datetime
from enum import Enum

# IO 状態を表す列挙型（独自定義または外部ファイルから）
from io_status import IOStatus

# --- GUIライブラリ ---
import tkinter as tk

# --- 外部ライブラリ ---
import matplotlib
matplotlib.use("Agg")  # GUI 非依存の描画専用モード
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
import numpy.linalg as LA
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# --- 派生変数一元計算モジュールをインポート ---
from tools.derived_constants import calculate_derived_constants

# --- ジオメトリ関連クラス/ファクトリ ---
from .geometry import create_shape  # 必要ならself-import等は調整

# --- データ・画像・動画保存パス等 ---
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = os.getcwd()

DATA_DIR  = os.path.join(_SCRIPT_DIR, "data")
IMG_DIR   = os.path.join(DATA_DIR, "graphs")
MOV_DIR   = os.path.join(DATA_DIR, "movies")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MOV_DIR, exist_ok=True)
DB_PATH_DEFAULT = os.path.join(DATA_DIR, "Trajectory.db")

def _safe_anim_save(anim, output_path):
    """
    ffmpegが無くてもPillowWriterで保存を試みる安全ラッパー
    """
    try:
        anim.save(output_path, writer="ffmpeg", codec="mpeg4", fps=5)
    except Exception as e:
        print(f"[WARN] ffmpeg保存失敗 ({e}) → pillow writerで再試行")
        try:
            anim.save(output_path, writer="pillow", fps=5)
        except Exception as e2:
            print(f"[ERROR] pillow writerでも保存に失敗: {e2}")

# --- 実行環境セットアップ ---
os.chdir(_SCRIPT_DIR)
np.set_printoptions(threshold=np.inf)

def get_program_version():
    """
    スクリプトファイル名をバージョン情報として返す。
    Jupyterや対話モードの場合は 'interactive' として返す。
    """
    try:
        file_name = os.path.basename(__file__)
    except NameError:
        file_name = "interactive"
    version = f"{file_name}"
    return version

def get_constants_from_gui(selected_data, shape, volume, sperm_conc):
    """
    GUI から受け取った選択データと shape, volume, sperm_conc を元に、
    シミュレーション定数をまとめた辞書を作成して返す。
    """
    constants = {}
    constants['shape'] = shape.lower()
    constants['volume'] = float(volume)
    constants['sperm_conc'] = int(sperm_conc)
    constants['spot_angle']        = float(selected_data.get('spot_angle', 60))
    constants['vsl']               = float(selected_data.get('vsl', 0.13))
    constants['deviation']         = float(selected_data.get('deviation', 0.04))
    constants['sampl_rate_hz']     = float(selected_data.get('sampl_rate_hz', 2))
    constants['sim_min']           = float(selected_data.get('sim_min', 10))
    constants['gamete_r']          = float(selected_data.get('gamete_r', 0.15))
    constants['stick_sec']         = int(selected_data.get('stick_sec', 2))
    constants['stick_steps'] = constants['stick_sec'] * constants['sampl_rate_hz']
    constants['step_length'] = constants['vsl'] / constants['sampl_rate_hz']
    constants['limit'] = 1e-10
    egg_localization = selected_data.get('egg_localization', 'bottom_center').strip()
    constants['egg_localization'] = egg_localization
    constants['initial_direction']    = selected_data.get('initial_direction', 'random').strip()
    constants['initial_stick'] = int(selected_data.get('initial_stick', 0))
    constants['seed_number']          = selected_data.get('seed_number', None)
    constants['N_repeat']             = int(selected_data.get('n_repeat', 1))
    outputs = selected_data.get('outputs', [])
    constants['draw_trajectory'] = 'yes' if 'graph' in outputs else 'no'
    constants['make_movie']      = 'yes' if 'movie' in outputs else 'no'
    if constants['seed_number'] and str(constants['seed_number']).lower() != IOStatus.NONE:
        np.random.seed(int(constants['seed_number']))
    constants['analysis_type'] = selected_data.get('analysis_type', "simulation")

    # ★ここで派生値をtools/derived_constants.pyで一元計算
    from tools.derived_constants import calculate_derived_constants
    constants = calculate_derived_constants(constants)

    return constants


def placement_of_eggs(constants):
    """
    shape, egg_localization に応じた卵子の配置座標などを返す。
    """
    shape = constants['shape'].lower()
    egg_localization = constants['egg_localization']
    gamete_r = constants['gamete_r']

    if shape == "cube":
        positions_map = {
            "center":           (0, 0, 0),
            "bottom_center":    (0, 0, constants['z_min'] + gamete_r),
            "bottom_side":      (constants['x_min'] / 2 + gamete_r, constants['y_min'] / 2 + gamete_r, constants['z_min'] + gamete_r),
            "bottom_corner":    (constants['x_min'] + gamete_r, constants['y_min'] + gamete_r, constants['z_min'] + gamete_r),
        }
    elif shape == "drop":
        drop_r = constants['drop_r']
        positions_map = {
            "center":           (0, 0, 0),
            "bottom_center":    (0, 0, -drop_r + gamete_r),
        }
    elif shape == "spot":
        spot_r = constants['spot_r']
        spot_bottom_height = constants['spot_bottom_height']
        positions_map = {
            "center": (
                0,
                0,
                (spot_bottom_height + spot_r) / 2
            ),
            "bottom_center": (
                0,
                0,
                spot_bottom_height + gamete_r
            ),
            "bottom_edge": (
                np.sqrt(
                    (spot_r - gamete_r) ** 2
                    - (spot_bottom_height + gamete_r) ** 2
                ),
                0,
                spot_bottom_height + gamete_r
            )
        }
    elif shape == "ceros":
        center_x = (constants['x_min'] + constants['x_max']) / 2
        center_y = (constants['y_min'] + constants['y_max']) / 2
        center_z = (constants['z_min'] + constants['z_max']) / 2
        positions_map = {
            "center":        (center_x, center_y, center_z),
            "bottom_center": (center_x, center_y, center_z),
            "bottom_edge":   (center_x, center_y, center_z),
        }
    else:
        raise RuntimeError(f"未知の形状 '{shape}' が指定されました。")

    if egg_localization not in positions_map:
        raise RuntimeError(f"指定された egg_localization '{egg_localization}' は、形状 '{shape}' に対して無効です。")

    egg_x, egg_y, egg_z = positions_map[egg_localization]
    e_x_min = egg_x - gamete_r
    e_y_min = egg_y - gamete_r
    e_z_min = egg_z - gamete_r
    e_x_max = egg_x + gamete_r
    e_y_max = egg_y + gamete_r
    e_z_max = egg_z + gamete_r
    egg_center = np.array([egg_x, egg_y, egg_z])
    egg_position_4d = np.array([egg_x, egg_y, egg_z, 0])
    return (
        egg_x, egg_y, egg_z,
        e_x_min, e_y_min, e_z_min,
        e_x_max, e_y_max, e_z_max,
        egg_center, egg_position_4d
    )



################

def get_reflection_initial_positions(constants):
    """
    Reflectionモード用の初期位置・初期ベクトルを返す関数。
    """
    # limitsを一括取得
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)

    # centerは必ずどのshapeでも使うので先に計算
    center = np.array([
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2,
    ])

    shape = constants['shape'].lower()
    if shape == 'cube' or shape == 'ceros':
        base_position = center
    elif shape == 'drop':
        base_position = np.array([0, 0, 0])
    elif shape == 'spot':
        base_position = np.array([0, 0, constants['spot_bottom_height']])
    else:
        base_position = center  # fallback

    # 方向ベクトルもスマートに
    directions = {
        'right': np.array([1, 0, 0]),
        'left': np.array([-1, 0, 0]),
        'up': np.array([0, 1, 0]),
        'down': np.array([0, -1, 0]),
        'forward': np.array([0, 0, 1]),
        'backward': np.array([0, 0, -1]),
        'random': np.random.normal(size=3),
    }
    initial_direction = constants.get('initial_direction', 'random').lower()
    direction_vec = directions.get(initial_direction, directions['random'])
    direction_vec = normalize_vector(direction_vec)
    first_temp = base_position + direction_vec * constants['step_length']
    return base_position, first_temp

def normalize_vector(v):
    norm = LA.norm(v)
    if norm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / norm

# ------------------------------------------------------------
# 共通ヘルパー: 残り移動距離計算  ★ADDED
# ------------------------------------------------------------
def calc_remaining(old_remaining, base_position, hit_point):
    """境界衝突後の残り移動距離を一括計算する共通ルーチン。"""
    return max(old_remaining - np.linalg.norm(hit_point - base_position), 0.0)

def cut_and_bend_cube(self, IO_status, base_position, temp_position, remaining_distance, constants):
    out_flags = []
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    cutting_ratios = {}
    axes = ['x', 'y', 'z']
    min_vals = [x_min, y_min, z_min]
    max_vals = [x_max, y_max, z_max]
    for axis, pos, min_val, max_val in zip(axes, temp_position, min_vals, max_vals):
        idx = axes.index(axis)
        if pos < min_val - constants['limit']:
            out_flags.append(f'{axis}_min_out')
            ratio = (min_val - base_position[idx]) / (temp_position[idx] - base_position[idx])
            cutting_ratios[axis] = ratio
        elif pos > max_val + constants['limit']:
            out_flags.append(f'{axis}_max_out')
            ratio = (max_val - base_position[idx]) / (temp_position[idx] - base_position[idx])
            cutting_ratios[axis] = ratio
    if cutting_ratios:
        first_out_axis = min(cutting_ratios, key=cutting_ratios.get)
        first_out_axis_index = axes.index(first_out_axis)
        cutting_ratio = cutting_ratios[first_out_axis]
        vector_to_be_cut = temp_position - base_position
        vector_to_surface = vector_to_be_cut * cutting_ratio
        intersection_point = base_position + vector_to_surface
        last_vec = temp_position - intersection_point
        if LA.norm(last_vec) < constants['limit']:
            temp_position = base_position + 1.1 * (temp_position - base_position)
            vector_to_be_cut = temp_position - base_position
            vector_to_surface = vector_to_be_cut * cutting_ratio
            intersection_point = base_position + vector_to_surface
            last_vec = temp_position - intersection_point
            if LA.norm(last_vec) < constants['limit']:
                raise RuntimeError("last_vec too small even after scaling")
        last_vec[first_out_axis_index] = 0
        nv = LA.norm(last_vec)
        # dist_to_surface = LA.norm(vector_to_surface)  # ★COMMENTED OUT
        remaining_distance = calc_remaining(remaining_distance, base_position, intersection_point)  # ★ADDED
        # remaining_distance -= dist_to_surface  # ★COMMENTED OUT
        if remaining_distance < 0:
            remaining_distance = 0
            raise RuntimeError("this is ありえない！")
        if nv == 0:
            raise RuntimeError("this is ありえない２２２！")
        else:
            last_vec_adjusted = last_vec / nv * remaining_distance
        temp_position = intersection_point + last_vec_adjusted
    else:
        intersection_point = base_position                                
    return temp_position, intersection_point, remaining_distance


def cut_and_bend_vertex(vertex_point, base_position, remaining_distance, constants):
    dist_to_vertex = np.linalg.norm(vertex_point - base_position)
    move_on_new_edge = remaining_distance - dist_to_vertex
    if move_on_new_edge < 0:
        move_on_new_edge = 0
    cube_center = np.array([0, 0, 0])
    candidate_edges = []
    for i in range(3):
        direction = -1 if vertex_point[i] > cube_center[i] else 1
        edge_vec = np.zeros(3)
        edge_vec[i] = direction
        candidate_edges.append(edge_vec)
    incoming_vec = vertex_point - base_position
    dist_incoming = np.linalg.norm(incoming_vec)
    if dist_incoming == 0:
        incoming_dir = np.zeros(3)
    else:
        incoming_dir = incoming_vec / dist_incoming
    filtered_edges = [
        edge for edge in candidate_edges
        if not (np.allclose(edge, incoming_dir) or np.allclose(edge, -incoming_dir))
    ]
    if filtered_edges:
        new_edge = random.choice(filtered_edges)
    new_temp_position = vertex_point + new_edge * move_on_new_edge
    intersection_point = vertex_point
    new_remaining_distance = constants['vsl'] / constants['sampl_rate_hz']
    return intersection_point, new_temp_position, new_remaining_distance
def cut_and_bend_bottom(self, IO_status, base_position, temp_position, remaining_distance, constants):
    """
    Spot底面との衝突を切り貼りする処理。
    """
    bottom_z = constants['spot_bottom_height']
    ratio = (bottom_z - base_position[2]) / (temp_position[2] - base_position[2])
    vector_to_be_cut = temp_position - base_position
    vector_to_surface = vector_to_be_cut * ratio
    intersection_point = base_position + vector_to_surface
    intersection_point[2] = bottom_z
    vector_to_surface = intersection_point - base_position
    last_vec = temp_position - intersection_point
    if LA.norm(last_vec) < constants['limit']:
        raise RuntimeError("last_vec too small")
    last_vec[2] = 0
    nv = LA.norm(last_vec)
    # remaining_distance -= LA.norm(vector_to_surface)  # ★COMMENTED OUT
    remaining_distance = calc_remaining(remaining_distance, base_position, intersection_point)  # ★ADDED
    if nv < constants['limit']:
        raise RuntimeError("vector finished on the surface: redo")
    else:
        last_vec_adjusted = last_vec / nv * remaining_distance
    threshold = constants['vsl'] / constants['sampl_rate_hz'] * 1e-7
    if LA.norm(last_vec_adjusted) < threshold:
        raise ValueError("last_vec_adjusted is too small; simulation aborted.")
    temp_position = intersection_point + last_vec_adjusted
    return temp_position, intersection_point, remaining_distance

##################
def cut_and_bend_spot_edge_out(self, IO_status, base_position, temp_position,
                                 remaining_distance, constants):
    """
    Spot底面の円周(底面から見たときの外縁)に衝突したときの処理。
    後者の正しく動くコードと同一のものに修正。
    """
    spot_bottom_r = constants['spot_bottom_r']
    inner_angle = constants['inner_angle']
    x0, y0 = base_position[0], base_position[1]
    x1, y1 = temp_position[0], temp_position[1]
    z = constants['spot_bottom_height']
    dx = x1 - x0
    dy = y1 - y0
    A = dx**2 + dy**2
    B = 2 * (x0 * dx + y0 * dy)
    C = x0**2 + y0**2 - spot_bottom_r**2
    discriminant = B**2 - 4*A*C
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-B + sqrt_discriminant) / (2*A)
    t2 = (-B - sqrt_discriminant) / (2*A)
    t_candidates = [t for t in [t1, t2] if 0 <= t <= 1]
    t_intersect = min(t_candidates) if t_candidates else 0
    xi = x0 + t_intersect * dx
    yi = y0 + t_intersect * dy
    intersection_point = np.array([xi, yi, z])
    # distance_to_intersection = LA.norm(intersection_point - base_position)  # ★COMMENTED OUT
    remaining_distance = calc_remaining(remaining_distance, base_position, intersection_point)  # ★ADDED
    # remaining_distance -= distance_to_intersection  # ★COMMENTED OUT
    bi = intersection_point - base_position
    bi_norm = LA.norm(bi)
    if bi_norm < 1e-12:
        bi_norm = 1e-8
    bi_normalized = bi / bi_norm
    oi = np.array([xi, yi, 0])
    oi_norm = LA.norm(oi)
    if oi_norm < 1e-12:
        oi_norm = 1e-8
    oi_normalized = oi / oi_norm
    tangent_1 = np.array([-oi_normalized[1], oi_normalized[0], 0])
    tangent_2 = -tangent_1
    angle_with_tangent_1 = np.arccos(
        np.clip(tangent_1[:2] @ bi_normalized[:2], -1.0, 1.0)
    )
    angle_with_tangent_2 = np.arccos(
        np.clip(tangent_2[:2] @ bi_normalized[:2], -1.0, 1.0)
    )
    if angle_with_tangent_1 < angle_with_tangent_2:
        selected_tangent = tangent_1
    else:
        selected_tangent = tangent_2
    cross = selected_tangent[0]*bi_normalized[1] - selected_tangent[1]*bi_normalized[0]
    if cross > 0:
        angle_adjust = -inner_angle
    else:
        angle_adjust = inner_angle
    def rotate_vector_2d(vec, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        x_new = vec[0]*c - vec[1]*s
        y_new = vec[0]*s + vec[1]*c
        return np.array([x_new, y_new, 0])
    last_vec = rotate_vector_2d(selected_tangent, angle_adjust)
    last_vec /= (LA.norm(last_vec) + 1e-12)
    last_vec = last_vec * remaining_distance
    new_temp_position = intersection_point + last_vec
    new_temp_position[2] = z
    is_bottom_edge = True
    return new_temp_position, intersection_point, remaining_distance, is_bottom_edge
def line_sphere_intersection(base_position, temp_position, radius, remaining_distance, constants):
    d = temp_position - base_position
    d_norm = LA.norm(d)
    if d_norm < constants['limit']:
        raise RuntimeError("too short")
    d_unit = d / d_norm
    f = base_position
    a = 1.0
    b = 2.0 * (f @ d_unit)
    c = (f @ f) - radius ** 2
    discriminant = b**2 - 4*a*c
    if discriminant < constants['limit']:
        return base_position, remaining_distance
    sqrt_discriminant = np.sqrt(discriminant)
    q = -0.5 * (b + np.copysign(sqrt_discriminant, b))
    t1 = q / a
    t2 = c / q if abs(q) > constants['limit'] else np.inf
    t_candidates = [t for t in [t1, t2] if t > constants['limit']]
    if not t_candidates:
        raise RuntimeError("No positive t found for intersection.")
    t = min(t_candidates)
    intersection_point = base_position + t * d_unit
    distance_traveled = t * d_norm
    updated_remaining_distance = remaining_distance - distance_traveled
    return intersection_point, updated_remaining_distance
def compute_normalized_vectors(base_position, intersection_point, constants):
    oi = intersection_point
    oi_normalized = normalize_vector(oi)
    bi = intersection_point - base_position
    bi_norm = LA.norm(bi)
    if bi_norm < constants['limit']:
        print("oi", oi)
        print("LA.norm(bi)", LA.norm(bi))
        raise RuntimeError("Vector bi_norm is zero!")
    bi_normalized = bi / bi_norm
    return oi_normalized, bi_normalized
def determine_rotation_direction(selected_tangent, normal_B, bi_normalized, modify_angle):
    cross_product = (np.cross(selected_tangent, normal_B) @ bi_normalized)
    if cross_product < 0:
        modify_angle = -modify_angle
    return modify_angle
def compute_tangent_vectors(oi_normalized, bi_normalized):
    normal_B = normalize_vector(np.cross(bi_normalized, oi_normalized))
    tangent_1 = normalize_vector(np.cross(normal_B, oi_normalized))
    tangent_2 = -tangent_1
    angle1 = calculate_angle_between_vectors(tangent_1, bi_normalized)
    angle2 = calculate_angle_between_vectors(tangent_2, bi_normalized)
    if angle1 < angle2:
        selected_tangent = tangent_1
    else:
        selected_tangent = tangent_2
    return selected_tangent, normal_B
def calculate_angle_between_vectors(v1, v2):
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    dot_product = np.clip(v1_u @ v2_u, -1.0, 1.0)
    return np.arccos(dot_product)
def rotate_vector(vector, axis, angle):
    axis = normalize_vector(axis)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    return (vector * cos_theta +
            np.cross(axis, vector) * sin_theta +
            (axis @ vector) * (1 - cos_theta))
######
def cut_and_bend_sphere(base_position, remaining_distance, temp_position, constants):
    radius = constants['radius']
    modify_angle = constants['inner_angle']
    intersection_point, remaining_distance = line_sphere_intersection(
        base_position, temp_position, radius, remaining_distance, constants
    )
    oi_normalized, bi_normalized = compute_normalized_vectors(
        base_position, intersection_point, constants
    )
    selected_tangent, normal_B = compute_tangent_vectors(oi_normalized, bi_normalized)
    modify_angle = determine_rotation_direction(
        selected_tangent, normal_B, bi_normalized, modify_angle
    )
    last_vec = rotate_vector(selected_tangent, normal_B, modify_angle)
    last_vec_normalized = normalize_vector(last_vec)
    last_vec = last_vec_normalized * remaining_distance
    new_temp_position = intersection_point + last_vec
    lv_dot = np.dot(last_vec, last_vec)
    if abs(lv_dot) < constants['limit']:
        inward_dir = np.array([0.0, 0.0, 0.0])
    else:
        t = - np.dot(intersection_point, last_vec) / lv_dot
        F = intersection_point + t * last_vec
        inward_dir = -F
        norm_id = LA.norm(inward_dir)
        if norm_id < constants['limit']:
            inward_dir = np.array([0.0,0.0,0.0])
        else:
            inward_dir /= norm_id
    return new_temp_position, intersection_point, remaining_distance, inward_dir
def _calculate_inward_dir_from_axes_hit(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, constants):
    on_x_min = abs(x - x_min) <= constants['limit']
    on_x_max = abs(x - x_max) <= constants['limit']
    on_y_min = abs(y - y_min) <= constants['limit']
    on_y_max = abs(y - y_max) <= constants['limit']
    on_z_min = abs(z - z_min) <= constants['limit']
    on_z_max = abs(z - z_max) <= constants['limit']
    hit_count = sum([on_x_min, on_x_max, on_y_min, on_y_max, on_z_min, on_z_max])
    pinned_coords = []
    free_axes = []
    if on_x_min:
        pinned_coords.append(('x', x_min))
    elif on_x_max:
        pinned_coords.append(('x', x_max))
    else:
        free_axes.append('x')
    if on_y_min:
        pinned_coords.append(('y', y_min))
    elif on_y_max:
        pinned_coords.append(('y', y_max))
    else:
        free_axes.append('y')
    if on_z_min:
        pinned_coords.append(('z', z_min))
    elif on_z_max:
        pinned_coords.append(('z', z_max))
    else:
        free_axes.append('z')
    return pinned_coords, free_axes, hit_count
def face_and_inward_dir(temp_position, base_position, last_vec, IO_status, stick_status, constants):
    if IO_status == IOStatus.INSIDE or IO_status ==IOStatus.TEMP_ON_POLYGON:
        denom = np.dot(last_vec, last_vec)
        if abs(denom) < constants['limit']:
            raise RuntimeError("last_vec is zero or near-zero in face_and_inward_dir")
        t = - np.dot(base_position, last_vec) / denom
        F = base_position + t * last_vec
        if np.linalg.norm(F) <= constants['limit']:
            raise RuntimeError("原点を通るのでredo")
        inward_dir = -F / np.linalg.norm(F)
        return inward_dir
    elif IO_status in [IOStatus.TEMP_ON_EDGE, IOStatus.TEMP_ON_SURFACE]:
        x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
        x, y, z = temp_position
        pinned_coords, free_axes, hit_count = _calculate_inward_dir_from_axes_hit(
            x, y, z, x_min, x_max, y_min, y_max, z_min, z_max, constants
        )
        if hit_count == 1:
            (ax_name, ax_val) = pinned_coords[0]
            if ax_name == 'x':
                if abs(ax_val - x_max) <= constants['limit']:
                    inward_dir = np.array([-1, 0, 0])
                else:
                    inward_dir = np.array([1, 0, 0])
            elif ax_name == 'y':
                if abs(ax_val - y_max) <= constants['limit']:
                    inward_dir = np.array([0, -1, 0])
                else:
                    inward_dir = np.array([0, 1, 0])
            elif ax_name == 'z':
                if abs(ax_val - z_max) <= constants['limit']:
                    inward_dir = np.array([0, 0, -1])
                else:
                    inward_dir = np.array([0, 0, 1])
            else:
                raise RuntimeError("no way - face")
            return inward_dir
        elif hit_count == 2:
            mid_x = mid_y = mid_z = 0.0
            for (ax_name, ax_val) in pinned_coords:
                if ax_name == 'x':
                    mid_x = ax_val
                elif ax_name == 'y':
                    mid_y = ax_val
                elif ax_name == 'z':
                    mid_z = ax_val
            if len(free_axes) == 1:
                fa = free_axes[0]
                if fa == 'x':
                    mid_x = (x_min + x_max) / 2
                elif fa == 'y':
                    mid_y = (y_min + y_max) / 2
                elif fa == 'z':
                    mid_z = (z_min + z_max) / 2
            midpoint_of_edge = np.array([mid_x, mid_y, mid_z], dtype=float)
            direction_vec = -midpoint_of_edge
            norm_dv = np.linalg.norm(direction_vec)
            if norm_dv < constants['limit']:
                raise RuntimeError("no way3")
            inward_dir = direction_vec / norm_dv
            return inward_dir
        else:
            return None
    elif IO_status == IOStatus.VERTEX_OUT:
        return None
    else:
        return None
    ###########
def sample_random_angles(sigma, constants, cone_type="full"):
    max_theta = np.pi
    if cone_type == "quarter":
        min_theta = constants['limit']
        max_theta = np.pi - constants['limit']
        min_phi = -np.pi/4 + constants['limit']
        max_phi = np.pi/4 - constants['limit']
    elif cone_type == "half":
        min_theta = constants['limit']
        max_theta = np.pi - constants['limit']
        min_phi = -np.pi/2 + constants['limit']
        max_phi = np.pi/2 - constants['limit']
    else:
        min_theta = constants['limit']
        max_theta = np.pi - constants['limit']
        min_phi = -np.pi
        max_phi = np.pi
    while True:
        theta = abs(np.random.normal(0, sigma))
        if min_theta < theta < max_theta:
            break
    phi = np.random.uniform(min_phi, max_phi)
    return theta, phi
def make_local_xy(v, inward_dir=None):
    v_norm = LA.norm(v)
    if v_norm < 1e-12:
        v = np.array([0, 0, 1], dtype=float)
    else:
        v = v / v_norm
    if inward_dir is None:
        arbitrary = np.array([1, 0, 0], dtype=float)
        if abs(v[0]) > 0.9:
            arbitrary = np.array([0, 1, 0], dtype=float)
        local_x = np.cross(v, arbitrary)
        lx = LA.norm(local_x)
        if lx < 1e-12:
            local_x = np.array([1, 0, 0], dtype=float)
        else:
            local_x /= lx
    else:
        inward_norm = np.linalg.norm(inward_dir)
        if inward_norm < 1e-12:
            local_x = np.array([1, 0, 0], dtype=float)
        else:
            local_x = inward_dir / inward_norm
        if np.allclose(np.abs(np.dot(local_x, v)), 1.0, atol=1e-12):
            raise ValueError("inward_dir は v と平行でないベクトルを指定してください。")
    local_y = np.cross(v, local_x)
    local_y /= LA.norm(local_y)
    return local_x, local_y
def generate_cone_vector(v, local_x, local_y, inward_dir, constants, sigma, remaining_distance,
                         cone_type='full', do_projection=False):
    theta, phi = sample_random_angles(sigma, constants, cone_type="full")
    x_local = np.sin(theta)*np.cos(phi)
    y_local = np.sin(theta)*np.sin(phi)
    z_local = np.cos(theta)
    temp_dir = x_local*local_x + y_local*local_y + z_local*v
    if do_projection:
        n = inward_dir
        n_norm = LA.norm(n)
        if n_norm > 1e-12:
            n_unit = n / n_norm
            dot_val = np.dot(temp_dir, n_unit)
            temp_dir = temp_dir - dot_val*n_unit
    nd = LA.norm(temp_dir)
    if nd < 1e-12:
        return np.zeros(3)
    temp_dir *= (remaining_distance / nd)
    return temp_dir


def make_local_xy(forward_vec, inward_dir):
    if inward_dir is None:
        # 任意のベースベクトルを使って直交系を作る
        if abs(forward_vec[0]) < 0.9:
            base = np.array([1.0, 0.0, 0.0])
        else:
            base = np.array([0.0, 1.0, 0.0])
        local_y = np.cross(forward_vec, base)
        local_y /= LA.norm(local_y)
        local_x = np.cross(local_y, forward_vec)
    else:
        local_y = np.cross(inward_dir, forward_vec)
        local_y /= LA.norm(local_y)
        local_x = np.cross(local_y, inward_dir)
    return local_x, local_y

def generate_cone_vector(forward, local_x, local_y, inward_dir, constants,
                         sigma, remaining_distance, cone_type, do_projection):
    theta = np.random.normal(0, sigma)
    phi = np.random.uniform(0, 2 * np.pi)

    if cone_type == "half":
        phi = np.abs(phi)  # 上半球に制限
    elif cone_type == "quarter":
        phi = np.abs(phi) / 2  # クォータコーン

    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    dir_vec = dx * local_x + dy * local_y + dz * forward
    dir_vec /= LA.norm(dir_vec)
    vec = dir_vec * remaining_distance

    if do_projection:
        vec = vec - np.dot(vec, inward_dir) * inward_dir  # 面への射影

    return vec

def prepare_new_vector(last_vec, constants,
                       boundary_type="free",
                       stick_status=0,
                       inward_dir=None):
    v_norm = LA.norm(last_vec)
    if v_norm < constants['limit']:
        raise ValueError("prepare_new_vector: last_vec が短すぎます。")
    v = last_vec / v_norm

    if boundary_type == "edge":
        cone_type = "quarter"
        do_proj = False
        if stick_status > 0:
            return v * constants['step_length']
    elif boundary_type in ["surface", "polygon"]:
        cone_type = "half"
        do_proj = (stick_status > 0)
    else:
        cone_type = "full"
        do_proj = False

    local_x, local_y = make_local_xy(v, inward_dir)

    new_vec = generate_cone_vector(
        v, local_x, local_y, inward_dir, constants,
        sigma=constants['deviation'],
        remaining_distance=constants['step_length'],
        cone_type=cone_type,
        do_projection=do_proj
    )
    return new_vec
#########

def IO_check_cube(temp_position, constants):
    x_min, x_max, y_min, y_max, z_min, z_max = get_limits(constants)
    def classify_dimension(pos, min_val, max_val, constants):
        if pos < min_val - constants['limit']:
            return IOStatus.OUTSIDE
        elif pos > max_val + constants['limit']:
            return IOStatus.OUTSIDE
        elif abs(pos - min_val) <= constants['limit'] or abs(pos - max_val) <= constants['limit']:
            return IOStatus.SURFACE
        else:
            return IOStatus.INSIDE
    x_class = classify_dimension(temp_position[0], x_min, x_max, constants)
    y_class = classify_dimension(temp_position[1], y_min, y_max, constants)
    z_class = classify_dimension(temp_position[2], z_min, z_max, constants)
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
        vx, vy, vz = None, None, None
        x, y, z = temp_position
        if x_class == IOStatus.SURFACE:
            if abs(x - x_min) <= constants['limit']:
                vx = x_min
            else:
                vx = x_max
        elif x_class == IOStatus.OUTSIDE:
            if x < x_min - constants['limit']:
                vx = x_min
            else:
                vx = x_max
        if y_class == IOStatus.SURFACE:
            if abs(y - y_min) <= constants['limit']:
                vy = y_min
            else:
                vy = y_max
        elif y_class == IOStatus.OUTSIDE:
            if y < y_min - constants['limit']:
                vy = y_min
            else:
                vy = y_max
        if z_class == IOStatus.SURFACE:
            if abs(z - z_min) <= constants['limit']:
                vz = z_min
            else:
                vz = z_max
        elif z_class == IOStatus.OUTSIDE:
            if z < z_min - constants['limit']:
                vz = z_min
            else:
                vz = z_max
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
def IO_check_drop(temp_position, stick_status, constants):
    distance_from_center = LA.norm(temp_position)
    radius = constants['drop_R']
    if distance_from_center > radius + constants['limit']:
        IO_status = IOStatus.SPHERE_OUT
    elif distance_from_center < radius - constants['limit']:
        if stick_status > 0:
            IO_status = IOStatus.TEMP_ON_POLYGON
        else:
            IO_status = IOStatus.INSIDE
    else:
        IO_status = IOStatus.BORDER
    return IO_status
def IO_check_spot(base_position, temp_position, constants, IO_status):
    """
    Spot形状におけるIO判定。
    後者の「うまくいくプログラム」と同一仕様になるように修正。
    """
    radius   = constants['radius']
    bottom_z = constants['spot_bottom_height']
    bottom_r = constants['spot_bottom_r']
    z_tip = temp_position[2]
    r_tip = LA.norm(temp_position)                    
    xy_dist = np.sqrt(temp_position[0]**2 + temp_position[1]**2)
    if z_tip > bottom_z + constants['limit']:
        if r_tip > radius + constants['limit']:
            return IOStatus.SPHERE_OUT
        else:
#             return "inside"  # ★DEPRECATED
            return IOStatus.INSIDE
    elif z_tip < bottom_z - constants['limit']:
        denom = (temp_position[2] - base_position[2])
        t = (bottom_z - base_position[2]) / denom
        if t < 0 or t > 1:
            return IOStatus.SPHERE_OUT
        intersect_xy = base_position[:2] + t*(temp_position[:2] - base_position[:2])
        dist_xy = np.sqrt(intersect_xy[0]**2 + intersect_xy[1]**2)
        if dist_xy < bottom_r + constants['limit']:
            return IOStatus.BOTTOM_OUT
        else:
            return IOStatus.SPHERE_OUT
    elif bottom_z - constants['limit'] < z_tip < bottom_z + constants['limit']:
        if xy_dist > bottom_r + constants['limit']:
            return IOStatus.SPOT_EDGE_OUT
        elif abs(xy_dist - bottom_r) <= constants['limit']:
            return IOStatus.BORDER
        elif xy_dist < bottom_r - constants['limit']:
            if IO_status in [IOStatus.SPOT_EDGE_OUT, IOStatus.POLYGON_MODE]:
                return IOStatus.POLYGON_MODE
            else:
                return IOStatus.SPOT_BOTTOM
    return IOStatus.INSIDE
class SpermSimulation:
    def initialize_thickness(self):
        for j in range(self.number_of_sperm):
            for i in range(self.number_of_steps):
                self.vec_thickness_2d[j, i] = 0.4
                self.vec_thickness_3d[j, i] = 1.5

    
    def __init__(self, constants, visualizer, simulation_data):

        self.constants = constants  # ✅ 最初に設定

        self.number_of_sperm = self.constants["number_of_sperm"]
        self.number_of_steps = self.constants["number_of_steps"]

        self.vec_thickness_2d = np.zeros((self.number_of_sperm, self.number_of_steps))
        self.vec_thickness_3d = np.zeros((self.number_of_sperm, self.number_of_steps))
        self.initialize_thickness()

        self.visualizer = visualizer
        self.simulation = simulation_data
        self.n_stop = self.constants.get('n_stop', 0)

        if constants.get('reflection_analysis', 'no') == "yes":
            self.initial_stick = constants['initial_stick']
        else:
            self.initial_stick = 0
        self.vec_colors = np.empty(self.number_of_sperm, dtype=object)
        self.vec_thickness_2d = np.zeros((self.number_of_sperm, self.number_of_steps), dtype=float)
        self.vec_thickness_3d = np.zeros((self.number_of_sperm, self.number_of_steps), dtype=float)
        self.trajectory = np.zeros((self.number_of_sperm, self.number_of_steps, 3), dtype=float)
        print(f"[DEBUG] self.trajectory.shape = {self.trajectory.shape}, dtype = {self.trajectory.dtype}")

        self.shape = create_shape(self.constants["shape"], self.constants)

        self.prev_IO_status = [None] * self.number_of_sperm
        self.intersection_records = []
        self.initialize_colors()
        self.initialize_thickness()
        for j in range(self.number_of_sperm):
            base_position, temp_position = self.initial_vec(j, constants)
            print(f"[DEBUG] base_position = {base_position}, type={type(base_position)}, shape={base_position.shape}")
            print(f"[DEBUG] trajectory[j, 0] dtype: {self.trajectory[j, 0].dtype if hasattr(self.trajectory[j, 0], 'dtype') else 'N/A'}")

            self.trajectory[j, 0] = base_position
            if self.number_of_steps > 1:
                self.trajectory[j, 1] = temp_position
        print("初期化時のconstants:", constants)
    def merge_contact_events(self):
        """
        接触イベント（(精子番号, ステップ番号)）をまとめて連続接触を1つに圧縮。
        """
        from collections import defaultdict
        events_by_sperm = defaultdict(list)
        for sperm_index, step in sorted(self.intersection_records, key=lambda x: (x[0], x[1])):
            events_by_sperm[sperm_index].append(step)
        merged_events = []
        for sperm_index, steps in events_by_sperm.items():
            if not steps:
                continue
            start_step = steps[0]
            end_step = steps[0]
            for step in steps[1:]:
                if step == end_step + 1:
                    end_step = step
                else:
                    merged_events.append((sperm_index, start_step))
                    start_step = step
                    end_step = step
            merged_events.append((sperm_index, start_step))
        return merged_events
    def initialize_colors(self):
        base_colors = [
            "#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
            "#98df8a", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d",
            "#9edae5", "#2f4f4f"
        ]
        self.vec_colors = np.empty(self.number_of_sperm, dtype=object)
        for j in range(self.number_of_sperm):
            self.vec_colors[j] = base_colors[j % len(base_colors)]

    
    def initial_vec(self, j, constants):
        shape = self.constants['shape']
        analysis_type = self.constants.get('analysis_type', 'simulation')

        # --- unified base_position for all modes ---
        base_position = self.shape.initial_position()

        if "number_of_sperm" not in self.constants:
            volume = float(self.constants['volume'])
            sperm_conc = float(self.constants['sperm_conc'])
            self.constants['number_of_sperm'] = int(sperm_conc * volume / 1000)

        if "number_of_steps" not in self.constants:
            sim_min = float(self.constants['sim_min'])
            sample_rate = float(self.constants['sampl_rate_hz'])
            steps = int(sim_min * 60 * sample_rate)
            self.constants['number_of_steps'] = steps

   
        if analysis_type == "reflection":
            if shape == "spot":
                spot_bottom_r = self.constants.get('spot_bottom_r', 1.0)
                spot_bottom_height = self.constants.get('spot_bottom_height', 0.5)
                base_position = np.array([
                    spot_bottom_r - constants['step_length'] * 1.5,
                    0.001,
                    spot_bottom_height
                ])
                direction_vec = (constants['step_length'], 0, 0)
                temp_position = base_position + direction_vec

            elif shape == "cube":
                temp_position = base_position + np.array([constants['step_length'], 0, 0])
                IO_status, vertex_point = self.shape.io_check(temp_position)

            elif shape == "drop":
                temp_position = base_position + np.array([constants['step_length'], 0, 0])
                IO_status = self.shape.io_check(temp_position, self.constants['initial_stick'])
                vertex_point = None

            else:
                raise ValueError(f"Unsupported shape for reflection: {shape}")

            local_stick = self.constants['initial_stick']
            if IO_status in [IOStatus.TEMP_ON_SURFACE, IOStatus.TEMP_ON_EDGE]:
                last_vec = temp_position - base_position
                if np.linalg.norm(last_vec) < self.constants['limit']:
                    last_vec = np.array([constants['step_length'], 0.0, 0.0])
                inward_dir = face_and_inward_dir(
                    temp_position,
                    base_position,
                    last_vec,
                    IO_status,
                    local_stick,
                    constants=self.constants
                )
                if inward_dir is None:
                    inward_dir = np.array([0.0, 0.0, 1.0])
                new_vec = prepare_new_vector(
                    last_vec, self.constants,
                    boundary_type=("edge" if IO_status == IOStatus.TEMP_ON_EDGE else "surface"),
                    stick_status=local_stick,
                    inward_dir=inward_dir
                )
                temp_position = base_position + new_vec

        elif analysis_type == "simulation":
            initial_vector = self.get_random_direction_3D() * constants['step_length']
            temp_position = base_position + initial_vector

        else:
            raise ValueError(f"Unknown analysis_type: {analysis_type}")

        return base_position, temp_position
    def set_vector_color(self, j, i, color):
        self.vec_colors[j, i] = color

#######
    def get_random_direction_3D(self):
        phi = np.random.uniform(0, 2*np.pi)
        costheta = np.random.uniform(-1, 1)
        theta = np.arccos(costheta)
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
    def simulate(self):
        step_desc = "シミュレーション中の精子数進捗"
        for j in tqdm(range(self.number_of_sperm), desc=step_desc, ncols=100):
            print(f"[DEBUG] sperm index = {j}")
            base_position = self.trajectory[j, 0]
            temp_position = self.trajectory[j, 1]
            remaining_distance = self.constants['step_length'] 
            self.single_sperm_simulation(
                j, base_position, temp_position,
                remaining_distance, self.constants
            )
    def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
        vector = temp_position - base_position
        if LA.norm(vector) < 1e-9:
            raise RuntimeError("zzz")
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
    def single_sperm_simulation(self, j, base_position, temp_position, remaining_distance, constants):
        if constants['analysis_type'] == "reflection":
            stick_status = self.initial_stick
        else:
            stick_status = 0
        self.trajectory[j, 0] = base_position
        i = 1
        intersection_point = np.array([])
        shape = constants['shape']
        gamete_r = constants['gamete_r']
        if shape != "ceros":
            (egg_x, egg_y, egg_z,
            e_x_min, e_y_min, e_z_min,
            e_x_max, e_y_max, e_z_max,
            egg_center, egg_position_4d) = placement_of_eggs(constants)
        else:
            egg_center = np.array([np.inf, np.inf, np.inf])
            gamete_r = constants['gamete_r']
        if self.n_stop is not None and not np.isnan(self.n_stop):
            max_steps = int(self.n_stop)
        else:
            max_steps = self.number_of_steps
        while i < self.number_of_steps:
            max_safety_loops = 1000  # 無限ループ防止

            if shape in ["cube", "ceros"]:
                new_IO_status, vertex_point = IO_check_cube(temp_position, constants)
            elif shape == "drop":
                new_IO_status = IO_check_drop(temp_position, stick_status, constants)
                vertex_point = None
                if new_IO_status == IOStatus.BORDER:
                    vec = temp_position - base_position
                    vec_length = np.linalg.norm(vec)
                    if vec_length > constants['limit']:
                        adjusted_vec = vec * 0.99
                        temp_position = base_position + adjusted_vec
                        new_IO_status = IO_check_drop(temp_position, stick_status, constants)
                    if new_IO_status == IOStatus.BORDER:
                        raise RuntimeError("drop: rethink logic for border")
            elif shape == "spot":
                prev_stat = self.prev_IO_status[j]
                if prev_stat is None:
                    prev_stat = "none"
                new_IO_status = IO_check_spot(base_position, temp_position, constants, prev_stat)
                vertex_point = None
                if new_IO_status == IOStatus.BORDER:
                    vec = temp_position - base_position
                    vec_length = np.linalg.norm(vec)
                    if vec_length > constants['limit']:
                        adjusted_vec = vec * 0.99
                        temp_position = base_position + adjusted_vec
                        new_IO_status = IO_check_spot(base_position, temp_position, constants, prev_stat)
                    if new_IO_status == IOStatus.BORDER:
                        raise RuntimeError("rethink logic 3")
            else:
                new_IO_status = "inside"
                vertex_point = None
            prev_stat = self.prev_IO_status[j]
            if prev_stat in [IOStatus.TEMP_ON_EDGE, IOStatus.TEMP_ON_SURFACE] and (stick_status > 0):
                if new_IO_status in [
                    IOStatus.INSIDE,
                    IOStatus.TEMP_ON_POLYGON,
                    IOStatus.TEMP_ON_SURFACE,
                    IOStatus.TEMP_ON_EDGE,
                    IOStatus.SPOT_BOTTOM,
            ]:
                    new_IO_status = prev_stat
                    vertex_point = None
            IO_status = new_IO_status
            self.prev_IO_status[j] = IO_status
            if remaining_distance < 0:
                raise RuntimeError("rd<0")
            if IO_status in [
                IOStatus.INSIDE,
                IOStatus.TEMP_ON_SURFACE,
                IOStatus.TEMP_ON_EDGE,
                IOStatus.SPOT_BOTTOM,
                IOStatus.ON_EDGE_BOTTOM,
                IOStatus.TEMP_ON_POLYGON,
        ]:
                self.trajectory[j, i] = temp_position
                base_position = self.trajectory[j, i]
                remaining_distance = constants['step_length']
                if stick_status > 0:
                    stick_status -= 1
                if len(intersection_point) != 0:
                    last_vec = temp_position - intersection_point
                    intersection_point = np.array([])
                else:
                    last_vec = self.trajectory[j, i] - self.trajectory[j, i - 1]
                if self.is_vector_meeting_egg(self.trajectory[j, i - 1], temp_position, egg_center, gamete_r):
                    self.intersection_records.append((j, i))
                    self.vec_colors[j, i - 1] = "red"
                    self.vec_thickness_2d[j, i - 1] = 2.0
                    self.vec_thickness_3d[j, i - 1] = 4.0
                if IO_status == IOStatus.TEMP_ON_EDGE:
                    inward_dir = face_and_inward_dir(
                        temp_position, base_position, last_vec, IO_status, stick_status, constants
                    )
                    if inward_dir is None:
                        inward_dir = np.array([0, 0, 1], dtype=float)
                    if stick_status > 0:
                        temp_position = base_position + last_vec
                    else:
                        new_vec = prepare_new_vector(
                            last_vec, constants,
                            boundary_type="edge",
                            stick_status=stick_status,
                            inward_dir=inward_dir
                        )
                        temp_position = base_position + new_vec
                elif IO_status == IOStatus.TEMP_ON_SURFACE:
                    inward_dir = face_and_inward_dir(
                        temp_position, base_position, last_vec, IO_status, stick_status, constants
                    )
                    if inward_dir is None:
                        inward_dir = np.array([0, 0, 1], dtype=float)
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="surface",
                        stick_status=stick_status,
                        inward_dir=inward_dir
                    )
                    temp_position = base_position + new_vec
                elif IO_status == IOStatus.SPOT_BOTTOM:
                    inward_dir = [0, 0, 1]
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="surface",
                        stick_status=stick_status,
                        inward_dir=inward_dir
                    )
                    temp_position = base_position + new_vec
                elif IO_status == IOStatus.ON_EDGE_BOTTOM:
                    raise RuntimeError("ありえるのか？")
                elif IO_status == IOStatus.TEMP_ON_POLYGON:
                    inward_dir = face_and_inward_dir(
                        temp_position, base_position, last_vec, IO_status, stick_status, constants
                    )
                    if inward_dir is None:
                        inward_dir = np.array([0, 0, 1], dtype=float)
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="polygon",
                        stick_status=stick_status,
                        inward_dir=inward_dir
                    )
                    temp_position = base_position + new_vec
                else:            
                    self.trajectory[j, i] = temp_position
                    base_position = self.trajectory[j, i]
                    remaining_distance = constants['step_length']
                    if stick_status > 0:
                        stick_status -= 1
                    if len(intersection_point) != 0:
                        last_vec = temp_position - intersection_point
                        intersection_point = np.array([])
                    else:
                        last_vec = self.trajectory[j, i] - self.trajectory[j, i - 1]
                    if LA.norm(last_vec) < constants['limit']:
                        raise RuntimeError("last vec is too short!")
                    new_vec = prepare_new_vector(
                        last_vec, constants,
                        boundary_type="free",
                        stick_status=stick_status,
                        inward_dir=None
                    )
                    temp_position = self.trajectory[j, i] + new_vec
                i += 1
                continue
            elif IO_status == IOStatus.SPHERE_OUT:
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_hz'])
                new_temp_pos, intersection_point, remaining_dist, inward_dir = cut_and_bend_sphere(
                    self.trajectory[j, i - 1],
                    remaining_distance,
                    temp_position,
                    constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue
            elif IO_status == IOStatus.POLYGON_MODE:
                self.trajectory[j, i] = temp_position
                base_position = self.trajectory[j, i]
                if len(intersection_point) != 0:
                    last_vec = temp_position - intersection_point
                    intersection_point = np.array([])
                else:
                    last_vec = self.trajectory[j, i] - self.trajectory[j, i - 1]
                new_temp_position, new_last_vec, updated_stick, next_state = self.bottom_edge_mode(
                    base_position, last_vec, stick_status, constants
                )
                temp_position = new_temp_position
                last_vec = new_last_vec
                stick_status = updated_stick
                i += 1
                IO_status = next_state
                continue
            elif IO_status == IOStatus.SPOT_EDGE_OUT:
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_hz'])
                (new_temp_pos,
                 intersection_point,
                 remaining_distance,
                 is_bottom_edge) = cut_and_bend_spot_edge_out(
                     self, IO_status, base_position, temp_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue
            elif IO_status == IOStatus.BOTTOM_OUT:
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_hz'])
                (new_temp_pos,
                 intersection_point,
                 remaining_distance) = cut_and_bend_bottom(
                     self, IO_status, base_position, temp_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue
            elif IO_status in [IOStatus.SURFACE_OUT, IOStatus.EDGE_OUT]:
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_hz'])
                (new_temp_pos,
                 intersection_point,
                 remaining_distance) = cut_and_bend_cube(
                     self, IO_status, base_position, temp_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue
            elif IO_status == IOStatus.VERTEX_OUT:
                if stick_status == 0:
                    stick_status = int(constants['stick_sec'] * constants['sampl_rate_hz'])
                (intersection_point,
                 new_temp_pos,
                 remaining_distance) = cut_and_bend_vertex(
                     vertex_point, base_position, remaining_distance, constants
                )
                base_position = intersection_point
                temp_position = new_temp_pos
                last_vec = temp_position - intersection_point
                continue
    def bottom_edge_mode(self, base_position, last_vec, stick_status, constants):
        """
        底面を這いつつ底面の円周に当たった時の処理。
        後者コードを反映して修正。 polygon_mode から呼ばれる。
        """
        z_floor = constants['spot_bottom_height']
        r_edge  = constants['spot_bottom_r']
        R_spot  = constants['radius']
        candidate_position = base_position + last_vec
        candidate_position[2] = z_floor
        dist2 = candidate_position[0]**2 + candidate_position[1]**2
        radius2 = r_edge**2
        if dist2 > radius2:
            x0, y0 = base_position[0], base_position[1]
            x1, y1 = candidate_position[0], candidate_position[1]
            dx = x1 - x0
            dy = y1 - y0
            A = dx**2 + dy**2
            B = 2*(x0*dx + y0*dy)
            C = x0**2 + y0**2 - r_edge**2
            discriminant = B**2 - 4*A*C
            if discriminant < 0:
                discriminant = 0
            sqrt_discriminant = np.sqrt(discriminant)
            t1 = (-B + sqrt_discriminant)/(2*A)
            t2 = (-B - sqrt_discriminant)/(2*A)
            t_candidates = [t for t in [t1,t2] if 0<=t<=1]
            t_intersect = min(t_candidates) if t_candidates else 0.0
            xi = x0 + t_intersect*dx
            yi = y0 + t_intersect*dy
            intersection_point = np.array([xi, yi, z_floor], dtype=float)
            distance_to_intersection = np.linalg.norm(intersection_point - base_position)
            new_remaining = np.linalg.norm(last_vec) - distance_to_intersection
            if new_remaining < 0:
                new_remaining = 0
            if stick_status > 0:
                bi = intersection_point - base_position
                bi_norm = np.linalg.norm(bi)
                if bi_norm < constants['limit']:
                    bi_norm = 1e-8
                bi_normalized = bi / bi_norm
                oi = np.array([xi, yi, 0.0])
                oi_norm = np.linalg.norm(oi)
                if oi_norm < constants['limit']:
                    oi_norm = 1e-8
                oi_normalized = oi / oi_norm
                tangent_1 = np.array([-oi_normalized[1], oi_normalized[0], 0])
                tangent_2 = -tangent_1
                angle_with_t1 = np.arccos(
                    np.clip(tangent_1[:2] @ bi_normalized[:2], -1.0,1.0)
                )
                angle_with_t2 = np.arccos(
                    np.clip(tangent_2[:2] @ bi_normalized[:2], -1.0,1.0)
                )
                if angle_with_t1 < angle_with_t2:
                    selected_tangent = tangent_1
                else:
                    selected_tangent = tangent_2
                cross_val = (selected_tangent[0]*bi_normalized[1]
                             - selected_tangent[1]*bi_normalized[0])
                modify_angle = constants['inner_angle']
                if cross_val > 0:
                    angle_adjust = -modify_angle
                else:
                    angle_adjust = modify_angle
                def rotate_vector_2d(vec, angle):
                    c = np.cos(angle)
                    s = np.sin(angle)
                    x_new = vec[0]*c - vec[1]*s
                    y_new = vec[0]*s + vec[1]*c
                    return np.array([x_new,y_new,0])
                new_tangent = rotate_vector_2d(selected_tangent, angle_adjust)
                norm_tan = np.linalg.norm(new_tangent)
                if norm_tan < constants['limit']:
                    new_tangent = selected_tangent
                    norm_tan = np.linalg.norm(new_tangent)
                new_tangent /= norm_tan
                last_vec_corrected = new_tangent * new_remaining
                new_temp_position = intersection_point + last_vec_corrected
                new_temp_position[2] = z_floor
                new_last_vec = new_temp_position - intersection_point
            else:
                new_remaining = np.linalg.norm(last_vec)
                sphere_normal_3d = intersection_point
                norm_sphere = np.linalg.norm(sphere_normal_3d)
                if norm_sphere < constants['limit']:
                    sphere_normal_3d = np.array([0,0,1])
                    norm_sphere = 1.0
                sphere_normal_3d /= norm_sphere
                plane_normal = np.array([0,0,1], dtype=float)
                dot_val = np.clip(np.dot(sphere_normal_3d, plane_normal), -1, 1)
                angle_plane_sphere = np.arccos(dot_val)
                def sample_vector_in_cone(axis, max_angle):
                    cos_max = np.cos(max_angle)
                    z_ = np.random.uniform(cos_max, 1.0)
                    phi_ = np.random.uniform(0, 2*np.pi)
                    sqrt_part = np.sqrt(1 - z_*z_)
                    x_local = sqrt_part * np.cos(phi_)
                    y_local = sqrt_part * np.sin(phi_)
                    z_local = z_
                    rot = R.align_vectors([[0,0,1]], [axis])[0]
                    v_local = np.array([x_local, y_local, z_local])
                    return rot.apply(v_local)
                center_axis = (plane_normal + sphere_normal_3d) / 2
                center_axis_norm = np.linalg.norm(center_axis)
                if center_axis_norm < constants['limit']:
                    center_axis = plane_normal
                    center_axis_norm = 1.0
                center_axis /= center_axis_norm
                open_angle = angle_plane_sphere
                random_3d_dir = sample_vector_in_cone(center_axis, open_angle)
                last_vec_corrected = random_3d_dir * new_remaining
                new_temp_position = intersection_point + last_vec_corrected
                new_last_vec = new_temp_position - intersection_point
        else:
            new_temp_position = candidate_position
            new_last_vec = last_vec
        new_stick_status = stick_status
        if new_stick_status > 0:
            new_stick_status -= 1
        if new_stick_status <= 0:
            new_state = IOStatus.INSIDE
        else:
            new_state = IOStatus.BOTTOM_EDGE_MODE
        return new_temp_position, new_last_vec, new_stick_status, new_state
class SpermPlot:
    already_saved_global_flag = False

    def __init__(self, simulation):
        self.simulation = simulation
        self.constants  = self.simulation.constants
        self.colors     = self.simulation.vec_colors

        keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
        missing = [k for k in keys if self.constants.get(k) is None]
        if missing:
            raise RuntimeError(
                f"constantsに範囲値未セット: {missing} → tools/derived_constants.pyを修正"
            )
        # 一括代入
        (self.x_min, self.x_max,
         self.y_min, self.y_max,
         self.z_min, self.z_max) = (self.constants[k] for k in keys)

    def set_ax_3D(self, ax):
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.z_min, self.z_max)
        ax.set_box_aspect([
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        ])

    def _draw_graph(self, shape):
        plt.close('all')
        plt.rcdefaults()
        if hasattr(self, "already_saved") and self.already_saved:
            return None

        # サブプロットの構成
        if shape == "ceros":
            fig, ax_single = plt.subplots(figsize=(4, 4), dpi=300)
            axes = [ax_single]
            axis_combi = [('X', 'Y', 0)]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
            axis_combi = [('X', 'Y', 0), ('X', 'Z', 1), ('Y', 'Z', 2)]

        index_map = {'X': 0, 'Y': 1, 'Z': 2}

        # 卵子の可視化
        if shape != "ceros":
            egg_constants = self.constants.copy()
            egg_x, egg_y, egg_z, *_ , egg_center, _ = placement_of_eggs(egg_constants)
            for ax, (x, y) in zip(axes, [(egg_x, egg_y), (egg_x, egg_z), (egg_y, egg_z)]):
                ax.add_patch(
                    patches.Circle(
                        (x, y),
                        radius=self.constants['gamete_r'],
                        facecolor='yellow', alpha=0.8, ec='gray', linewidth=1.0
                    )
                )
            self.draw_motion_area(shape, axes, self.constants)

        # 軌跡描画
        pbar = tqdm(
            total=self.simulation.number_of_sperm * (self.simulation.number_of_steps - 1) * len(axis_combi),
            desc="Plotting trajectories", ncols=100, ascii=True
        )
        for j in range(self.simulation.number_of_sperm):
            for i in range(self.simulation.number_of_steps - 1):
                for axis1, axis2, idx in axis_combi:
                    axes[idx].plot(
                        self.simulation.trajectory[j, i:i+2, index_map[axis1]],
                        self.simulation.trajectory[j, i:i+2, index_map[axis2]],
                        color=self.simulation.vec_colors[j, i],
                        linewidth=self.simulation.vec_thickness_2d[j, i]
                    )
                    pbar.update(1)
        pbar.close()

        # --- 描画範囲・アスペクト比をconstantsから統一設定 ---
        if shape == "ceros":
            ax_single.set_xlim(self.constants['x_min'], self.constants['x_max'])
            ax_single.set_ylim(self.constants['y_min'], self.constants['y_max'])
            ax_single.set_aspect('equal', adjustable='box')
        else:
            for idx, ax in enumerate(axes):
                # 軸ごとに範囲をセット
                if idx == 0:  # X-Y
                    ax.set_xlim(self.constants['x_min'], self.constants['x_max'])
                    ax.set_ylim(self.constants['y_min'], self.constants['y_max'])
                elif idx == 1:  # X-Z
                    ax.set_xlim(self.constants['x_min'], self.constants['x_max'])
                    ax.set_ylim(self.constants['z_min'], self.constants['z_max'])
                elif idx == 2:  # Y-Z
                    ax.set_xlim(self.constants['y_min'], self.constants['y_max'])
                    ax.set_ylim(self.constants['z_min'], self.constants['z_max'])
                ax.set_aspect('equal', adjustable='box')
                ax.set_anchor('C')

        # --- 軸ラベル ---
        for axis1, axis2, idx in axis_combi:
            axes[idx].set_xlabel(f"{axis1}")
            axes[idx].set_ylabel(f"{axis2}")

        # --- タイトル ---
        if shape in ("cube", "drop", "spot"):
            merged_events = self.simulation.merge_contact_events()
            contacts_per_hour = len(merged_events) / (self.constants['sim_min'] / 60)
            title_str = (
                f"vol: {self.constants['volume']} μl, "
                f"conc: {self.constants['sperm_conc']}/ml, "
                f"vsl: {self.constants['vsl']} mm, "
                f"sampling: {self.constants['sampl_rate_hz']} Hz, "
                f"dev: {self.constants['deviation']}, "
                f"stick: {self.constants['stick_sec']} sec,\n"
                f"sperm/egg interaction: {len(merged_events)} during {self.constants['sim_min']} min, "
                f"egg: {self.constants['egg_localization']}, "
            )
            if shape == "spot":
                title_str += (
                    f"spot_angle: {self.constants.get('spot_angle', 'N/A')} degree"
                )
            fig.suptitle(title_str, fontsize=8, y=0.98)
        n_title_lines  = fig._suptitle.get_text().count("\n") + 1
        top_margin     = max(0.92 - 0.03 * (n_title_lines - 1), 0.80)
        fig.tight_layout(rect=[0.00, 0.00, 1.00, top_margin])

        # --- 保存処理 ---
        out_dir = IMG_DIR
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"graph_output_{ts}.svg"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, format='svg', dpi=300, bbox_inches='tight')
        print(out_path)
        self.already_saved = True
        plt.close(fig)
        return out_path

    #####
    def draw_motion_area(self, shape, axes, constants):
        if shape == 'spot':
            spot_bottom_radius = constants['spot_bottom_r']
            spot_r = constants['spot_r']
            spot_bottom_height = constants['spot_bottom_height']
            axes[0].add_patch(patches.Circle((0, 0), spot_bottom_radius, ec='none', facecolor='red', alpha=0.1))
            for ax in axes[1:]:
                ax.add_patch(patches.Circle((0, 0), spot_r, ec='none', facecolor='red', alpha=0.1))
                ax.axhline(spot_bottom_height, color='gray', linestyle='--', linewidth=0.8)
        elif shape == 'drop':
            drop_R = constants.get('drop_R', constants['radius'])
            for ax in axes:
                ax.add_patch(patches.Circle((0, 0), drop_R, ec='none', facecolor='red', alpha=0.1))
        elif shape == 'cube':
            pass
def draw_motion_area(self, shape, axes, constants):
        if shape == 'spot':
            spot_bottom_radius = constants['spot_bottom_r']
            spot_r = constants['spot_r']
            spot_bottom_height = constants['spot_bottom_height']
            axes[0].add_patch(patches.Circle((0, 0), spot_bottom_radius, ec='none', facecolor='red', alpha=0.1))
            for ax in axes[1:]:
                ax.add_patch(patches.Circle((0, 0), spot_r, ec='none', facecolor='red', alpha=0.1))
                ax.axhline(spot_bottom_height, color='gray', linestyle='--', linewidth=0.8)
        elif shape == 'drop':
            drop_R = constants.get('drop_R', constants['radius'])
            for ax in axes:
                ax.add_patch(patches.Circle((0, 0), drop_R, ec='none', facecolor='red', alpha=0.1))
        elif shape == 'cube':
            pass
class SpermTrajectoryVisualizer:
    def __init__(self, simulation):
        self.simulation = simulation
        self.constants = self.simulation.constants
        self.sperm_plot = SpermPlot(self.simulation)
        (
            egg_x, egg_y, egg_z,
            e_x_min, e_y_min, e_z_min,
            e_x_max, e_y_max, e_z_max,
            egg_center, egg_position_4d
        ) = placement_of_eggs(self.constants)
        self.egg_center = np.array([egg_x, egg_y, egg_z])
        self.egg_radius = self.constants['gamete_r']
    def animate_trajectory(self):
        if self.constants.get("make_movie", "no").lower() != "yes":
            return None
        shape = self.constants.get("shape", "spot")
        num_sperm = self.simulation.number_of_sperm
        n_sim = self.simulation.number_of_steps
        if shape == "ceros":
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlim(-0.815, 0.815)
            ax.set_ylim(-0.62, 0.62)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("CEROS 2D Animation (Zoomed)")
            lines = [ax.plot([], [], lw=2)[0] for _ in range(num_sperm)]
            def init():
                for line in lines:
                    line.set_data([], [])
                return lines
            def animate(i):
                if i % 10 == 0:
                    percentage = (i / (n_sim - 1)) * 100
                    print(f"Progress: {percentage:.2f}%")
                for j, line in enumerate(lines):
                    base_pos = self.simulation.trajectory[j, i]
                    end_pos  = self.simulation.trajectory[j, i + 1]
                    xdata = [base_pos[0], end_pos[0]]
                    ydata = [base_pos[1], end_pos[1]]
                    line.set_data(xdata, ydata)
                    line.set_color(self.simulation.vec_colors[j, i])
                    line.set_linewidth(self.simulation.vec_thickness_3d[j, i])
                return lines
            anim = FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=n_sim - 1,
                interval=180,
                blit=False
            )
            output_folder = MOV_DIR                                             
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mov_filename = f"sperm_simulation_ceros_{timestamp}.mp4"
            output_path = os.path.join(output_folder, mov_filename)
            _safe_anim_save(anim, output_path)                                        
            print(f"{output_path}")
            plt.show()
            return output_path
        else:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            merged_events = self.simulation.merge_contact_events()
            contacts_count = len(merged_events)
            if self.constants["sim_min"] > 0:
                contacts_per_hour = contacts_count / (self.constants["sim_min"] / 60)
            else:
                contacts_per_hour = 0
            title_str_3d = (
                f"vol: {self.constants['volume']} μl, "
                f"conc: {self.constants['sperm_conc']}/ml, "
                f"vsl: {self.constants['vsl']} mm, "
                f"sampling: {self.constants['sampl_rate_hz']} Hz, "
                f"dev: {self.constants['deviation']}, "
                f"stick: {self.constants['stick_sec']} sec,\n"
                f"sperm/egg interaction: {contacts_count} during {self.constants['sim_min']} min, "
                f"egg: {self.constants['egg_localization']}, "
            )
            if shape == "spot":
                title_str_3d += f"spot_angle: {self.constants.get('spot_angle', 'N/A')} degree"
            fig.suptitle(title_str_3d, fontsize=8, y=0.93)
            egg_u = np.linspace(0, 2 * np.pi, 50)
            egg_v = np.linspace(0, np.pi, 50)
            ex = (
                self.egg_center[0]
                + self.egg_radius * np.outer(np.cos(egg_u), np.sin(egg_v))
            )
            ey = (
                self.egg_center[1]
                + self.egg_radius * np.outer(np.sin(egg_u), np.sin(egg_v))
            )
            ez = (
                self.egg_center[2]
                + self.egg_radius * np.outer(
                    np.ones(np.size(egg_u)), np.cos(egg_v)
                )
            )
            ax.plot_surface(ex, ey, ez, color='yellow', alpha=0.2)
            if shape == "spot":
                spot_r = self.constants.get('spot_r', 5)
                spot_angle_deg = self.constants.get('spot_angle', 60)
                shape_u = np.linspace(0, 2*np.pi, 60)
                theta_max_rad = np.deg2rad(spot_angle_deg)
                shape_v = np.linspace(0, theta_max_rad, 60)
                sx = spot_r * np.outer(np.sin(shape_v), np.cos(shape_u))
                sy = spot_r * np.outer(np.sin(shape_v), np.sin(shape_u))
                sz = spot_r * np.outer(np.cos(shape_v), np.ones(np.size(shape_u)))
                ax.plot_surface(sx, sy, sz, color='red', alpha=0.15)
            elif shape == "drop":
                drop_R = self.constants.get('drop_R', 5)
                shape_u = np.linspace(0, 2*np.pi, 60)
                shape_v = np.linspace(0, np.pi, 60)
                sx = drop_R * np.outer(np.sin(shape_v), np.cos(shape_u))
                sy = drop_R * np.outer(np.sin(shape_v), np.sin(shape_u))
                sz = drop_R * np.outer(np.cos(shape_v), np.ones(np.size(shape_u)))
                ax.plot_surface(sx, sy, sz, color='red', alpha=0.15)
            lines = [ax.plot([], [], [], lw=2)[0] for _ in range(num_sperm)]
            def init():
                for line in lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
                return lines
            def animate(i):
                if i % 10 == 0:
                    percentage = (i / (n_sim - 1)) * 100
                    print(f"Progress: {percentage:.2f}%")
                for j, line in enumerate(lines):
                    base_pos = self.simulation.trajectory[j, i]
                    end_pos = self.simulation.trajectory[j, i + 1]
                    line.set_data(
                        [base_pos[0], end_pos[0]],
                        [base_pos[1], end_pos[1]]
                    )
                    line.set_3d_properties([base_pos[2], end_pos[2]])
                    line.set_color(self.simulation.vec_colors[j, i])
                    line.set_linewidth(self.simulation.vec_thickness_3d[j, i])
                return lines
            self.sperm_plot.set_min_max(self.constants.get('volume', 1))
            self.sperm_plot.set_ax_3D(ax)
            anim = FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=n_sim - 1,
                interval=180,
                blit=False
            )
            output_folder = MOV_DIR                                             
            os.makedirs(output_folder, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mov_filename = f"sperm_simulation_{timestamp}.mp4"
            output_path = os.path.join(output_folder, mov_filename)
            _safe_anim_save(anim, output_path)                                        
            print(f"{output_path}")
            plt.show()
            return output_path
def setup_database(conn):
    """
    Create required tables if they do not exist.
    (DDL 文を dict にまとめ可読性を向上)
    """
    c = conn.cursor()                                # ★ADDED

    SQL_DDL = {                                      # ★ADDED
        "basic_data": '''
            CREATE TABLE IF NOT EXISTS basic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id          TEXT,
                version         TEXT,
                shape           TEXT,
                volume          REAL,
                sperm_conc      INTEGER,
                N_contact       INTEGER,
                vsl             REAL,
                stick_sec       INTEGER,
                sim_min         REAL,
                deviation       REAL,
                egg_localization TEXT,
                image_id        TEXT,
                mov_id          TEXT,
                spot_angle      INTEGER,
                sampl_rate_hz   INTEGER,
                seed_number     TEXT
            )
        ''',
        "intersection": '''
            CREATE TABLE IF NOT EXISTS intersection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id  INTEGER,
                sperm_index    INTEGER,
                start_step     INTEGER
            )
        ''',
        "summary": '''
            CREATE TABLE IF NOT EXISTS summary (
                simulation_id    INTEGER,
                shape            TEXT,
                sperm_conc       INTEGER,
                volume           REAL,
                vsl              REAL,
                deviation        REAL,
                sim_min          REAL,
                stick_sec        INTEGER,
                spot_angle       INTEGER,
                sampl_rate_hz    INTEGER,
                egg_localization TEXT,
                mean_contact_hr  REAL,
                SD1              REAL,
                N_sperm          INTEGER,
                C_per_N          REAL,
                SD2              REAL,
                total_simulations INTEGER
            )
        '''
    }                                                # ★ADDED

    for ddl in SQL_DDL.values():                     # ★ADDED
        c.execute(ddl)                              # ★ADDED

    conn.commit()                                    # ★ADDED
def insert_sim_record(conn, exp_id, version, constants, image_id, mov_id, contact_count):
    """
    basic_data テーブルに 1 件分のシミュレーション結果を挿入し、
    挿入した行の simulation_id を返す。
    既存仕様互換のため、dict も同じ値で処理する。
    """
    c = conn.cursor()

    # ★MODIFIED: 可読性向上のためフィールドと値をリスト化
    fields = [
        "exp_id", "version", "shape", "volume", "sperm_conc", "N_contact",
        "vsl", "stick_sec", "sim_min", "deviation", "egg_localization",
        "image_id", "mov_id", "spot_angle", "sampl_rate_hz", "seed_number"
    ]

    # 値の組み立て（spot_angle は spot 形状以外では None）
    values = [
        exp_id,
        version,
        constants['shape'],
        constants['volume'],
        int(constants['sperm_conc']),
        contact_count,
        constants['vsl'],
        int(constants['stick_sec']),
        constants['sim_min'],
        constants['deviation'],
        constants.get('egg_localization', 'bottom_center').strip(),
        image_id,
        mov_id,
        int(constants['spot_angle']) if constants['shape'] == 'spot' else None,
        int(constants['sampl_rate_hz']),
        constants.get('seed_number', 'None'),
    ]

    placeholders = ",".join("?" * len(fields))
    sql = f"INSERT INTO basic_data ({','.join(fields)}) VALUES ({placeholders})"
    c.execute(sql, values)
    conn.commit()

    # 挿入された行の simulation_id を返す
    return c.lastrowid
def insert_intersection_records(conn, simulation_id, merged_events):
    c = conn.cursor()
    for (sperm_index, start_step) in merged_events:
        c.execute('''
            INSERT INTO intersection (simulation_id, sperm_index, start_step)
            VALUES (?, ?, ?)
        ''', (simulation_id, sperm_index, start_step))
    conn.commit()
def aggregate_results(conn, exp_id):
    c = conn.cursor()
    c.execute('''
        SELECT id, shape, sperm_conc, volume, vsl, deviation,
               sim_min, stick_sec, spot_angle, sampl_rate_hz,
               egg_localization, N_contact
          FROM basic_data
    ''')
    rows = c.fetchall()
    import statistics
    from collections import defaultdict
    grouping = defaultdict(list)
    for rec in rows:
        sim_id            = rec[0]
        shape_val         = rec[1]
        sperm_conc_val    = rec[2]
        volume_val        = float(rec[3])
        vsl_val           = float(rec[4])
        deviation_val     = float(rec[5])
        sim_min           = float(rec[6])
        stick_sec         = rec[7]
        spot_angle        = rec[8]
        sampl_rate_hz     = rec[9]
        egg_loc_val       = rec[10]
        N_contact         = rec[11]
        N_sperm = int(sperm_conc_val * volume_val / 1000)
        ratio   = (N_contact / sim_min * 60.0) if sim_min > 0 else 0.0
        cps     = (N_contact / N_sperm) if N_sperm != 0 else 0.0
        key = (
            shape_val, sperm_conc_val, volume_val, vsl_val,
            deviation_val, sim_min, stick_sec, spot_angle,
            sampl_rate_hz, egg_loc_val
        )
        grouping[key].append((sim_id, ratio, N_sperm, cps))
    for key, values in grouping.items():
        (shape_val, sperm_conc_val, volume_val, vsl_val,
         deviation_val, sim_min, stick_sec, spot_angle,
         sampl_rate_hz, egg_loc_val) = key
        total_sim         = len(values)
        ratios            = [v[1] for v in values]
        Ns                = [v[2] for v in values]
        cps_list          = [v[3] for v in values]
        mean_contact_hr   = round(statistics.mean(ratios), 1) if ratios else 0.0
        SD1_val           = round(statistics.pstdev(ratios), 1) if len(ratios) > 1 else 0.0
        mean_N_sperm      = int(statistics.mean(Ns)) if Ns else 0
        C_per_N_val       = round(mean_contact_hr / mean_N_sperm, 1) if mean_N_sperm else 0.0
        SD2_val           = round(statistics.pstdev(cps_list), 1) if len(cps_list) > 1 else 0.0
        new_latest_sim_id = max(v[0] for v in values)
        c.execute('''
            INSERT OR REPLACE INTO summary (
                simulation_id, shape, sperm_conc, volume, vsl, deviation,
                sim_min, stick_sec, spot_angle, sampl_rate_hz, egg_localization,
                mean_contact_hr, SD1, N_sperm, C_per_N, SD2, total_simulations
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            new_latest_sim_id, shape_val, sperm_conc_val, volume_val,
            vsl_val, deviation_val, sim_min, stick_sec, spot_angle,
            sampl_rate_hz, egg_loc_val,
            mean_contact_hr, SD1_val, mean_N_sperm, C_per_N_val,
            SD2_val, total_sim
        ))
    conn.commit()
def calculate_n_sperm(df):
    df["N_sperm"] = df["N_sperm"].astype(int)
    return df
def load_previous_selection():
    config = configparser.ConfigParser()
    config.read(["user_selection.ini", "config.ini"])
    def get_list(section, key, default):
        return config.get(section, key, fallback=default).split(",")
    analysis_options = {}
    if "AnalysisOptions" in config:
        for key in config["AnalysisOptions"]:
            analysis_options[key] = get_list("AnalysisOptions", key, "")
    else:
        analysis_options = {}
    analysis_settings = {}
    if "Analysis" in config:
        for key in config["Analysis"]:
            analysis_settings[key] = config.get("Analysis", key, fallback="")
    else:
        analysis_settings = {}
    userselection_section = "UserSelection"
    def get_str(key, default):
        return config.get(userselection_section, key, fallback=default)
    n_repeat_default             = get_str("n_repeat",                  "3")
    seed_number_default          = get_str("seed_number",              "None")
    sim_min_default              = get_str("sim_min",                   "10")
    sampl_rate_hz_default        = get_str("sampl_rate_hz",            "2")
    spot_angle_default           = get_str("spot_angle",               "60")
    vsl_default                  = get_str("vsl",                       "0.15")
    deviation_default            = get_str("deviation",                 "0.4")
    stick_sec_default            = get_str("stick_sec",               "2")
    egg_localization_default     = get_str("egg_localization",          "bottom_center")
    gamete_r_default             = get_str("gamete_r",                 "0.15")
    initial_direction_default    = get_str("initial_direction",         "random")
    initial_stick_default = get_str("initial_stick",      "10")
    analysis_type_default        = get_str("analysis_type",             "simulation")
    def get_list_from_userselection(key, fallback_str, conv_func=None):
        raw_str = config.get(userselection_section, key, fallback=fallback_str)
        splitted = raw_str.split(",")
        splitted = [x for x in splitted if x]
        if conv_func is not None:
            return [conv_func(x) for x in splitted]
        else:
            return splitted
    volumes_list             = get_list_from_userselection("volumes",             "6.25,12.5,25,50,100,200,400,800,1600,3200", float)
    sperm_list               = get_list_from_userselection("sperm_concentrations","1000,3162,10000,31620,100000",         int)
    shapes_list              = get_list_from_userselection("shapes",              "cube,drop,spot",         None)
    outputs_list = get_list_from_userselection("outputs", "graph,movie", None)
    return {
        "analysis_options": analysis_options,
        "analysis_settings": analysis_settings,
        "n_repeat": n_repeat_default,
        "seed_number": seed_number_default,
        "sim_min": sim_min_default,
        "sampl_rate_hz": sampl_rate_hz_default,
        "spot_angle": spot_angle_default,
        "vsl": vsl_default,
        "deviation": deviation_default,
        "stick_sec": stick_sec_default,
        "egg_localization": egg_localization_default,
        "gamete_r": gamete_r_default,
        "initial_direction": initial_direction_default,
        "initial_stick": initial_stick_default,
        "analysis_type": analysis_type_default,
        "volumes": volumes_list,
        "sperm_concentrations": sperm_list,
        "shapes": shapes_list,
        "outputs": outputs_list,
    }
def save_previous_selection(values):
    config = configparser.ConfigParser()
    config.read("user_selection.ini")
    userselection_section = "UserSelection"
    if not config.has_section(userselection_section):
        config.add_section(userselection_section)
    single_keys = [
        "n_repeat", "seed_number", "sim_min", "sampl_rate_hz", "spot_angle",
        "vsl", "deviation", "stick_sec", "egg_localization",
        "gamete_r", "initial_direction", "initial_stick", "analysis_type"
    ]
    for key in single_keys:
        config.set(userselection_section, key, str(values.get(key, "")))
    list_keys = ["volumes", "sperm_concentrations", "shapes", "outputs"]
    for key in list_keys:
        items = values.get(key, [])
        config.set(userselection_section, key, ",".join(map(str, items)))
    with open("user_selection.ini", "w") as configfile:
        config.write(configfile)
def show_selection_ui(no_gui=False):
    selections = load_previous_selection()
    def get_str(key, default=""):
        return selections.get(key, default)
    if no_gui:
        # GUIを起動せず直前の設定を使用
        return load_previous_selection()
    # --- GUI 起動 ---
    root = tk.Tk()
    root.title("シミュレーションのパラメータ選択")
    root.geometry("600x600")
    root.attributes("-topmost", True)            
    root.update()                                
    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", on_configure)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    gui_elements = {}
    frames = {}
    radio_button_defaults = {
        "n_repeat_options":               ["1", "2", "3", "4", "5"],
        "seed_number_options":            ["None", "0", "1","2","3"],
        "sim_min_options":                ["0.2", "1", "10", "20","60","100"],
        "sampl_rate_hz_options":          ["1", "2", "3", "4"],
        "spot_angle_options":             ["30", "50","60","70", "90"],
        "vsl_options":                    ["0.073","0.1", "0.11", "0.12", "0.13", "0.14","0.15"],
        "deviation_options":              ["0", "0.2", "0.3","0.4","0.8","1.6", "3.2", "6.4", "12.8"],
        "stick_sec_options":              ["0", "1", "2", "3", "4","5", "6","7","8"],
        "egg_localization_options":       ["bottom_center", "center", "bottom_edge"],
        "analysis_type_options":          ["simulation", "reflection"],
        "initial_direction_options":      ["right", "left", "up", "down", "random"],
        "initial_stick_options":   ["0", "1", "10", "20"],
    }
    ordered_keys = [
        ("繰り返し回数 (N Repeat)",             "n_repeat_options"),
        ("乱数シード (Seed Number)",            "seed_number_options"),
        ("シミュレーション時間 (Sim Min)",      "sim_min_options"),
        ("サンプリングレート (Sample Rate Hz)", "sampl_rate_hz_options"),
        ("形状 (Shape)",                        "shapes"),
        ("Spot角度 (Spot Angle)",              "spot_angle_options"),
        ("体積 (Volume)",                      "volumes"),
        ("精子濃度 (Sperm Conc)",              "sperm_concentrations"),
        ("vsl",                                 "vsl_options"),
        ("偏差 (Deviation)",                   "deviation_options"),
        ("Stick_秒数 (Stick Sec)",             "stick_sec_options"),
        ("卵位置 (Egg Localization)",           "egg_localization_options"),
        ("Gamete半径 (Gamete R)",              "gamete_r_options"),
        ("出力設定 (Outputs)",                 "outputs"),
        ("解析タイプ (Analysis Type)",         "analysis_type_options"),
        ("初期方向 (Initial Direction)",       "initial_direction_options"),
        ("初期Stick状態 (Initial Stick)",      "initial_stick_options"),
    ]
    def update_states():
        shape_selected_spot = False
        if "shapes" in gui_elements and isinstance(gui_elements["shapes"], list):
            shape_selected_spot = any(var.get() and (val == "spot") for val, var in gui_elements["shapes"])
        state_spot_angle = "normal" if shape_selected_spot else "disabled"
        if "spot_angle_options" in frames:
            for child in frames["spot_angle_options"].winfo_children():
                child.config(state=state_spot_angle)
        analysis_type_val = gui_elements["analysis_type"].get() if "analysis_type" in gui_elements else ""
        is_reflection = (analysis_type_val == "reflection")
        state_reflection = "normal" if is_reflection else "disabled"
        for key in ["initial_direction_options", "initial_stick_options"]:
            if key in frames:
                for child in frames[key].winfo_children():
                    child.config(state=state_reflection)
    for label_text, key in ordered_keys:
        frame = tk.LabelFrame(scrollable_frame, text=label_text)
        frame.pack(fill="x", padx=10, pady=5)
        frames[key] = frame
        if key in ["shapes", "outputs", "volumes", "sperm_concentrations"]:
            default_list = selections.get(key, [])
            if key == "shapes":
                options = ["cube", "drop", "spot", "ceros"]
            elif key == "outputs":
                options = ["graph", "movie"]
            elif key == "volumes":
                options = ["6.25", "12.5", "25", "50", "100", "200","400","800","1600","3200"]
            elif key == "sperm_concentrations":
                options = ["1000", "3162", "10000", "31623", "100000"]
            vars_list = []
            for val in options:
                if key == "volumes":
                    conv_val = float(val)
                    is_checked = (conv_val in default_list)
                elif key == "sperm_concentrations":
                    conv_val = int(val)
                    is_checked = (conv_val in default_list)
                else:
                    conv_val = val
                    is_checked = (conv_val in default_list)
                var = tk.BooleanVar(value=is_checked)
                cb = tk.Checkbutton(frame, text=str(val), variable=var, command=update_states)
                cb.pack(side="left", padx=5, pady=5)
                vars_list.append((val, var))
            gui_elements[key] = vars_list
        elif key == "gamete_r_options":
            param_name = "gamete_r"
            default_value = str(selections.get(param_name, "0.1"))
            options = ["0.04", "0.05", "0.15"]
            var = tk.StringVar(value=default_value)
            for val in options:
                rb = tk.Radiobutton(frame, text=str(val), variable=var, value=str(val), command=update_states)
                rb.pack(side="left", padx=5, pady=5)
            gui_elements[param_name] = var
        else:
            param_name = key.replace("_options", "")
            default_value = str(selections.get(param_name, ""))
            possible_options = selections["analysis_options"].get(key, [])
            if not possible_options:
                possible_options = radio_button_defaults.get(key, [])
            if not possible_options:
                possible_options = [default_value]
            var = tk.StringVar(value=default_value)
            for val in possible_options:
                rb = tk.Radiobutton(frame, text=str(val), variable=var, value=str(val), command=update_states)
                rb.pack(side="left", padx=5, pady=5)
            gui_elements[param_name] = var
    update_states()
    def on_ok():
        selected_data = {}
        for key, element in gui_elements.items():
            if isinstance(element, list):
                if key == "volumes":
                    selected_data[key] = [float(val) for (val, var) in element if var.get()]
                elif key == "sperm_concentrations":
                    selected_data[key] = [int(val) for (val, var) in element if var.get()]
                else:
                    selected_data[key] = [val for (val, var) in element if var.get()]
            else:
                selected_data[key] = element.get()
        save_previous_selection(selected_data)
        root.selected_data = selected_data
        root.attributes("-topmost", False)
        root.destroy()
    btn_ok = tk.Button(scrollable_frame, text="OK", command=on_ok)
    btn_ok.pack(pady=10)
    root.mainloop()
    return getattr(root, 'selected_data', {})
def repeat_simulation(constants, repeat):
    simulations = []
    for r in range(repeat):
        print(f"n--- Simulation run {r+1} / {repeat} for shape={constants['shape']}, vol={constants['volume']}, conc={constants['sperm_conc']} ---")
       
        simulation_data = type('simulation_data', (object,), {
            'trajectory': np.zeros((
                int(constants['number_of_sperm']),
                int(constants['number_of_steps']),
                3
            ))
        })()
        simulation = SpermSimulation(constants, None, simulation_data)
        visualizer = SpermTrajectoryVisualizer(simulation)
        simulation.visualizer = visualizer
        simulation.simulate()
        print("🔍 ★simulate 実行完了")
        print("🔍 ★軌跡のサンプル座標:", simulation.trajectory[0, :3, :])
        for j in range(min(5, constants["number_of_sperm"])):
            print(f"sperm {j} trajectory:")
            print(simulation.trajectory[j, :, :])
        print("🔍 ★repeat_simulation() は呼ばれました")
        merged_events = simulation.merge_contact_events()
        print(f"Simulation run {r+1}: 接触数 = {len(merged_events)}")
        print("1時間あたり", len(merged_events)/constants["sim_min"]*60)
        image_id = None
        mov_id = None
        if constants['draw_trajectory'] == 'yes':
            plot = SpermPlot(simulation)
            image_id = plot._draw_graph(shape=constants['shape'])
        if constants.get('make_movie', 'no').lower() == 'yes':
            mov_filename = visualizer.animate_trajectory()
            mov_id = mov_filename
        simulations.append((simulation, image_id, mov_id, merged_events))
    return simulations
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gui", action="store_true",
                        help="Tkinter GUIを起動せず前回設定でバッチ実行")
    args, _ = parser.parse_known_args()

    start_time = time.time()
    version = get_program_version()

    # --- DB 準備 ------------------------------------------------------------
    db_path = DB_PATH_DEFAULT
    conn = sqlite3.connect(db_path)
    setup_database(conn)
    exp_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- GUI or 前回設定 ----------------------------------------------------
    selected_data = show_selection_ui(no_gui=args.no_gui)

    # --- GUI で取得した値を個別に取り出し -------------------------------
    volumes_list      = selected_data.get('volumes', [])
    sperm_conc_list   = selected_data.get('sperm_concentrations', [])
    shapes_list       = selected_data.get('shapes', [])
    spot_angle        = float(selected_data.get('spot_angle', 70))
    vsl               = float(selected_data.get('vsl', 0.13))
    deviation         = float(selected_data.get('deviation', 0.4))
    sampl_rate_hz     = float(selected_data.get('sampl_rate_hz', 2))
    sim_min           = float(selected_data.get('sim_min', 10))
    gamete_r          = float(selected_data.get('gamete_r', 0.1))
    stick_sec         = int(selected_data.get('stick_sec', 2))
    egg_localization  = selected_data.get('egg_localization', 'bottom_center')
    initial_direction = selected_data.get('initial_direction', 'right')
    initial_stick = int(selected_data.get('initial_stick', 10))
    seed_number       = selected_data.get('seed_number', None)
    n_repeat          = int(selected_data.get('n_repeat', 1))
    draw_trajectory   = 'yes' if 'graph' in selected_data.get('outputs', []) else 'no'
    make_movie        = 'yes' if 'movie' in selected_data.get('outputs', []) else 'no'

    # 乱数シード
    if seed_number and seed_number.lower() != IOStatus.NONE:
        np.random.seed(int(seed_number))

    # ============================================================
    #                      メインループ
    # ============================================================
    for shape in shapes_list:
        for volume in volumes_list:
            for sperm_conc in sperm_conc_list:

                # ---- constants を構築（GUI 値 + shape, volume, conc） ----
                constants = get_constants_from_gui(
                    selected_data, shape, volume, sperm_conc
                )
                # Spot形状では、radius を spot_r に合わせて明示的に設定する
                if constants["shape"] == "spot" and "spot_r" in constants:
                    constants["radius"] = constants["spot_r"]


                # ---- shape 固有パラメータの追加 ------------------------
                from tools.derived_constants import calculate_derived_constants

                # --- constantsの初期生成 ---
                constants = get_constants_from_gui(selected_data, shape, volume, sperm_conc)

                # --- shapeや派生値分岐を一切書かず、calculate_derived_constantsで自動計算 ---
                constants = calculate_derived_constants(constants)

                # ---- 共通パラメータの追加 ------------------------------
                constants['number_of_sperm'] = (
                    constants['sperm_conc'] * constants['volume'] / 1000
                )
                constants.update({
                    'spot_angle_rad': np.deg2rad(constants['spot_angle']),
                    'egg_volume'    : 4 * np.pi * constants['gamete_r']**3 / 3,
                    'stick_steps'   : constants['stick_sec'] * constants['sampl_rate_hz'],
                    'inner_angle'   : 2 * np.pi / 70,
                })
                constants['number_of_steps'] = int(
                    constants['sim_min'] * 60 * constants['sampl_rate_hz']
                )

                # ====================================================
                # ★ ここが  “constants を確定した直後” ★
                # ====================================================
                # まだ使わないが、後の置換フェーズで使えるよう生成しておく
                from spermsim.geometry import create_shape
                shape_obj = create_shape(constants["shape"], constants)
                # ----------------------------------------------------

                # ---- シミュレーション実行 ----------------------------
                from spermsim.simulation import SpermSimulation          # ★ 追加
                sim_engine = SpermSimulation(constants, shape_obj)       # ★ 追加
                simulations = sim_engine.run(n_repeat)                   # ★ 置換
                    # ---------------------------------------------------------------

                # ---- DB へ記録・集計 -------------------------------
                for i, (sim, image_id, mov_id, merged_events) in enumerate(simulations, start=1):
                    contact_count_merged = len(merged_events)
                    simulation_id = insert_sim_record(
                        conn, exp_id, version, constants,
                        image_id, mov_id, contact_count_merged
                    )
                    insert_intersection_records(conn, simulation_id, merged_events)

    # ---- 集計 ---------------------------------------------------------------
    aggregate_results(conn, exp_id)
    df_summary = pd.read_sql_query("SELECT * FROM summary", conn)
    if not df_summary.empty:
        df_summary = calculate_n_sperm(df_summary)
        df_summary.to_sql("summary", conn, if_exists="replace", index=False)
    else:
        print("summary テーブルに集計結果がありません。")

    conn.close()
    print(f"実行時間: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    main()


def _detect_boundary(shape, base_position, temp_position, stick_status, constants, prev_stat):
    """shape に応じて IOStatus と vertex_point を返す純粋関数
    戻り値: (IOStatus, vertex_point, temp_position)
    """
    if shape in ("cube", "ceros"):
        status, vertex = IO_check_cube(temp_position, constants)
        return status, vertex, temp_position

    if shape == "drop":
        status = IO_check_drop(temp_position, stick_status, constants)
        if status == IOStatus.BORDER:
            vec = temp_position - base_position
            temp_position = base_position + vec * 0.99
            status = IO_check_drop(temp_position, stick_status, constants)
        return status, None, temp_position

    if shape == "spot":
        status = IO_check_spot(base_position, temp_position, constants, prev_stat)
        if status == IOStatus.BORDER:
            vec = temp_position - base_position
            temp_position = base_position + vec * 0.99
            status = IO_check_spot(base_position, temp_position, constants, prev_stat)
        return status, None, temp_position

    return IOStatus.INSIDE, None, temp_position
