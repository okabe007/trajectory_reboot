# tools/derived_constants.py

import numpy as np

def calculate_derived_constants(raw_constants):
    """
    GUI から渡される raw_constants の中にある
      - "shape"                   : str ("cube", "drop", "spot", …)
      - "gamete_r"                : μm 単位の卵子半径
      - "drop_r"                  : μm 単位の drop 半径
      - "spot_r"                  : μm 単位の spot 半径
      - "spot_bottom_r"           : μm 単位の spot 底面半径
      - "spot_bottom_height"      : μm 単位の spot 底面高さ
      - "medium_volume_uL"        : μL 単位（= mm³）の媒質体積
      - "gamete_x_um", "gamete_y_um", "gamete_z_um"（卵子中心の μm 座標）
      - さらにシミュレーションに必要なパラメータ（例: "step_length", "time_step", "n_sperm" など）
    を受け取り、次の処理をして辞書を返します。

    1. raw_constants を丸ごとコピーして保持する（シミュレーション用キーを消さないため）
    2. 各種 μm 単位パラメータを mm 単位に変換して同名キーで上書き
       - "gamete_r", "drop_r", "spot_r", "spot_bottom_r", "spot_bottom_height" → mm
    3. 媒質体積 ("medium_volume_uL"：μL=mm³) → 半径 (mm) を計算し "medium_radius" を追加
    4. "shape" ごとに x_min～z_max を mm 単位で計算して上書き
    5. 卵子中心座標を μm→mm に変換して "egg_center" キーに np.array([x_mm, y_mm, z_mm]) として追加

    戻り値の constants には、上記の処理で追加・上書きされた mm 単位キーと、
    raw_constants にもともと含まれていたすべてのシミュレーション用キーが混在します。
    """

    # ─── ① raw_constants を丸ごとコピーして「既存のキー」をすべて保持 ───
    constants = raw_constants.copy()

    # ─── ② shape を lower() して上書き ───
    shape = constants.get("shape", "cube").lower()
    constants["shape"] = shape

    # ─── ③ 卵子半径 (μm → mm) ───
    gamete_r_um = float(constants.get("gamete_r", 50.0))  # μm
    gamete_r_mm = gamete_r_um / 1000.0                     # → mm
    constants["gamete_r"] = gamete_r_mm

    # ─── ④ drop 形状の半径 (μm → mm) ───
    drop_r_um = float(constants.get("drop_r", 0.0))       # μm
    drop_r_mm = drop_r_um / 1000.0                        # → mm
    constants["drop_r"] = drop_r_mm

    # ─── ⑤ spot 形状の半径および底面半径・底面高さ (μm → mm) ───
    spot_r_um = float(constants.get("spot_r", 0.0))       # μm
    spot_r_mm = spot_r_um / 1000.0                        # → mm
    constants["spot_r"] = spot_r_mm

    spot_bottom_r_um = float(constants.get("spot_bottom_r", spot_r_um))  # μm
    spot_bottom_r_mm = spot_bottom_r_um / 1000.0                          # → mm
    constants["spot_bottom_r"] = spot_bottom_r_mm

    spot_bottom_h_um = float(constants.get("spot_bottom_height", 0.0))     # μm
    spot_bottom_h_mm = spot_bottom_h_um / 1000.0                           # → mm
    constants["spot_bottom_height"] = spot_bottom_h_mm

    # ─── ⑥ 媒質体積 (μL=mm³) → 半径 (mm) ───
    #     1 μL = 1 mm³ なので、raw_constants["medium_volume_uL"] をそのまま mm³ として扱う
    medium_volume_uL = float(constants.get("medium_volume_uL", 0.0))
    medium_volume_mm3 = medium_volume_uL  # μL=mm³
    if medium_volume_mm3 > 0:
        medium_radius_mm = ((3.0 * medium_volume_mm3) / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        medium_radius_mm = 0.0
    constants["medium_radius"] = medium_radius_mm

    # ─── ⑦ プロット用リミット (x_min, x_max, y_min, y_max, z_min, z_max) を mm 単位で計算 ───
    if shape == "cube":
        half = medium_radius_mm
        constants["x_min"] = -half
        constants["x_max"] =  half
        constants["y_min"] = -half
        constants["y_max"] =  half
        constants["z_min"] =  0.0
        constants["z_max"] =  2.0 * half

    elif shape == "drop":
        r = drop_r_mm
        constants["x_min"] = -r
        constants["x_max"] =  r
        constants["y_min"] = -r
        constants["y_max"] =  r
        constants["z_min"] =  0.0
        constants["z_max"] =  2.0 * r

    elif shape == "spot":
        R   = spot_r_mm
        b_r = spot_bottom_r_mm
        b_h = spot_bottom_h_mm

        constants["x_min"] = -b_r
        constants["x_max"] =  b_r
        constants["y_min"] = -b_r
        constants["y_max"] =  b_r
        constants["z_min"] =  0.0
        constants["z_max"] =  R

    else:
        # その他の形状は cube と同様に囲む
        half = medium_radius_mm
        constants["x_min"] = -half
        constants["x_max"] =  half
        constants["y_min"] = -half
        constants["y_max"] =  half
        constants["z_min"] =  0.0
        constants["z_max"] =  2.0 * half

    # ─── ⑧ 卵子中心座標を μm→mm で計算し "egg_center" キーに追加 ───
    #      _egg_position(raw_constants) は μm 単位で (x_um, y_um, z_um) を返す想定
    egg_pos_um = _egg_position(raw_constants)  # [x_um, y_um, z_um]

    egg_x_mm = egg_pos_um[0] / 1000.0
    egg_y_mm = egg_pos_um[1] / 1000.0

    # "spot" の場合は底面高さを raw_z_min_um とし、それ以外は 0
    if shape == "spot":
        raw_z_min_um = float(constants.get("spot_bottom_height", 0.0))
    else:
        raw_z_min_um = 0.0

    egg_z_mm = (raw_z_min_um + gamete_r_um) / 1000.0
    constants["egg_center"] = np.array([egg_x_mm, egg_y_mm, egg_z_mm])

    return constants


def get_limits(constants):
    """
    calculate_derived_constants が返した mm 単位の
    x_min ～ z_max をそのまま返す。
    """
    return (
        constants["x_min"],
        constants["x_max"],
        constants["y_min"],
        constants["y_max"],
        constants["z_min"],
        constants["z_max"]
    )


def _egg_position(raw_constants):
    """
    GUI から渡された raw_constants に基づき、卵子中心を μm 単位で返す関数。
    ここでは例として、"gamete_x_um", "gamete_y_um", "gamete_z_um" の値を返しますが、
    実際は適宜ロジックを実装してください。
    """
    x_um = raw_constants.get("gamete_x_um", 0.0)
    y_um = raw_constants.get("gamete_y_um", 0.0)
    z_um = raw_constants.get("gamete_z_um", 0.0)
    return np.array([x_um, y_um, z_um])
