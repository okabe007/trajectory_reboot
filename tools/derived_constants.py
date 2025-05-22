from __future__ import annotations     # ← 1 行目
from typing import Dict, Tuple

# ------------------------------------------------------------
# メイン関数  -------------------------------------------------
# ------------------------------------------------------------
def calculate_derived_constants(constants: Dict[str, float]) -> Dict[str, float]:
    """
    GUI／.ini で受け取った設定から派生変数を生成するユーティリティ。
        ・空間座標・形状パラメータ → mm に統一
        ・速度 vsl                 → mm/s で扱う
    """
    # ---------- 基本パラメータ ------------------------------------------
    shape   = str(constants.get("shape", "cube")).lower()
    vol_ul  = float(constants.get("vol", 1.0))            # µL
    vsl_mm  = float(constants.get("vsl", 0.0))            # mm/s
    hz      = float(constants.get("sampl_rate_hz", 1.0))  # Hz

    # ---------- 形状ごとの空間パラメータ（mm） --------------------------
    if shape == "cube":
        edge_mm = vol_ul ** (1 / 3)          # 1 µL = 1 mm³ なのでそのまま立方根
        h       = edge_mm / 2
        spatial = dict(edge=edge_mm,
                       x_min=-h, x_max= h,
                       y_min=-h, y_max= h,
                       z_min=-h, z_max= h)

    elif shape == "spot":
        spot_r_mm      = float(constants.get("spot_r", 0.0)) / 1_000.0
        spot_bottom_mm = float(constants.get("spot_bottom_height", 0.0)) / 1_000.0
        spatial = dict(spot_r=spot_r_mm,
                       x_min=-spot_r_mm, x_max= spot_r_mm,
                       y_min=-spot_r_mm, y_max= spot_r_mm,
                       z_min= spot_bottom_mm - spot_r_mm,
                       z_max= spot_bottom_mm + spot_r_mm)

    elif shape == "drop":
        r_mm   = float(constants.get("drop_r", 0.0)) / 1_000.0
        spatial = dict(radius=r_mm,
                       x_min=-r_mm, x_max=r_mm,
                       y_min=-r_mm, y_max=r_mm,
                       z_min=-r_mm, z_max=r_mm)

    elif shape == "ceros":    # テスト用固定範囲
        spatial = dict(x_min=-8.15, x_max=8.15,
                       y_min=-8.15, y_max=8.15,
                       z_min=-8.15, z_max=8.15)
    else:
        raise ValueError(f"未知の shape: {shape}")

    # ---------- 共通パラメータ ------------------------------------------
    spatial.update(vsl=vsl_mm,
                   step_length=vsl_mm / hz if hz else 0.0,
                   limit=1e-9)


    constants.update(spatial)
    # ★ADD: ここで派生値をすべて表示
    print("[DEBUG] derived_constants =", {k: constants[k] for k in (
        "vsl", "step_length",
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"
    )})
    return constants


# ------------------------------------------------------------
# plot_utils.py 用の軽量ヘルパー --------------------------------
# ------------------------------------------------------------
def get_limits(constants: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    """x/y/z 各軸の min・max（mm）を tuple で返す"""
    return (constants["x_min"], constants["x_max"],
            constants["y_min"], constants["y_max"],
            constants["z_min"], constants["z_max"])
