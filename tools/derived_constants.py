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
    vsl     = float(constants.get("vsl", 0.0))            # mm/s
    hz      = float(constants.get("sampl_rate_hz", 1.0))  # Hz

    # 単位変換 -----------------------------------------------------
    gamete_raw = float(constants.get("gamete_r", 0.0))
    gamete_r = gamete_raw / 1_000.0 if gamete_raw > 10 else gamete_raw
    constants["gamete_r"] = gamete_r

    if "drop_r" in constants:
        r_raw = float(constants["drop_r"])
        constants["drop_r"] = r_raw / 1_000.0 if r_raw > 10 else r_raw
        constants.setdefault("radius", constants["drop_r"])

    if "spot_r" in constants:
        r_raw = float(constants["spot_r"])
        constants["spot_r"] = r_raw / 1_000.0 if r_raw > 10 else r_raw
        constants.setdefault("radius", constants["spot_r"])

    if "spot_bottom_height" in constants:
        b_raw = float(constants["spot_bottom_height"])
        constants["spot_bottom_height"] = b_raw / 1_000.0 if b_raw > 10 else b_raw

    if "spot_bottom_r" in constants:
        br_raw = float(constants["spot_bottom_r"])
        constants["spot_bottom_r"] = br_raw / 1_000.0 if br_raw > 10 else br_raw

    # ---------- 共通パラメータ ------------------------------------------
    constants.update(
        vsl=vsl,
        step_length=vsl / hz if hz else 0.0,
        limit=1e-9,
        gamete_r=gamete_r,
    )

    return constants


# ------------------------------------------------------------
# plot_utils.py 用の軽量ヘルパー --------------------------------
# ------------------------------------------------------------
def get_limits(constants: Dict[str, float]) -> Tuple[float, float, float, float, float, float]:
    """x/y/z 各軸の min・max（mm）を tuple で返す"""
    return (constants["x_min"], constants["x_max"],
            constants["y_min"], constants["y_max"],
            constants["z_min"], constants["z_max"])
