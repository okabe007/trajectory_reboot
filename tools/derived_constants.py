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
    hz      = float(constants.get("sample_rate_hz", 1.0))  # Hz

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

    # number_of_sperm を濃度と体積から計算（µL→mL 換算）
    if "vol" in constants and "sperm_conc" in constants:
        try:
            vol_ul = float(constants["vol"])
            conc = float(constants["sperm_conc"])
            constants["number_of_sperm"] = int(conc * vol_ul / 1000)
        except Exception:
            pass

    return constants

def _egg_position(constants: dict) -> list[float]:
    """
    卵子の位置を shape と egg_localization に応じて返す。
    cube, drop, spot 各形状で計算式が異なる。
    """
    mode = constants.get("egg_localization", "center")
    shape = constants.get("shape", "cube").lower()
    gamete_r = constants.get("gamete_r", 0.05)

    if shape == "cube" or shape == "drop":
        z_min = constants.get("z_min", -1.0)
        z_max = constants.get("z_max", 1.0)
        if mode == "center":
            return [0.0, 0.0, 0.0]
        elif mode == "bottom_center":
            return [0.0, 0.0, z_min + gamete_r]
        elif mode == "top_center":
            return [0.0, 0.0, z_max - gamete_r]

    elif shape == "spot":
        spot_r = constants.get("spot_r", 1.0)
        spot_bottom_height = constants.get("spot_bottom_height", 0.0)
        if mode == "center":
            return [0.0, 0.0, (spot_r + spot_bottom_height) / 2]
        elif mode == "bottom_center":
            return [0.0, 0.0, spot_bottom_height + gamete_r]
        elif mode == "top_center":
            return [0.0, 0.0, spot_r - gamete_r]

    raise ValueError(f"Unknown shape '{shape}' or egg_localization mode '{mode}'")


# ------------------------------------------------------------
# plot_utils.py 用の軽量ヘルパー --------------------------------
# ------------------------------------------------------------
def get_limits(constants: dict) -> tuple:
    x_min = constants.get("x_min", -1.0)
    x_max = constants.get("x_max", 1.0)
    y_min = constants.get("y_min", -1.0)
    y_max = constants.get("y_max", 1.0)
    z_min = constants.get("z_min", -1.0)
    z_max = constants.get("z_max", 1.0)

    return x_min, x_max, y_min, y_max, z_min, z_max
