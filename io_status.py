"""
I/O 状態を表す列挙型 ― spermsim_step1 版
シミュレーション内部で参照される 18種類の状態を完全に網羅。
"""

from enum import Enum


class IOStatus(str, Enum):
    # ── 基本 ───────────────────────────
    NONE = "none"                 # 未定義

    INSIDE = "inside"             # 領域内
    OUTSIDE = "outside"           # 完全に外
    BORDER = "border"             # 辺・面・頂点など境界上
    SURFACE = "surface"           # 面上

    # ── 一時判定（temp_ で始まるもの） ─
    TEMP_ON_SURFACE = "temp_on_surface"
    TEMP_ON_EDGE = "temp_on_edge"
    TEMP_ON_POLYGON = "temp_on_polygon"

    # ── 領域別・形状別 ──────────────────
    SPHERE_OUT = "sphere_out"         # 球外
    POLYGON_MODE = "polygon_mode"     # 多角形モード
    SPOT_BOTTOM = "spot_bottom"
    SPOT_EDGE_OUT = "spot_edge_out"
    ON_EDGE_BOTTOM = "on_edge_bottom"
    BOTTOM_EDGE_MODE = "bottom_edge_mode"
    VERTEX_OUT = "vertex_out"

    # ── “〜_OUT” グループ ────────────────
    BOTTOM_OUT = "bottom_out"
    SURFACE_OUT = "surface_out"
    EDGE_OUT = "edge_out"
