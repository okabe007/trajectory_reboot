"""
I/O 状態を表す列挙型 ― spermsim_step1 版
シミュレーション内部で参照される 18種類の状態を完全に網羅。
"""
from enum import Enum
class IOStatus(Enum):
    """
    すべての形状（cube, drop, spot, ceros）に対応する接触判定ステータス
    """
    # --- 共通・基本 ---
    INSIDE = "inside"           # 内部
    BORDER = "border"           # 境界ぎりぎり
    REFLECT = "reflect"         # 反射（跳ね返り）
    STICK = "stick"             # 表面に貼り付き（一定時間停止）

    # --- spot専用 ---
    SPHERE_OUT = "sphere_out"         # 球の外全体
    BOTTOM_OUT = "bottom_out"         # 底面より下へ出た
    SPOT_EDGE_OUT = "spot_edge_out"   # 円柱外周の外
    POLYGON_MODE = "polygon_mode"     # spot内で貼り付きモードに遷移
    SPOT_BOTTOM = "spot_bottom"       # 円柱の底面内（潜在的貼り付きエリア）






# class IOStatus(str, Enum):
#     # ── 基本 ───────────────────────────
#     NONE = "none"                 # 未定義
#     INSIDE = "inside"             # 領域内
#     OUTSIDE = "outside"           # 完全に外
#     BORDER = "border"             # 辺・面・頂点など境界上
#     SURFACE = "surface"           # 面上
#     REFLECT = "reflect"
#     STICK = "stick"        # ← 今回追加すべき部分
#     # ── 一時判定（temp_ で始まるもの） ─
#     TEMP_ON_SURFACE = "temp_on_surface"
#     TEMP_ON_EDGE = "temp_on_edge"
#     TEMP_ON_POLYGON = "temp_on_polygon"

#     # ── 領域別・形状別 ──────────────────
#     SPHERE_OUT = "sphere_out"         # 球外
#     POLYGON_MODE = "polygon_mode"     # 多角形モード
#     SPOT_BOTTOM = "spot_bottom"
#     SPOT_EDGE_OUT = "spot_edge_out"
#     ON_EDGE_BOTTOM = "on_edge_bottom"
#     BOTTOM_EDGE_MODE = "bottom_edge_mode"
#     VERTEX_OUT = "vertex_out"

#     # ── “〜_OUT” グループ ────────────────
#     BOTTOM_OUT = "bottom_out"
#     SURFACE_OUT = "surface_out"
#     EDGE_OUT = "edge_out"
