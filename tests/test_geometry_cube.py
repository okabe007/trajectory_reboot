# tests/test_geometry_cube.py
"""
CubeShape の edge_um が
   edge_um = (体積[µm^3])**(1/3)
になることを確認する基本テスト
"""

import numpy as np
import math

# パッケージルートを import 対象にする（相対パス調整がいらなくなる）
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.geometry import CubeShape


def test_edge_length_basic():
    """体積 1.0 µm³ → 一辺 1.0 µm になるか"""
    cube = CubeShape({"vol_um3": 1.0})  # ✅ 修正点
    assert math.isclose(cube.edge_um, 1.0, rel_tol=0, abs_tol=1e-12)


def test_edge_length_random():
    """ランダムに 5 個選んでチェック"""
    rng = np.random.default_rng(42)
    vols = rng.uniform(0.1, 1000.0, size=5)  # 0.1〜1000 µm³
    for v in vols:
        cube = CubeShape({"vol_um3": v})  # ✅ 修正点
        expect = v ** (1.0 / 3.0)
        assert math.isclose(cube.edge_um, expect, rel_tol=1e-12)
