"""
spermsim パッケージ・トップレベル

`main()` を再エクスポートするが、**呼ばれるまで import しない**
ことで循環 import を回避する。
"""

__all__ = ["main"]

def main(*args, **kwargs):        # noqa: D401
    """エイリアス: `spermsim.main.main()` を呼び出す"""
    from .main import main as _main   # 遅延 import
    return _main(*args, **kwargs)
