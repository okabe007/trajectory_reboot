# 🧬 trajectory_reboot

精子の運動軌跡を3次元空間でシミュレートするPythonベースの科学研究用ツールです。  
体積や運動速度などを変数として精子の卵子との遭遇率を観察に基づいて、仮想的な環境においてsimulationします。

---

## 📦 機能概要

- **立方体・球体・液滴モデル**による運動空間の設定
- 精子の初期位置・速度・ランダムな進行方向の変更
- 境界条件（反射／吸着／多様な境界モード）の切替
- 卵子との遭遇頻度の統計的出力
- `matplotlib` によるXY/XZ/YZ/3Dグラフ描画
- `.ini` ファイルによるパラメータ指定と再現実験
- `pytest` によるユニットテスト対応済み

---

## 🚀 セットアップ方法

このプロジェクトを実行するには、Python 3.8 以上が必要です。

### 1. 通常の環境で

```bash
git clone https://github.com/<your-username>/trajectory_reboot.git
cd trajectory_reboot
pip install -r requirements.txt
nano README.md## 🚀 セットアップ方法（初回のみ）

このプロジェクトを実行するには、Python の依存ライブラリをインストールする必要があります。

### ✅ 1. 通常の Python 環境でのセットアップ方法

```bash
pip install -r requirements.txt

