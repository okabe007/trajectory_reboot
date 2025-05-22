
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
import os
from tools.plot_utils import plot_2d_trajectories
from core.geometry import CubeShape, DropShape, SpotShape, CerosShape
from tools.derived_constants import calculate_derived_constants


class SpermSimulation:
    def __init__(self, constants):
        self.constants = constants

        # --- 型安全化：数値パラメータはfloat/intに変換 ---
        float_keys = [
            "spot_angle", "vol", "sperm_conc", "vsl", "deviation", "surface_time",
            "gamete_r", "sim_min", "sampl_rate_hz"
        ]
        int_keys = [
            "sim_repeat"
        ]
        for key in float_keys:
            if key in self.constants and not isinstance(self.constants[key], float):
                try:
                    self.constants[key] = float(self.constants[key])
                except Exception:
                    print(f"[WARNING] {key} = {self.constants[key]} をfloat変換できませんでした")
        for key in int_keys:
            if key in self.constants and not isinstance(self.constants[key], int):
                try:
                    self.constants[key] = int(float(self.constants[key]))
                except Exception:
                    print(f"[WARNING] {key} = {self.constants[key]} をint変換できませんでした")
        # shape, egg_localization, などはstr型のままでOK
        self.constants = calculate_derived_constants(self.constants)
        

    
        # ------------------------------------------------------------------
    # ここから 全面置換  ※クラス定義はそのまま、run() だけ差し替え
    # ------------------------------------------------------------------
       # ------------------------------------------------------------------
    # ★★★ ここから run() 全面置換 ★★★
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # ★★★ ここから run() 全面置換 ★★★
    # ------------------------------------------------------------------
    def run(self, sim_repeat: int = 1):
        """
        sim_repeat 回シミュレーションを実行して
        self.trajectory に各精子の N×3 軌跡配列を格納する。
        座標・距離の単位は **mm** で統一。
        """
        # ---- デバッグ：派生値確認 -------------------------------------
        print("[DEBUG] SpermSimulation パラメータ:", self.constants)

        # ---- 形状オブジェクト生成 -------------------------------------
        shape = self.constants.get("shape", "cube")
        if shape == "cube":
            shape_obj = CubeShape(self.constants)
        elif shape == "spot":
            shape_obj = SpotShape(self.constants)
        elif shape == "drop":
            shape_obj = DropShape(self.constants)
        elif shape == "ceros":
            shape_obj = CerosShape(self.constants)
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        # ---- シミュレーション設定 -------------------------------------
        number_of_sperm  = int(self.constants.get("number_of_sperm", 10))
        number_of_steps  = int(self.constants.get("number_of_steps", 10))
        step_len         = self.constants["step_length"]     # ← mm / step
        seed_val         = self.constants.get("seed_number")
        if seed_val is not None and str(seed_val).lower() != "none":
            try:
                rng = np.random.default_rng(int(seed_val))
            except Exception:
                rng = np.random.default_rng()
        else:
            rng = np.random.default_rng()

        self.trajectory = []   # ← 毎 run() でリセット

        # ---- ループ ---------------------------------------------------
        for rep in range(int(sim_repeat)):
            for i in range(number_of_sperm):

                pos = shape_obj.initial_position()     # mm
                traj = [pos.copy()]

                for j in range(number_of_steps):
                    # ランダム方向の単位ベクトル
                    vec = rng.normal(size=3)
                    vec /= np.linalg.norm(vec) + 1e-12

                    pos = pos + vec * step_len         # ★ mm 単位で更新
                    traj.append(pos.copy())

                    # デバッグ：最初の一歩だけ確認
                    if rep == 0 and i == 0 and j == 0:
                        print(f"[DEBUG] 1step_disp(mm) = {np.linalg.norm(vec*step_len):.5f}")

                self.trajectory.append(np.vstack(traj))

        print(f"[DEBUG] run完了: sperm={len(self.trajectory)}, steps={number_of_steps}, "
              f"step_len={step_len} mm")



    import matplotlib.pyplot as plt

    def plot_trajectories(self, max_sperm=5, save_path=None):
        """
        インスタンスのself.trajectory（リスト of N×3 配列）を可視化
        max_sperm: 表示する精子軌跡の最大本数
        save_path: Noneなら画面表示のみ、パス指定で保存
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        trajectories = np.array(self.trajectory)
        constants = self.constants

        if trajectories is None or len(trajectories) == 0:
            print("[WARNING] 軌跡データがありません。run()実行後にplot_trajectoriesしてください。")
            return

        # --- 軸幅の統一 ---
        all_mins = [constants["x_min"], constants["y_min"], constants["z_min"]]
        all_maxs = [constants["x_max"], constants["y_max"], constants["z_max"]]
        global_min = min(all_mins)
        global_max = max(all_maxs)

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax_xy, ax_xz, ax_yz = axes
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        n_sperm = min(len(trajectories), max_sperm)

        # XY投影
        for i in range(n_sperm):
            ax_xy.plot(trajectories[i][:,0], trajectories[i][:,1], color=colors[i % len(colors)])
        ax_xy.set_xlim(global_min, global_max)
        ax_xy.set_ylim(global_min, global_max)
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("XY projection")

        # XZ投影
        for i in range(n_sperm):
            ax_xz.plot(trajectories[i][:,0], trajectories[i][:,2], color=colors[i % len(colors)])
        ax_xz.set_xlim(global_min, global_max)
        ax_xz.set_ylim(global_min, global_max)
        ax_xz.set_aspect('equal')
        ax_xz.set_xlabel("X")
        ax_xz.set_ylabel("Z")
        ax_xz.set_title("XZ projection")

        # YZ投影
        for i in range(n_sperm):
            ax_yz.plot(trajectories[i][:,1], trajectories[i][:,2], color=colors[i % len(colors)])
        ax_yz.set_xlim(global_min, global_max)
        ax_yz.set_ylim(global_min, global_max)
        ax_yz.set_aspect('equal')
        ax_yz.set_xlabel("Y")
        ax_yz.set_ylabel("Z")
        ax_yz.set_title("YZ projection")

        fig.suptitle(f"shape={constants.get('shape')}, vol={constants.get('vol')}, ...")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


    # def plot_trajectories(self, max_sperm=5, save_path=None):
        """
        シミュレーションで記録したself.trajectory（リスト of N×3 配列）を可視化します。
        max_sperm: 表示する精子軌跡の最大本数
        save_path: Noneならこのスクリプトと同じ階層のFigs_and_Moviesに自動保存
        """
        trajectories = self.trajectory  # run()でセットされているはず
        if not trajectories or len(trajectories) == 0:
            print("[WARNING] 軌跡データがありません。run()実行後にplot_trajectoriesしてください。")
            return

        # 描画本数を制限
        n_plot = min(max_sperm, len(trajectories))
        perc_shown = n_plot / len(trajectories) * 100

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # XY, XZ, YZ
        ax_labels = [("X", "Y"), ("X", "Z"), ("Y", "Z")]
        idxs = [(0, 1), (0, 2), (1, 2)]

        for ax, (label_x, label_y), (i, j) in zip(axes, ax_labels, idxs):
            for t in trajectories[:n_plot]:
                ax.plot(t[:, i], t[:, j], alpha=0.7)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.set_aspect('equal')
            ax.set_title(f"{label_x}{label_y} projection")

        # suptitleにパラメータ2行分
        param_summary = ', '.join(f"{k}={self.constants.get(k)}" for k in ["shape", "vol", "sperm_conc", "vsl", "deviation"])
        param_summary2 = ', '.join(f"{k}={self.constants.get(k)}" for k in ["surface_time", "egg_localization", "gamete_r", "sim_min", "sampl_rate_hz", "sim_repeat"])
        fig.suptitle(f"{param_summary}\n{param_summary2}", fontsize=12)

        # 本数注釈
        fig.text(0.99, 0.01, f"※ 表示は全体の{perc_shown:.1f}%（{n_plot}本/{len(trajectories)}本）", ha='right', fontsize=10, color="gray")

        fig.tight_layout(rect=[0, 0.03, 1, 0.92])

        # --- 保存先パスをスクリプトの場所基準で作る ---
        import datetime
        dtstr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # このスクリプトがあるディレクトリを基準にする
        base_dir = os.path.dirname(__file__)
        figs_dir = os.path.join(base_dir, "figs_and_movies")
        os.makedirs(figs_dir, exist_ok=True)

        if save_path is None:
            # 自動で日付入りファイル名
            save_path = os.path.join(figs_dir, f"trajectory_{dtstr}.png")
        else:
            # ファイル名だけ渡された場合もFigs_and_Moviesに入れる
            filename = os.path.basename(save_path)
            save_path = os.path.join(figs_dir, filename)

        plt.savefig(save_path)
        plt.close()
        print(f"[INFO] 軌跡画像を保存しました: {save_path}")