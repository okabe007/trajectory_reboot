
import os
import tkinter as tk
from tkinter import ttk
import configparser
import numpy as np
import math

from core.simulation import SpermSimulation       # ← ここで派生変数計算を呼ぶ
from tools.plot_utils import plot_2d_trajectories # plot_3d_trajectories
from tools.derived_constants import calculate_derived_constants

# ---------------------------------------------------------------------------
# .ini ファイルのパス
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "sperm_config.ini")

# 保存時のキー順（GUI 表示順と合わせる）
PARAM_ORDER = [
    "shape", "spot_angle", "vol", "sperm_conc", "vsl", "deviation",
    "surface_time", "egg_localization", "gamete_r", "sim_min",
    "sampl_rate_hz", "seed_number", "sim_repeat", "display_mode",
]

# デフォルト設定
default_values = {
    "shape": "cube",
    "spot_angle": 50.0,
    "vol": 6.25,               # µL
    "sperm_conc": 10_000.0,    # cells/mL
    "vsl": 0.13,               # mm/s
    "deviation": 0.4,
    "surface_time": 2.0,
    "egg_localization": "bottom_center",
    "gamete_r": 0.04,          # mm  (GUI 表示は µm)
    "sim_min": 1.0,            # min 実測値ではなく「分」→秒に変換は simulation 側
    "sampl_rate_hz": 4.0,
    "seed_number": "None",
    "sim_repeat": 1,
    "display_mode": ["2D"],    # 文字列リスト
}

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------
def _calc_spot_geometry(volume_ul: float, angle_deg: float) -> tuple[float, float, float]:
    """Return (spot_r_mm, bottom_r_mm, bottom_height_mm) from volume and angle."""
    angle_rad = math.radians(angle_deg)
    vol_um3 = volume_ul * 1e9

    def cap_volume(R: float) -> float:
        h = R * (1 - math.cos(angle_rad))
        return math.pi * h * h * (3 * R - h) / 3

    low = 0.0
    high = max(vol_um3 ** (1 / 3), 1.0)
    while cap_volume(high) < vol_um3:
        high *= 2.0

    for _ in range(60):
        mid = (low + high) / 2.0
        if cap_volume(mid) < vol_um3:
            low = mid
        else:
            high = mid

    R_um = (low + high) / 2.0
    bottom_r_um = R_um * math.sin(angle_rad)
    bottom_height_um = R_um * math.cos(angle_rad)
    return R_um / 1000.0, bottom_r_um / 1000.0, bottom_height_um / 1000.0

# ---------------------------------------------------------------------------
# .ini 読み書きユーティリティ
# ---------------------------------------------------------------------------
def save_config(values: dict) -> None:
    cfg = configparser.ConfigParser()
    ordered = {}
    for k in PARAM_ORDER:
        if k not in values:
            continue
        v = values[k]
        if k == "display_mode":
            ordered[k] = ",".join(v) if isinstance(v, list) else str(v)
        else:
            ordered[k] = str(v)
    # 保存対象外のキーもまとめて保存
    for k in sorted(values.keys()):
        if k in ordered or k in PARAM_ORDER:
            continue
        ordered[k] = str(values[k])
    cfg["simulation"] = ordered
    with open(CONFIG_PATH, "w") as f:
        cfg.write(f)


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save_config(default_values)
        return default_values.copy()

    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)
    values = default_values.copy()
    c = cfg["simulation"]

    # 型を安全に解釈
    for k in PARAM_ORDER:
        raw = c.get(k, str(default_values[k]))
        try:
            if k in ["vsl", "spot_angle", "gamete_r", "sperm_conc",
                     "vol", "sampl_rate_hz", "sim_min"]:
                values[k] = float(raw)
            elif k == "sim_repeat":
                values[k] = int(float(raw))
            elif k == "display_mode":
                values[k] = [v for v in raw.split(",") if v]
            else:
                values[k] = raw
        except Exception:
            values[k] = raw
    return values

# ---------------------------------------------------------------------------
# Tkinter GUI クラス
# ---------------------------------------------------------------------------
class SimApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sperm Simulation GUI")
        self.geometry("780x900")

        self.config_data = load_config()  # .ini → dict
        self.tk_vars: dict[str, tk.Variable] = {}  # Param ↔ Tk 変数

        # スクロールキャンバス
        canvas = tk.Canvas(self)
        vbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        # 各ウィジェットを配置
        self._create_widgets(self.scroll_frame)
        self._restore_from_config()  # 値を復元

    # ---------------------------------------------------------------------
    # ウィジェット生成
    # ---------------------------------------------------------------------
    def _create_widgets(self, parent: ttk.Frame) -> None:
        # --- shape -------------------------------------------------------
        self.tk_vars["shape"] = tk.StringVar()
        ttk.Label(parent, text="shape:").pack(anchor="w", padx=10, pady=(10, 0))
        f_shape = ttk.Frame(parent); f_shape.pack(anchor="w", padx=30)
        for v in ["cube", "drop", "spot", "ceros"]:
            ttk.Radiobutton(f_shape, text=v, variable=self.tk_vars["shape"],
                            value=v, command=self._update_spot_angle_state
                            ).pack(side="left")

        # --- spot_angle --------------------------------------------------
        self.tk_vars["spot_angle"] = tk.DoubleVar()
        ttk.Label(parent, text="spot_angle (deg):").pack(anchor="w", padx=10, pady=(10, 0))
        f_angle = ttk.Frame(parent); f_angle.pack(anchor="w", padx=30)
        self.spot_angle_buttons = []
        for v in [30, 50, 70, 90]:
            rb = ttk.Radiobutton(f_angle, text=str(v),
                                 variable=self.tk_vars["spot_angle"], value=float(v))
            rb.pack(side="left")
            self.spot_angle_buttons.append(rb)

        # --- vol ---------------------------------------------------------
        self.tk_vars["vol"] = tk.DoubleVar()
        ttk.Label(parent, text="vol (µL):").pack(anchor="w", padx=10, pady=(10, 0))
        f_vol = ttk.Frame(parent); f_vol.pack(anchor="w", padx=30)
        for v in [6.25, 12.5, 25, 50, 100, 200]:
            ttk.Radiobutton(f_vol, text=str(v), variable=self.tk_vars["vol"],
                            value=float(v)).pack(side="left")

        # --- sperm_conc --------------------------------------------------
        self.tk_vars["sperm_conc"] = tk.DoubleVar()
        ttk.Label(parent, text="sperm_conc (cells/mL):").pack(anchor="w", padx=10, pady=(10, 0))
        f_conc = ttk.Frame(parent); f_conc.pack(anchor="w", padx=30)
        for v in [1e3, 3.16e3, 1e4, 3.162e4, 1e5]:
            ttk.Radiobutton(f_conc, text=f"{int(v):,}", variable=self.tk_vars["sperm_conc"],
                            value=float(v)).pack(side="left")

        # --- vsl ---------------------------------------------------------
        self.tk_vars["vsl"] = tk.DoubleVar()
        ttk.Label(parent, text="vsl (µm/s):").pack(anchor="w", padx=10, pady=(10, 0))
        f_vsl = ttk.Frame(parent); f_vsl.pack(anchor="w", padx=30)
        for v in [73, 90, 110, 130, 150]:
            ttk.Radiobutton(f_vsl, text=str(v), variable=self.tk_vars["vsl"],
                            value=float(v)).pack(side="left")

        # --- deviation ---------------------------------------------------
        self.tk_vars["deviation"] = tk.DoubleVar()
        ttk.Label(parent, text="deviation:").pack(anchor="w", padx=10, pady=(10, 0))
        f_dev = ttk.Frame(parent); f_dev.pack(anchor="w", padx=30)
        for v in [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0]:
            ttk.Radiobutton(f_dev, text=str(v), variable=self.tk_vars["deviation"],
                            value=float(v)).pack(side="left")

        # --- surface_time ------------------------------------------------
        self.tk_vars["surface_time"] = tk.DoubleVar()
        ttk.Label(parent, text="surface_time (sec):").pack(anchor="w", padx=10, pady=(10, 0))
        f_surface = ttk.Frame(parent); f_surface.pack(anchor="w", padx=30)
        for v in [0, 1, 2, 3, 4]:
            ttk.Radiobutton(f_surface, text=str(v), variable=self.tk_vars["surface_time"],
                            value=float(v)).pack(side="left")

        # --- egg_localization -------------------------------------------
        self.tk_vars["egg_localization"] = tk.StringVar()
        ttk.Label(parent, text="egg_localization:").pack(anchor="w", padx=10, pady=(10, 0))
        f_egg = ttk.Frame(parent); f_egg.pack(anchor="w", padx=30)
        for v in ["bottom_center", "bottom_edge", "top", "random"]:
            ttk.Radiobutton(f_egg, text=v, variable=self.tk_vars["egg_localization"],
                            value=v).pack(side="left")

        # --- gamete_r ----------------------------------------------------
        self.tk_vars["gamete_r"] = tk.DoubleVar()
        ttk.Label(parent, text="gamete_r (µm):").pack(anchor="w", padx=10, pady=(10, 0))
        f_gr = ttk.Frame(parent); f_gr.pack(anchor="w", padx=30)
        for v in [40, 50, 150]:
            ttk.Radiobutton(f_gr, text=str(v), variable=self.tk_vars["gamete_r"],
                            value=float(v)).pack(side="left")

        # --- sim_min -----------------------------------------------------
        self.tk_vars["sim_min"] = tk.DoubleVar()
        ttk.Label(parent, text="sim_min (min):").pack(anchor="w", padx=10, pady=(10, 0))
        f_simmin = ttk.Frame(parent); f_simmin.pack(anchor="w", padx=30)
        for v in [0.2, 1, 10, 30, 60]:
            ttk.Radiobutton(f_simmin, text=str(v), variable=self.tk_vars["sim_min"],
                            value=float(v)).pack(side="left")

        # --- sampl_rate_hz ----------------------------------------------
        self.tk_vars["sampl_rate_hz"] = tk.DoubleVar()
        ttk.Label(parent, text="sampl_rate_hz:").pack(anchor="w", padx=10, pady=(10, 0))
        f_hz = ttk.Frame(parent); f_hz.pack(anchor="w", padx=30)
        for v in [1, 2, 3, 4]:
            ttk.Radiobutton(f_hz, text=str(v), variable=self.tk_vars["sampl_rate_hz"],
                            value=float(v)).pack(side="left")

        # --- seed_number ----------------------------------------------
        self.tk_vars["seed_number"] = tk.StringVar()
        ttk.Label(parent, text="seed_number:").pack(anchor="w", padx=10, pady=(10, 0))
        f_seed = ttk.Frame(parent); f_seed.pack(anchor="w", padx=30)
        for v in ["None", "0", "1", "2", "3", "4"]:
            ttk.Radiobutton(f_seed, text=v, variable=self.tk_vars["seed_number"],
                            value=v).pack(side="left")

        # --- sim_repeat --------------------------------------------------
        self.tk_vars["sim_repeat"] = tk.IntVar()
        ttk.Label(parent, text="sim_repeat:").pack(anchor="w", padx=10, pady=(10, 0))
        f_repeat = ttk.Frame(parent); f_repeat.pack(anchor="w", padx=30)
        for v in [1, 3, 10, 30]:
            ttk.Radiobutton(f_repeat, text=str(v), variable=self.tk_vars["sim_repeat"],
                            value=int(v)).pack(side="left")

        # --- display_mode ラジオボタン ------------------------------
        self.tk_vars["display_mode"] = tk.StringVar()
        ttk.Label(parent, text="display_mode:").pack(anchor="w", padx=10, pady=(10, 0))
        f_disp = ttk.Frame(parent); f_disp.pack(anchor="w", padx=30)
        for v in ["2D", "3D", "movie"]:
            ttk.Radiobutton(f_disp, text=v, variable=self.tk_vars["display_mode"], value=v
                            ).pack(side="left")

        # --- 実行ボタン --------------------------------------------------
        ttk.Button(parent, text="Save settings and run simulation",
                   command=self._on_save).pack(pady=20)

        self._update_spot_angle_state()

    # ---------------------------------------------------------------------
    # spot_angle の有効／無効
    # ---------------------------------------------------------------------
    def _update_spot_angle_state(self) -> None:
        enable = (self.tk_vars["shape"].get() == "spot")
        state = "normal" if enable else "disabled"
        for rb in self.spot_angle_buttons:
            rb.config(state=state)

    # ---------------------------------------------------------------------
    # 保存＆シミュレーション実行
    # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
    # 保存＆シミュレーション実行  ← ★ここを全面差し替え★
    # ---------------------------------------------------------------------
    def _on_save(self) -> None:
        """
        1. GUI の Tk 変数 → self.config_data へ安全にコピー
           （vsl は mm/s、gamete_r は mm）
        2. .ini に保存
        3. シミュレーションを実行
        """
        # --- ① Tk → config_data（vsl, gamete_r は mm で保持） -----
        for k, var in self.tk_vars.items():
            try:
                if isinstance(var, (tk.DoubleVar, tk.IntVar)):
                    val = float(var.get()) if isinstance(var, tk.DoubleVar) else int(var.get())
                    if k == "vsl":
                        val /= 1000.0  # µm/s → mm/s
                    elif k == "gamete_r":
                        val /= 1000.0  # µm  → mm
                    self.config_data[k] = val
                else:
                    self.config_data[k] = var.get()
            except Exception:
                self.config_data[k] = var.get()

        # display_mode（ラジオボタン）
        mode = self.tk_vars["display_mode"].get()
        modes = [mode] if mode else []
        self.config_data["display_mode"] = modes

        # --- ② shape specific parameters -------------------------------
        shape = str(self.config_data.get("shape", "")).lower()
        vol_ul = float(self.config_data.get("vol", 0.0))

        if shape == "cube":
            edge = vol_ul ** (1 / 3)
            half = edge / 2
            self.config_data.update(
                edge=edge,
                x_min=-half, x_max=half,
                y_min=-half, y_max=half,
                z_min=-half, z_max=half,
            )

        elif shape == "drop":
            r = (3 * vol_ul / (4 * math.pi)) ** (1 / 3)
            self.config_data.update(
                drop_r=r,
                radius=r,
                x_min=-r, x_max=r,
                y_min=-r, y_max=r,
                z_min=-r, z_max=r,
            )

        elif shape == "spot":
            angle = float(self.config_data.get("spot_angle", 0.0))
            spot_r, bottom_r, bottom_h = _calc_spot_geometry(vol_ul, angle)
            self.config_data.update(
                spot_r=spot_r,
                spot_bottom_r=bottom_r,
                spot_bottom_height=bottom_h,
                radius=spot_r,
                x_min=-spot_r, x_max=spot_r,
                y_min=-spot_r, y_max=spot_r,
                z_min=bottom_h,
                z_max=spot_r,
            )

        elif shape == "ceros":
            self.config_data.update(
                x_min=-8.15, x_max=8.15,
                y_min=-8.15, y_max=8.15,
                z_min=-8.15, z_max=8.15,
            )

        # --- ③ generic parameters ------------------------------------
        vsl = float(self.config_data.get("vsl", 0.0))
        hz = float(self.config_data.get("sampl_rate_hz", 1.0))
        self.config_data.update(
            step_length=vsl / hz if hz else 0.0,
            limit=1e-9,
        )

        # --- シミュレーションステップ数を sim_min と sampl_rate_hz から計算 ---
        sim_min = float(self.config_data.get("sim_min", 0.0))
        self.config_data["number_of_steps"] = int(sim_min * 60 * hz)

        # --- ④ 派生値計算（互換性用） ---------------------------------
        self.config_data = calculate_derived_constants(self.config_data)

        # --- ③ ini 保存 -----------------------------------------------
        save_config(self.config_data)
        # --- ④ シミュレーション実行 --------------------------------------
        sim = SpermSimulation(self.config_data)
        sim.run(self.config_data["sim_repeat"])

        # --- ⑤ 描画 -----------------------------------------------------
        if "2D" in modes:
            plot_2d_trajectories(np.array(sim.trajectory), self.config_data)
        elif "3D" in modes:
            plot_3d_trajectories(np.array(sim.trajectory), self.config_data)
        elif "movie" in modes:
            sim.plot_movie_trajectories()   # 実装に合わせて


    # ---------------------------------------------------------------------
    # 起動時に .ini から各 Tk 変数を復元
    # ---------------------------------------------------------------------
    def _restore_from_config(self) -> None:
        for k, var in self.tk_vars.items():
            if k not in self.config_data:
                continue
            v = self.config_data[k]
            try:
                if isinstance(var, tk.DoubleVar):
                    val = float(v)
                    if k == "vsl":
                        val *= 1000.0  # mm/s → µm/s
                    elif k == "gamete_r":
                        val *= 1000.0  # mm  → µm
                    var.set(val)
                elif isinstance(var, tk.IntVar):
                    var.set(int(float(v)))
                else:
                    var.set(str(v))
            except Exception:
                var.set(v)
        # display_mode
        modes = self.config_data.get("display_mode", [])
        if isinstance(modes, list) and modes:
            self.tk_vars["display_mode"].set(modes[0])
        elif isinstance(modes, str):
            self.tk_vars["display_mode"].set(modes)

# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[DEBUG] starting SimApp ...")
    app = SimApp()
    app.mainloop()
