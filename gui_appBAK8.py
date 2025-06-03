
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from controller.simulation_manager import SimulationManager
from tools.plot_utils import plot_trajectories
from tools.movie_utils_gif_ready import render_3d_movie

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sperm Simulation App")

        # メインフレーム
        frame = ttk.Frame(root, padding=10)
        frame.pack()

        # .iniファイルパス入力
        ttk.Label(frame, text="Config Path (.ini):").pack(anchor="w")
        self.config_path = tk.StringVar(value="sperm_config.ini")
        ttk.Entry(frame, textvariable=self.config_path, width=50).pack(anchor="w")

        # 保存形式選択
        self.format_var = tk.StringVar(value="mp4")
        fmt_frame = ttk.LabelFrame(frame, text="保存形式", padding=5)
        fmt_frame.pack(anchor="w", padx=10, pady=5)
        ttk.Radiobutton(fmt_frame, text="MP4", variable=self.format_var, value="mp4").pack(anchor="w")
        ttk.Radiobutton(fmt_frame, text="GIF", variable=self.format_var, value="gif").pack(anchor="w")

        # 保存チェックボックス
        self.save_movie_var = tk.BooleanVar()
        self.save_movie_var.set(True)
        ttk.Checkbutton(frame, text="3Dムービーを保存する", variable=self.save_movie_var).pack(anchor="w", padx=10, pady=5)

        # 実行ボタン
        self.run_button = ttk.Button(frame, text="Run Simulation", command=self.run_simulation_from_gui)
        self.run_button.pack(pady=10)

        # 進捗バーとログ
        self.progress = ttk.Progressbar(frame, length=300, mode='determinate')
        self.progress.pack(padx=10, pady=5)

        self.log_box = scrolledtext.ScrolledText(frame, height=6, width=60, state='disabled')
        self.log_box.pack(padx=10, pady=5)

    def log(self, message: str):
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, message + '
')
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')
        with open("simulation_log.txt", "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def _update_progress(self, current_index: int, total: int):
        percentage = (current_index + 1) / total * 100
        self.progress["value"] = percentage
        self.progress.update_idletasks()

    def run_simulation_from_gui(self):
        threading.Thread(target=self._run_simulation_worker).start()

    def _run_simulation_worker(self):
        ini_path = self.config_path.get()
        if not ini_path or not os.path.isfile(ini_path):
            self.log("[ERROR] .iniファイルが指定されていません。")
            return

        self.log(f"[DEBUG] .iniファイル読込: {ini_path}")
        sim = SimulationManager(config_path=ini_path)
        trajectory, vectors = sim.engine.simulate(on_progress=self._update_progress)

        self.trajectory = trajectory
        self.vectors = vectors
        self.constants = sim.get_constants()

        plot_trajectories(self.trajectory, self.constants)

        if self.save_movie_var.get():
            render_3d_movie(self.trajectory, self.vectors, self.constants,
                            format=self.format_var.get())

        self.progress["value"] = 100
        self.log("[INFO] シミュレーション完了")

if __name__ == "__main__":
    root = tk.Tk()
    app = SimApp(root)
    root.mainloop()
