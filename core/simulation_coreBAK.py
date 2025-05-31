class SpermSimulation:

    def __init__(self, constants):
        self.constants = constants
        float_keys = ['spot_angle', 'vol', 'sperm_conc', 'vsl', 'deviation', 'surface_time', 'gamete_r', 'sim_min', 'sample_rate_hz']
        int_keys = ['sim_repeat']
        for key in float_keys:
            if key in self.constants and (not isinstance(self.constants[key], float)):
                try:
                    self.constants[key] = float(self.constants[key])
                except Exception:
                    print(f'[WARNING] {key} = {self.constants[key]} をfloat変換できませんでした')
        for key in int_keys:
            if key in self.constants and (not isinstance(self.constants[key], int)):
                try:
                    self.constants[key] = int(float(self.constants[key]))
                except Exception:
                    print(f'[WARNING] {key} = {self.constants[key]} をint変換できませんでした')
        self.constants = calculate_derived_constants(self.constants)

    def is_vector_meeting_egg(self, base_position, temp_position, egg_center, gamete_r):
        vector = temp_position - base_position
        distance_base = LA.norm(base_position - egg_center)
        distance_tip = LA.norm(temp_position - egg_center)
        if distance_base <= gamete_r or distance_tip <= gamete_r:
            return True
        f = base_position - egg_center
        a = vector @ vector
        b = 2 * (f @ vector)
        c = f @ f - gamete_r ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        if 0 <= t1 <= 1 or 0 <= t2 <= 1:
            return True
        return False

    def run(self, sim_repeat: int, surface_time: float, sample_rate_hz: int):
        """
        sim_repeat 回シミュレーションを実行して
        self.trajectory に各精子の N×3 軌跡配列を格納する。
        座標・距離の単位は **mm** で統一。
        """
        print('[DEBUG] SpermSimulation パラメータ:', self.constants)
        shape = self.constants.get('shape', 'cube')
        if shape == 'cube':
            shape_obj = CubeShape(self.constants)
        elif shape == 'spot':
            shape_obj = SpotShape(self.constants)
        elif shape == 'drop':
            shape_obj = DropShape(self.constants)
        elif shape == 'ceros':
            shape_obj = CerosShape(self.constants)
        else:
            raise ValueError(f'Unsupported shape: {shape}')
        number_of_sperm = self.constants['number_of_sperm']
        number_of_steps = self.constants['number_of_steps']
        trajectory = np.zeros((number_of_sperm, number_of_steps, 3))
        step_len = self.constants['step_length']
        seed_val = self.constants.get('seed_number')
        try:
            if seed_val is not None and str(seed_val).lower() != 'none':
                seed_int = int(seed_val)
                np.random.seed(seed_int)
                rng = np.random.default_rng(seed_int)
            else:
                rng = np.random.default_rng()
        except Exception:
            rng = np.random.default_rng()
        egg_x, egg_y, egg_z = _egg_position(self.constants)
        egg_center = np.array([egg_x, egg_y, egg_z])
        gamete_r = self.constants['gamete_r']
        intersection_records = []
        self.trajectory = []
        prev_states = ['inside' for _ in range(number_of_sperm)]
        bottom_modes = [False for _ in range(number_of_sperm)]
        stick_statuses = [0 for _ in range(number_of_sperm)]
        surface_modes = [False for _ in range(number_of_sperm)]
        for rep in range(int(sim_repeat)):
            for i in range(number_of_sperm):
                pos = shape_obj.initial_position()
                traj = [pos.copy()]
                vec = rng.normal(size=3)
                vec /= np.linalg.norm(vec) + 1e-12
                for j in range(number_of_steps):
                    if j > 0:
                        vec = _perturb_direction(vec, self.constants['deviation'], rng)
                    candidate = pos + vec * step_len
                    if shape == 'drop':
                        status = IO_check_drop(candidate, stick_status, self.constants)
                        if status == IOStatus.POLYGON_MODE:
                            stick_status = self.constants['surface_time'] / self.constants['sample_rate_hz']
                            normal = pos / (np.linalg.norm(pos) + 1e-12)
                            vec = vec - np.dot(vec, normal) * normal
                            vec /= np.linalg.norm(vec) + 1e-12
                        else:
                            stick_status = max(0, stick_status - 1)
                    pos = candidate
                    trajectory[i, j] = pos
                    if shape == 'spot' and bottom_modes[i] and (stick_statuses[i] > 0):
                        vec[2] = 0.0
                        vec /= np.linalg.norm(vec) + 1e-12
                    candidate = pos + vec * step_len
                    if j > 0 and self.is_vector_meeting_egg(traj[-1], candidate, egg_center, gamete_r):
                        intersection_records.append((i, j))
                    if shape == 'drop':
                        base_pos = pos
                        move_len = step_len
                        status = _io_check_drop(candidate, self.constants, base_pos)
                        if status == 'outside':
                            vec, base_pos, stick_statuses[i] = _handle_drop_outside(vec, base_pos, self.constants, surface_time, sample_rate_hz, stick_statuses[i])
                            if stick_statuses[i] > 0:
                                surface_modes[i] = True
                            candidate = base_pos + vec * move_len
                    elif shape == 'spot':
                        prev = prev_states[i]
                        candidate, status, bottom_hit = _io_check_spot(pos, candidate, self.constants, prev, stick_statuses[i])
                        prev_states[i] = status
                        if bottom_hit or status in [SpotIO.SPOT_BOTTOM, SpotIO.POLYGON_MODE]:
                            bottom_modes[i] = True
                            if stick_statuses[i] == 0:
                                stick_statuses[i] = int(self.constants['surface_time'] / self.constants['sample_rate_hz'])
                        vec = (candidate - pos) / step_len
                    disp_len = np.linalg.norm(candidate - pos)
                    max_len = step_len
                    if disp_len > max_len * 1.05:
                        print(f'[ERROR] displacement {disp_len} mm exceeds step_length {max_len} mm at rep={rep}, sperm={i}, step={j}')
                        print(f'pos={pos}, candidate={candidate}, vec={vec}')
                        raise RuntimeError('step length exceeded')
                    pos = candidate
                    traj.append(pos.copy())
                    if shape == 'spot':
                        if bottom_modes[i]:
                            if stick_statuses[i] > 0:
                                stick_statuses[i] -= 1
                            if stick_statuses[i] == 0:
                                bottom_modes[i] = False
                    if shape == 'drop' and surface_modes[i]:
                        if stick_statuses[i] > 0:
                            stick_statuses[i] -= 1
                        if stick_statuses[i] == 0:
                            surface_modes[i] = False
                    if rep == 0 and i == 0 and (j == 0):
                        print(f'[DEBUG] 1step_disp(mm) = {np.linalg.norm(vec * step_len):.5f}')
                self.trajectory.append(np.vstack(traj))
        self.trajectories = np.array(self.trajectory)
        print(f'[DEBUG] run完了: sperm={len(self.trajectory)}, steps={number_of_steps}, step_len={step_len} mm')
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
            print('[WARNING] 軌跡データがありません。run()実行後にplot_trajectoriesしてください。')
            return
        all_mins = [constants['x_min'], constants['y_min'], constants['z_min']]
        all_maxs = [constants['x_max'], constants['y_max'], constants['z_max']]
        global_min = min(all_mins)
        global_max = max(all_maxs)
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        ax_xy, ax_xz, ax_yz = axes
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        n_sperm = min(len(trajectories), max_sperm)
        egg_x, egg_y, egg_z = _egg_position(constants)
        for ax, (x, y) in zip(axes, [(egg_x, egg_y), (egg_x, egg_z), (egg_y, egg_z)]):
            ax.add_patch(patches.Circle((x, y), radius=constants.get('gamete_r', 0), facecolor='yellow', alpha=0.8, ec='gray', linewidth=0.5))
        for i in range(n_sperm):
            ax_xy.plot(trajectories[i][:, 0], trajectories[i][:, 1], color=colors[i % len(colors)])
        ax_xy.set_xlim(global_min, global_max)
        ax_xy.set_ylim(global_min, global_max)
        ax_xy.set_aspect('equal')
        ax_xy.set_xlabel('X')
        ax_xy.set_ylabel('Y')
        ax_xy.set_title('XY projection')
        for i in range(n_sperm):
            ax_xz.plot(trajectories[i][:, 0], trajectories[i][:, 2], color=colors[i % len(colors)])
        ax_xz.set_xlim(global_min, global_max)
        ax_xz.set_ylim(global_min, global_max)
        ax_xz.set_aspect('equal')
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title('XZ projection')
        for i in range(n_sperm):
            ax_yz.plot(trajectories[i][:, 1], trajectories[i][:, 2], color=colors[i % len(colors)])
        ax_yz.set_xlim(global_min, global_max)
        ax_yz.set_ylim(global_min, global_max)
        ax_yz.set_aspect('equal')
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.set_title('YZ projection')
        param_summary = ', '.join((f'{k}={constants.get(k)}' for k in ['shape', 'vol', 'sperm_conc', 'vsl', 'deviation']))
        param_summary2 = ', '.join((f'{k}={constants.get(k)}' for k in ['surface_time', 'egg_localization', 'gamete_r', 'sim_min', 'sample_rate_hz', 'sim_repeat']))
        fig.suptitle(f'{param_summary}\n{param_summary2}', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        '\n        シミュレーションで記録したself.trajectory（リスト of N×3 配列）を可視化します。\n        max_sperm: 表示する精子軌跡の最大本数\n        save_path: Noneならこのスクリプトと同じ階層のFigs_and_Moviesに自動保存\n        '
        trajectories = self.trajectory
        if not trajectories or len(trajectories) == 0:
            print('[WARNING] 軌跡データがありません。run()実行後にplot_trajectoriesしてください。')
            return
        n_plot = min(max_sperm, len(trajectories))
        perc_shown = n_plot / len(trajectories) * 100
        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        ax_labels = [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
        idxs = [(0, 1), (0, 2), (1, 2)]
        for ax, (label_x, label_y), (i, j) in zip(axes, ax_labels, idxs):
            for t in trajectories[:n_plot]:
                ax.plot(t[:, i], t[:, j], alpha=0.7)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.set_aspect('equal')
            ax.set_title(f'{label_x}{label_y} projection')
        param_summary = ', '.join((f'{k}={self.constants.get(k)}' for k in ['shape', 'vol', 'sperm_conc', 'vsl', 'deviation']))
        param_summary2 = ', '.join((f'{k}={self.constants.get(k)}' for k in ['surface_time', 'egg_localization', 'gamete_r', 'sim_min', 'sample_rate_hz', 'sim_repeat']))
        fig.suptitle(f'{param_summary}\n{param_summary2}', fontsize=12)
        fig.text(0.99, 0.01, f'※ 表示は全体の{perc_shown:.1f}%（{n_plot}本/{len(trajectories)}本）', ha='right', fontsize=10, color='gray')
        fig.tight_layout(rect=[0, 0.03, 1, 0.92])
        import datetime
        dtstr = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.dirname(__file__)
        figs_dir = os.path.join(base_dir, 'figs_and_movies')
        os.makedirs(figs_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(figs_dir, f'trajectory_{dtstr}.png')
        else:
            filename = os.path.basename(save_path)
            save_path = os.path.join(figs_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f'[INFO] 軌跡画像を保存しました: {save_path}')

    def plot_movie_trajectories(self, save_path=None, fps: int=5):
        """Animate recorded trajectories and save to a movie file."""
        import numpy as np
        from matplotlib.animation import FuncAnimation
        trajectories = np.array(self.trajectory)
        if trajectories is None or len(trajectories) == 0:
            print('[WARNING] 軌跡データがありません。run()実行後にplot_movie_trajectoriesしてください。')
            return None
        n_sperm, n_frames, _ = trajectories.shape
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        const = self.constants
        ax.set_xlim(const['x_min'], const['x_max'])
        ax.set_ylim(const['y_min'], const['y_max'])
        ax.set_zlim(const['z_min'], const['z_max'])
        ax.set_box_aspect([const['x_max'] - const['x_min'], const['y_max'] - const['y_min'], const['z_max'] - const['z_min']])
        lines = [ax.plot([], [], [], lw=0.7)[0] for _ in range(n_sperm)]

        def init():
            for ln in lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
            return lines

        def update(frame):
            for i, ln in enumerate(lines):
                ln.set_data(trajectories[i, :frame + 1, 0], trajectories[i, :frame + 1, 1])
                ln.set_3d_properties(trajectories[i, :frame + 1, 2])
            return lines
        anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=1000 / fps, blit=False)
        base_dir = os.path.dirname(__file__)
        mov_dir = os.path.join(base_dir, 'figs_and_movies')
        os.makedirs(mov_dir, exist_ok=True)
        if save_path is None:
            dtstr = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(mov_dir, f'trajectory_{dtstr}.mp4')
        else:
            filename = os.path.basename(save_path)
            save_path = os.path.join(mov_dir, filename)
        try:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        except Exception as e:
            print(f'[WARN] ffmpeg保存失敗 ({e}) → pillow writerで再試行')
            try:
                anim.save(save_path, writer='pillow', fps=fps)
            except Exception as e2:
                print(f'[ERROR] pillow writerでも保存に失敗: {e2}')
                return None
        plt.close(fig)
        print(f'[INFO] 動画を保存しました: {save_path}')
        return save_path
        return (np.array(traj), intersection_records)