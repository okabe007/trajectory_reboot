import numpy as np
from simcore.engine import SimulationEngine
from tools.config_loader import load_config_dict

class SimulationManager:
    def __init__(self, config_path: str = None, constants: dict = None):
        """
        シミュレーションマネージャ
        - GUIからの設定読み込みまたは constants を直接受け取って SimulationEngine を準備
        """
        if constants is not None:
            self.constants = constants
        elif config_path is not None:
            self.constants = load_config_dict(config_path)  # .ini読み込み関数（tools内にある想定）
        else:
            raise ValueError("config_path または constants のどちらかを指定してください")

        self.rng = np.random.default_rng(int(self.constants.get("seed_number", 0)))
        self.engine = SimulationEngine(self.constants)

    def run_simulation(self):
        """
        軌跡とベクトルを生成して返す。
        Returns:
            trajectory: np.ndarray (N精子 × Mステップ × 3)
            vectors: np.ndarray    (N精子 × Mステップ × 3)
        """
        trajectory, vectors = self.engine.simulate()
        return trajectory, vectors

    def get_constants(self) -> dict:
        return self.constants
