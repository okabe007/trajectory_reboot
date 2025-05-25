import matplotlib.pyplot as plt
import numpy as np

def plot_2d_trajectories(traj: np.ndarray, max_sperm: int = 5):
    """
    複数の精子軌跡をxy平面上にプロットする。
    
    Parameters:
    traj : np.ndarray
        軌跡配列。形状は (num_sperm, num_steps, 3)
    max_sperm : int
        表示する精子の最大数（0〜max_sperm-1 の軌跡を描画）
    """
    num_sperm = min(traj.shape[0], max_sperm)

    plt.figure(figsize=(10, 4))
    for i in range(num_sperm):
        x = traj[i, :, 0]
        y = traj[i, :, 1]
        plt.plot(x, y, label=f"Sperm {i}")
        plt.plot(x[0], y[0], 'go')  # Start point
        plt.plot(x[-1], y[-1], 'ro')  # End point

    plt.xlabel("X position (μm)")
    plt.ylabel("Y position (μm)")
    plt.title("Sperm Trajectories (2D)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
