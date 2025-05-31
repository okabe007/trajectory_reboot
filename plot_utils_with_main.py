
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_trajectories(trajs, constants, save_path=None, show=True, max_sperm=None):
    if max_sperm is None:
        max_sperm = trajs.shape[0]

    fig, ax = plt.subplots()
    for s in range(min(max_sperm, trajs.shape[0])):
        traj = trajs[s]
        x = traj[:, 0]
        y = traj[:, 1]
        ax.plot(x, y, label=f"Sperm {s}")
    ax.set_xlim(constants.get("x_min", -1), constants.get("x_max", 1))
    ax.set_ylim(constants.get("y_min", -1), constants.get("y_max", 1))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("2D Trajectories")
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_3d_movie_trajectories(trajs, constants):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for s in range(trajs.shape[0]):
        traj = trajs[s]
        x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
        ax.plot(x, y, z, label=f"Sperm {s}")
    ax.set_xlim(constants.get("x_min", -1), constants.get("x_max", 1))
    ax.set_ylim(constants.get("y_min", -1), constants.get("y_max", 1))
    ax.set_zlim(constants.get("z_min", -1), constants.get("z_max", 1))
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("3D Trajectories")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    print("[DEBUG] plot_utils を直接実行しています")

    # テスト用データの作成
    test_trajs = np.cumsum(np.random.randn(3, 50, 3) * 0.01, axis=1)
    test_constants = {
        "x_min": -0.2, "x_max": 0.2,
        "y_min": -0.2, "y_max": 0.2,
        "z_min": -0.2, "z_max": 0.2,
    }

    plot_2d_trajectories(test_trajs, test_constants)
    plot_3d_movie_trajectories(test_trajs, test_constants)
