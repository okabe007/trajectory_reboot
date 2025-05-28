import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 定数（再現用）
spot_r = 2.606
spot_bottom_height = 1.675
gamete_r = 0.15
egg_center = np.array([0.0, 0.0, spot_bottom_height + gamete_r])

# 軸固定
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(0, 3)
ax.view_init(elev=30, azim=45)

# メディウム（spot の半球）
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi / 2, 40)
x = spot_r * np.outer(np.cos(u), np.sin(v))
y = spot_r * np.outer(np.sin(u), np.sin(v))
z = spot_r * np.outer(np.ones_like(u), np.cos(v)) + spot_bottom_height
ax.plot_surface(x, y, z, color='pink', alpha=0.2)

# 卵子（黄球）
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
ex = egg_center[0] + gamete_r * np.outer(np.cos(u), np.sin(v))
ey = egg_center[1] + gamete_r * np.outer(np.sin(u), np.sin(v))
ez = egg_center[2] + gamete_r * np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(ex, ey, ez, color='yellow', alpha=0.8)

plt.tight_layout()
plt.show()
