import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import ObstacleModulation as OM
import pickle

sdf, dim, cell_count, robot, obs = pickle.load(open("rotational_experiment_data.p", "rb"))

# ax = plt.figure().add_subplot(projection='3d')
# ax.voxels(sdf < 0, edgecolor='k')
# plt.show()

x = np.linspace(-dim/2, dim/2, cell_count)
y = np.linspace(-dim/2, dim/2, cell_count)
theta = np.linspace(-np.pi, np.pi, cell_count)
x, y = np.meshgrid(x, y)

idx = int(cell_count * 0.26)

fig, ax = plt.subplots(1,2)

R = np.array([
    [np.cos(theta[idx]), np.sin(theta[idx])],
    [-np.sin(theta[idx]), np.cos(theta[idx])],
])
points = (R @ robot.T).T
robot_model = Polygon(points, color='r')
ax[0].add_patch(robot_model)
obs.plot_obstacle(ax[0])
ax[0].set_xlim(xmin=-dim/2, xmax=dim/2)
ax[0].set_ylim(ymin=-dim/2, ymax=dim/2)
ax[0].set_title(f"Environment with Orientation {theta[idx]/np.pi:.2f}$\pi$")

CT = ax[1].contourf(x, y, sdf[idx], 50, cmap='coolwarm')
ax[1].contour(x, y, sdf[idx], 50, colors='k')
fig.colorbar(CT)
ax[1].set_xlim(xmin=-dim/2, xmax=dim/2)
ax[1].set_ylim(ymin=-dim/2, ymax=dim/2)
ax[1].set_title(f"SDF with Orientation {theta[idx]/np.pi:.2f}$\pi$")
plt.show()