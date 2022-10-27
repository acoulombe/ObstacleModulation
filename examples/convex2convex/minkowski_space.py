import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM

# Obstacle
theta = 45 * np.pi/180
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

vertices = np.array([
    [1.5, 1],
    [1.5, -1],
    [-1.5, -1],
    [-1.5, 1],
])

box = OM.ConvexConvex(np.array([4, 0]), vertices, R, safety_factor=1, reactivity=1)


# Make Environment Grid
dim = 20
cell_count = 151
cell_size = dim / (cell_count - 1)
cell_mid_idx = cell_count // 2

def location_to_idx(x):
    return (np.round(x / cell_size) + cell_mid_idx).astype(int)

x = np.linspace(-dim/2, dim/2, cell_count)
y = np.linspace(-dim/2, dim/2, cell_count)
x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T

dist = np.zeros((cell_count, cell_count)).flatten()
gammas = np.zeros((cell_count, cell_count)).flatten()

for i in range(pos.shape[0]):
    dist[i] = box.sdf(pos[i], pos[i] + vertices)
    gammas[i] = box.gamma_func(pos[i], pos[i] + vertices)

dist = dist.reshape(cell_count, cell_count)
gammas = gammas.reshape(cell_count, cell_count)

fig, ax = plt.subplots(1,2)
CT = ax[0].contourf(x, y, dist, 50, cmap='coolwarm')
CS_dist = ax[0].contour(x, y, dist, 50, colors='k')
ax[0].clabel(CS_dist, inline=True, fontsize=5)
fig.colorbar(CT)
box.plot_obstacle(ax[0])

CT = ax[1].contourf(x, y, gammas, 50, cmap='coolwarm')
CS_g = ax[1].contour(x, y, gammas, 50, colors='k')
ax[1].clabel(CS_g, inline=True, fontsize=5)
fig.colorbar(CT)
box.plot_obstacle(ax[1])
plt.show()
