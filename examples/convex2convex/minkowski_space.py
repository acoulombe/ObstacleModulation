import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM

# Obstacle
theta = 0 * np.pi/180
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

# Random Convex hull
rand_vertices = np.array([
    [1.5,1.5],
    [-1.5,1],
    [-1.5,-1],
    [1.5,-1.5],
    [2.5,0],
])
obs = OM.ConvexConvex(np.array([5, 3]), rand_vertices, np.eye(2), safety_factor=1, reactivity=1)
obs2 = OM.ConvexConvex(np.array([5, -3]), rand_vertices, np.eye(2), safety_factor=1, reactivity=1)

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(obs)
obs_avoid.add_obstacle(obs2)

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

goal = np.array([[0,0]])
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

dist = np.zeros((cell_count, cell_count)).flatten()
gammas = np.zeros((cell_count, cell_count)).flatten()
mod_dyn = np.zeros(pos.shape)

body = (R @ vertices.T).T

for i in range(pos.shape[0]):
    dist[i] = obs_avoid.get_sdf(pos[i].reshape(1,-1), pos[i] + body)
    gammas[i] = obs_avoid.get_gamma(pos[i].reshape(1,-1), pos[i] + body)

    if dist[i] > 0:
        mod_dyn[i,:] = obs_avoid.get_action(pos[i].reshape(1,-1), pos[i] + body, dynamics(pos[i], goal), tail_effects=True)
    else:
        mod_dyn[i,:] = np.nan

dist = dist.reshape(cell_count, cell_count)
gammas = gammas.reshape(cell_count, cell_count)
mod_dyn = mod_dyn.reshape(cell_count, cell_count, -1)

fig, ax = plt.subplots(1,3)
CT = ax[0].contourf(x, y, dist, 50, cmap='coolwarm')
CS_dist = ax[0].contour(x, y, dist, 50, colors='k')
ax[0].clabel(CS_dist, inline=True, fontsize=5)
fig.colorbar(CT)
obs_avoid.plot_environment(ax[0])
ax[0].set_title("SDF")

CT = ax[1].contourf(x, y, gammas, 50, cmap='coolwarm')
CS_g = ax[1].contour(x, y, gammas, 50, colors='k')
ax[1].clabel(CS_g, inline=True, fontsize=5)
fig.colorbar(CT)
obs_avoid.plot_environment(ax[1])
ax[1].set_title("Gamma Function")

ax[2].streamplot(x, y, mod_dyn[:,:,0], mod_dyn[:,:,1], density=5, color='g')
obs_avoid.plot_environment(ax[2])
ax[2].scatter(0, 0, s=50, color='r', label='Goal')
ax[2].contourf(x, y, dist, 50, cmap='coolwarm')
ax[2].contour(x, y, dist, 50, colors='k')
ax[2].set_title("Motion Field in Minkowski Space")

plt.show()
