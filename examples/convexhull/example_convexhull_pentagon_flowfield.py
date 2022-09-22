import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
theta = 90 * np.pi/180
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

vertices = np.array([
    [1.5,1.5],
    [-1.5,1],
    [-1.5,-1],
    [1.5,-1.5],
    [2.5,0],
])

pentagon = OM.ConvexHull(np.array([3, 0]), vertices, R, safety_factor=1.2, reactivity=1)

splits = 151
x = np.linspace(-10, 10, splits)
y = np.linspace(-10, 10, splits)
z = np.zeros((splits, splits, 2))

x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T

in_collision = pentagon.check_collision(pos)
pos_valide = pos[~in_collision]

dist = pentagon.gamma_func(pos).reshape((splits, splits))

# fig, ax = plt.subplots()
# CT = ax.contourf(x, y, dist, 50, cmap='coolwarm')
# ax.contour(x, y, dist, 20, colors='k')
# fig.colorbar(CT)
# plt.show()

# plt.imshow(in_collision.reshape((splits, splits)))
# plt.show()

# fig, ax = plt.subplots()
# pentagon.plot_obstacle(ax)
# plt.show()

goal = np.array([[0,0]])
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

# Get Modulated Dynamics
modulation = pentagon.get_modulation_matrix(pos_valide, dynamics(pos_valide, goal), tail_effects=True)

dyn = dynamics(pos_valide, goal).reshape(-1, 2, 1)
mod_dyn = np.matmul(modulation, dyn)
tmp = np.zeros(pos.shape)
tmp[~in_collision] = mod_dyn.reshape(-1, 2)
tmp[in_collision] = np.nan      # Make in collision positions NAN so they are ignored
mod_dyn = tmp.reshape(z.shape)

# Plot the flow field
fig, ax = plt.subplots()
ax.streamplot(x, y, mod_dyn[:,:,0], mod_dyn[:,:,1], density=5, color='b')
pentagon.plot_obstacle(ax)
ax.scatter(goal[:,0], goal[:,1], s=50, color='r', label='Goal')
plt.show()
