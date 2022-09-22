import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
theta = 45 * np.pi/180
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

box = OM.Cuboid(np.array([3, 0]), R, np.array([4,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1, repulsion_coeff=2)

splits = 501
x = np.linspace(-10, 10, splits)
y = np.linspace(-10, 10, splits)
z = np.zeros((splits, splits, 2))

x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T

in_collision = box.check_collision(pos)
pos_valide = pos[~in_collision]

goal = np.array([[0,0]])
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

modulation = box.get_modulation_matrix(pos_valide, dynamics(pos_valide, goal), tail_effects=True)

# Get Modulated Dynamics
dyn = dynamics(pos_valide, goal).reshape(-1, 2, 1)
mod_dyn = np.matmul(modulation, dyn)
tmp = np.zeros(pos.shape)
tmp[~in_collision] = mod_dyn.reshape(-1, 2)
tmp[in_collision] = np.nan      # Make in collision positions NAN so they are ignored
mod_dyn = tmp.reshape(z.shape)

# Plot the flow field
fig, ax = plt.subplots()
ax.streamplot(x, y, mod_dyn[:,:,0], mod_dyn[:,:,1], density=5, color='b')
box.plot_obstacle(ax)
ax.scatter(goal[:,0], goal[:,1], s=50, color='r', label='Goal')
plt.show()
