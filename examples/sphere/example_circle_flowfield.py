import numpy as np
import ObstacleModulation as OM

# Obstacle
s = OM.Sphere(np.array([3, 0]), 1, safety_factor=1.2, reactivity=1, repulsion_coeff=2)

splits = 501
x = np.linspace(-10, 10, splits)
y = np.linspace(-10, 10, splits)
z = np.zeros((splits, splits, 2))

x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T
in_collision = s.check_collision(pos)
pos_valide = pos[~in_collision] 

def dynamics(pos, goal):
    v = -(pos - goal)
    return v

# Get Modulated Dynamics
goal = np.array([[0,0]])
dyn = dynamics(pos_valide, goal).reshape(-1, 2, 1)
modulation = s.get_modulation_matrix(pos_valide, dynamics(pos_valide, goal), tail_effects=False)
mod_dyn = np.matmul(modulation, dyn)
tmp = np.zeros(pos.shape)
tmp[~in_collision] = mod_dyn.reshape(-1, 2)
tmp[in_collision] = np.nan      # Make in collision positions NAN so they are ignored
mod_dyn = tmp.reshape(z.shape)

# Plot the flow field
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.streamplot(x, y, mod_dyn[:,:,0], mod_dyn[:,:,1], density=5, color='b')
s.plot_obstacle(ax)
ax.scatter(goal[:,0], goal[:,1], s=50, color='r', label='Goal')
plt.show()
