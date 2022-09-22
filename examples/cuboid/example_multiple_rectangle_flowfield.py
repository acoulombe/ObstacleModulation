import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
theta = 0 * np.pi/180
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

box1 = OM.Cuboid(np.array([3, 2]), R, np.array([4,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)
box2 = OM.Cuboid(np.array([3, -2]), R, np.array([4,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)
box3 = OM.Cuboid(np.array([-3, 0]), R, np.array([1,4]), safety_factor=np.array([1.2, 1.1]), reactivity=1)

def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(box1)
obs_avoid.add_obstacle(box2)
obs_avoid.add_obstacle(box3)
obs_avoid.set_system_dynamics(dynamics)

splits = 501
x = np.linspace(-10, 10, splits)
y = np.linspace(-10, 10, splits)
z = np.zeros((splits, splits, 2))

x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T

in_collision = obs_avoid.check_collision(pos)
pos_valide = pos[~in_collision]

mod_dyn = obs_avoid.get_action(pos_valide, dynamics(pos_valide))

tmp = np.zeros(pos.shape)
tmp[~in_collision] = mod_dyn.reshape(-1, 2)
tmp[in_collision] = np.nan      # Make in collision positions NAN so they are ignored
mod_dyn = tmp.reshape(z.shape)

# Plot the flow field
fig, ax = plt.subplots()
ax.streamplot(x, y, mod_dyn[:,:,0], mod_dyn[:,:,1], density=5, color='b')
obs_avoid.plot_environment(ax)
ax.scatter(0, 0, s=50, color='r', label='Goal')
plt.show()
