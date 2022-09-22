import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
R = np.eye(2)

division_wall   = OM.Cuboid(np.array([-2, 3]), R, np.array([16,2]), safety_factor=np.array([1.025, 1.2]), reactivity=1)
bottom_wall     = OM.Cuboid(np.array([3, -3]), R, np.array([14,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)
left_wall       = OM.Cuboid(np.array([-2, 3]), R, np.array([1,14]), safety_factor=np.array([1.2, 1.1]), reactivity=1)
right_wall      = OM.Cuboid(np.array([10, 3]), R, np.array([1,14]), safety_factor=np.array([1.2, 1.1]), reactivity=1)
top_wall        = OM.Cuboid(np.array([3, 9]), R, np.array([14,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)

def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(division_wall)
obs_avoid.add_obstacle(bottom_wall)
obs_avoid.add_obstacle(left_wall)
obs_avoid.add_obstacle(right_wall)
obs_avoid.add_obstacle(top_wall)
obs_avoid.set_system_dynamics(dynamics)

splits = 501
x = np.linspace(-2, 10, splits)
y = np.linspace(-2, 10, splits)
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
ax.set_xlim(-2,10)
ax.set_ylim(-3,9)
plt.show()
