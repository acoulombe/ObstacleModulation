import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM

# Obstacle
theta = 45 * np.pi/180
R1 = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

theta = 120 * np.pi/180
R2 = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

theta = 270 * np.pi/180
R3 = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

vertices = np.array([
    [2,1],
    [-2,1],
    [-2,-1],
    [2,-1],
]) - np.array([-1.5, 0])

box1 = OM.ConvexHull(np.array([3, 0]), vertices, R1, safety_factor=1.2, reactivity=1)
box2 = OM.ConvexHull(np.array([3, 0]), vertices, R2, safety_factor=1.2, reactivity=1)
box3 = OM.ConvexHull(np.array([3, 0]), vertices, R3, safety_factor=1.2, reactivity=1)

def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(box1)
obs_avoid.add_obstacle(box2)
obs_avoid.add_obstacle(box3)

obs_avoid.set_system_dynamics(dynamics)

splits = 151
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

mod_dyn = obs_avoid.get_action(pos_valide, vel=dynamics(pos_valide))

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

