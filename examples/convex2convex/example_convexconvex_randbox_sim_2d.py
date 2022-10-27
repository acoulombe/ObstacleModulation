import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM
from matplotlib.patches import Polygon

# Obstacle
theta = 0 * np.pi/180
num_vertices = 15

R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])


# Box vertices
vertices = np.array([
    [1.5, 1],
    [1.5, -1],
    [-1.5, -1],
    [-1.5, 1],
])

# Random Convex hull
rand_vertices = (np.random.random((num_vertices,2)) - 0.5) * 4

box = OM.ConvexConvex(np.array([5, 3]), rand_vertices, R, safety_factor=1.1, reactivity=1)

splits = 5
x = np.linspace(-5, 5, splits)

# Dynamics
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

# Goal
goal = np.array([[0,0]])
dt = 1e-2

# Get Modulations
path_length = 10000

# Get Starting points
pos = [10, 6.1]

# Plot the trajectory
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-5, 15])
ax.set_ylim([-10, 10])

pos = np.array(pos).reshape(-1,2)
box.plot_obstacle(ax)

points = pos + vertices
rectangle = Polygon(points, color='b')
ax.add_patch(rectangle)

for t in range(path_length):
    rectangle.remove()

    dyn = dynamics(pos, goal)
    modulation = box.get_modulation_matrix(pos, pos + vertices, dyn, tail_effects=True)
    mod_dyn = np.matmul(modulation, dyn.reshape(-1, 2, 1)).reshape(-1, 2)
    gammas = box.gamma_func(pos, pos + vertices)
    dist = box.sdf(pos, pos + vertices)

    flag = np.any(np.logical_and(np.all(np.isclose(mod_dyn, 0, atol=1e-2)), np.isclose(gammas, 1, atol=1e-3)))
    if flag:
        dyn = np.random.random(dyn.shape)
        mod_dyn = np.matmul(modulation, dyn).reshape(-1, 2)
    pos = pos + mod_dyn * dt

    if np.all(np.isclose(dyn, 0, atol=1e-1)):
        break

    # Plot the obstacle
    points = pos + vertices
    rectangle = Polygon(points, color='b')
    ax.add_patch(rectangle)

    plt.draw()
    plt.pause(dt/10)

    # Check if window got closed, if it did stop the program
    if not plt.fignum_exists(1):
        break
