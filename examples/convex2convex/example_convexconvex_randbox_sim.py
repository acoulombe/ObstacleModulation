import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull as ConvHull

# Obstacle
theta_X = 0 * np.pi/180
theta_Y = 45 * np.pi/180
theta_Z = 45 * np.pi/180
num_vertices = 15

R1 = np.array([
    [np.cos(theta_X), np.sin(theta_X), 0],
    [-np.sin(theta_X), np.cos(theta_X), 0],
    [0, 0, 1],
])

R2 = np.array([
    [1, 0, 0],
    [0, np.cos(theta_Z), np.sin(theta_Z)],
    [0, -np.sin(theta_Z), np.cos(theta_Z)],
])

R3 = np.array([
    [np.cos(theta_Y), 0, -np.sin(theta_Y)],
    [0, 1, 0],
    [np.sin(theta_Y), 0, np.cos(theta_Y)],
])

R = R1.dot(R2).dot(R3)

# Box vertices
vertices = np.array([
    [1.5, 1, 0.5],
    [1.5, 1, -0.5],
    [1.5, -1, 0.5],
    [1.5, -1, -0.5],
    [-1.5, 1, 0.5],
    [-1.5, 1, -0.5],
    [-1.5, -1, 0.5],
    [-1.5, -1, -0.5],
])

# Random Convex hull
rand_vertices = (np.random.random((num_vertices,3)) - 0.5) * 4

box = OM.ConvexConvex(np.array([5, 3, 0]), rand_vertices, R, safety_factor=1.2, reactivity=1)

splits = 5
x = np.linspace(-5, 5, splits)

# Dynamics
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

# Goal
goal = np.array([[0,0,0]])
dt = 1e-2

# Get Modulations
path_length = 10000

# Get Starting points
pos = [10, 6, -0.1]

# Plot the trajectory
plt.ion()
fig = plt.figure(1)
ax = Axes3D(fig)
ax.set_xlim(-5, 15)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

pos = np.array(pos).reshape(-1,3)

for t in range(path_length):
    ax.collections.clear()

    modulation = box.get_modulation_matrix(pos, pos + vertices)
    dyn = dynamics(pos, goal).reshape(-1, 3, 1)
    mod_dyn = np.matmul(modulation, dyn).reshape(-1, 3)
    gammas = box.gamma_func(pos, pos + vertices)
    flag = np.any(np.logical_and(np.all(np.isclose(mod_dyn, 0, atol=1e-2)), np.isclose(gammas, 1, atol=1e-3))) 
    if flag:
        dyn = np.random.random(dyn.shape)
        mod_dyn = np.matmul(modulation, dyn).reshape(-1, 3)
    pos = pos + mod_dyn * dt

    if np.all(np.isclose(dyn, 0, atol=1e-1)):
        break

    # Plot the obstacle
    box.plot_obstacle(ax)

    body = pos + vertices
    hull = ConvHull(body)
    # draw the polygons of the convex hull
    for s in hull.simplices:
        v = body[s].reshape(1, 3, 3)
        tri = Poly3DCollection(v, facecolor='red', linewidth=1, edgecolor='k')
        ax.add_collection3d(tri)

    plt.draw()
    plt.pause(dt/10)

    # Check if window got closed, if it did stop the program
    if not plt.fignum_exists(1):
        break
