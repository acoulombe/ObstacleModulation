import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM
from matplotlib.patches import Polygon

# Obstacle
theta = 90 * np.pi/180
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
pos = [10, 1]

# Plot the trajectory
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-5, 15])
ax.set_ylim([-10, 10])

pos = np.array(pos).reshape(-1,2)
obs.plot_obstacle(ax)
obs2.plot_obstacle(ax)

body = pos + (R @ vertices.T).T
rectangle = Polygon(body, color='b')
ax.add_patch(rectangle)

def sdf(pos, theta, shape):
    R_new = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])
    new_points = (R_new @ shape.T).T + pos
    sd = obs_avoid.get_sdf(pos, new_points)
    return sd

alpha = 0.01
h = 0.01
new_theta = theta
for t in range(path_length):
    rectangle.remove()

    sd_p = sdf(pos, new_theta + h/2, vertices)[0]
    sd_n = sdf(pos, new_theta - h/2, vertices)[0]
    new_theta = new_theta + alpha * (sd_p - sd_n) / h

    R = np.array([
        [np.cos(new_theta), np.sin(new_theta)],
        [-np.sin(new_theta), np.cos(new_theta)],
    ])

    dyn = dynamics(pos, goal)
    body = pos + (R @ vertices.T).T
    mod_dyn = obs_avoid.get_action(pos, body, dyn, tail_effects=True)
    gammas = obs_avoid.get_gamma(pos, body)
    dist = obs_avoid.get_sdf(pos, body)

    if dist < 0:
        print("Colliding!", end='\r')
    else:
        print("Safe Motion", end='\r')

    pos = pos + mod_dyn * dt

    if np.all(np.isclose(dyn, 0, atol=1e-1)):
        break

    # Plot the obstacle
    rectangle = Polygon(body, color='b')
    ax.add_patch(rectangle)

    plt.draw()
    plt.pause(dt/10)

    # Check if window got closed, if it did stop the program
    if not plt.fignum_exists(1):
        break
