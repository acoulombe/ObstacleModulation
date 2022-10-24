from turtle import color

import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM
import scipy

# Obstacle
# vertices = np.array([
#     [1.5,1.5],
#     [0, 2],
#     [-1.5,1],
#     [-2,0],
#     [-1.5,-1],
#     [0,-2],
#     [1.5,-1.5],
#     [2.5,0],
# ])
# octagon = OM.ConvexHull(np.array([3, 0]), vertices, np.eye(2), safety_factor=1, reactivity=1)

# vertices = np.array([
#     [1.5,1.5],
#     [-1.5,1],
#     [-1.5,-1],
#     [1.5,-1.5],
#     [2.5,0],
# ])

# pentagon = OM.ConvexHull(np.array([-4, 5]), vertices, np.eye(2), safety_factor=1, reactivity=1)

# vertices = np.array([
#     [2,1],
#     [-2,1],
#     [-2,-1],
#     [2,-1],
# ])

# box = OM.ConvexHull(np.array([-5, -3]), vertices, np.eye(2), safety_factor=1, reactivity=1)

# def dynamics(pos):
#     goal = np.array([[0,0]])
#     v = -(pos - goal)
#     return v

# obs_avoid = OM.ObstacleAvoidance()
# obs_avoid.add_obstacle(box)
# obs_avoid.add_obstacle(pentagon)
# obs_avoid.add_obstacle(octagon)

# obs_avoid.set_system_dynamics(dynamics)

R = np.eye(2)

theta1 = 45 * np.pi/180
R1 = np.array([
    [np.cos(theta1), np.sin(theta1)],
    [-np.sin(theta1), np.cos(theta1)],
])

theta2 = -45 * np.pi/180
R2 = np.array([
    [np.cos(theta2), np.sin(theta2)],
    [-np.sin(theta2), np.cos(theta2)],
])

box1 = OM.Cuboid(np.array([3, 2]), R, np.array([4,2]), safety_factor=np.array([1, 1]), reactivity=1)
box2 = OM.Cuboid(np.array([3, -2]), R, np.array([4,2]), safety_factor=np.array([1, 1]), reactivity=1)
box3 = OM.Cuboid(np.array([-3, 0]), R, np.array([1,4]), safety_factor=np.array([1, 1]), reactivity=1)
box4 = OM.Cuboid(np.array([-6, 4]), R1, np.array([1,4]), safety_factor=np.array([1, 1]), reactivity=1)
box5 = OM.Cuboid(np.array([6, 6]), R2, np.array([1,4]), safety_factor=np.array([1, 1]), reactivity=1)
circle1 = OM.Sphere(np.array([0, 5]), 1, safety_factor=1, reactivity=1)
circle2 = OM.Sphere(np.array([0, -5]), 1, safety_factor=1, reactivity=1)
circle3 = OM.Sphere(np.array([5, -7]), 1, safety_factor=1, reactivity=1)
circle4 = OM.Sphere(np.array([-4, -6]), 1, safety_factor=1, reactivity=1)

def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(box1)
obs_avoid.add_obstacle(box2)
obs_avoid.add_obstacle(box3)
obs_avoid.add_obstacle(box4)
obs_avoid.add_obstacle(box5)
obs_avoid.add_obstacle(circle1)
obs_avoid.add_obstacle(circle2)
obs_avoid.add_obstacle(circle3)
obs_avoid.add_obstacle(circle4)
obs_avoid.set_system_dynamics(dynamics)

# Make Environment Grid
dim = 20
cell_count = 501
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

in_collision = obs_avoid.check_collision(pos).reshape(cell_count, cell_count)
dist = obs_avoid.get_sdf(pos).reshape(cell_count, cell_count)
gammas = obs_avoid.get_gamma(pos).reshape(cell_count, cell_count)

# Scan for local minima
W_size = 5
extrema = np.zeros(dist.shape)
for i in range(W_size//2, cell_count - W_size//2):
    for j in range(W_size//2, cell_count - W_size//2):
        extrema[i,j] = dist[i,j] <= np.min(dist[i-W_size//2 : i+W_size//2, j-W_size//2 : j+W_size//2])

# Plot the resulting found normal and tangents
fig, ax = plt.subplots()
CT = ax.contourf(x, y, dist, 50, cmap='coolwarm')
ax.contour(x, y, dist, 50, colors='k')
fig.colorbar(CT)
obs_avoid.plot_environment(ax)
ax.set_xlim(xmin=-dim/2, xmax=dim/2)
ax.set_ylim(ymin=-dim/2, ymax=dim/2)

ax.scatter(x[extrema==True], y[extrema==True], color="purple", s=10)

plt.show()
