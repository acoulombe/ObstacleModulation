import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM

# Make Environment Grid
dim = 20
cell_count = 51
cell_size = dim / (cell_count - 1)
cell_mid_idx = cell_count // 2

x = np.linspace(-dim/2, dim/2, cell_count)
y = np.linspace(-dim/2, dim/2, cell_count)
x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

pos = np.vstack((x_f,y_f)).T

# Make Convex pair
num_vertices = 15

# Box vertices
vertices = np.array([
    [1.5, 1],
    [1.5, -1],
    [-1.5, -1],
    [-1.5, 1],
])

# Random Convex hull
# rand_vertices = (np.random.random((num_vertices,2)) - 0.5) * 4
rand_vertices = np.array([
    [1.5,1.5],
    [-1.5,1],
    [-1.5,-1],
    [1.5,-1.5],
    [2.5,0],
])
obs = OM.ConvexConvex(np.array([5, 3]), rand_vertices, np.eye(2), safety_factor=1.2, reactivity=1)

# Dynamics
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

dist = np.zeros(pos.shape[0])
theta = np.linspace(-np.pi, np.pi, cell_count)

sdf = np.zeros((cell_count, cell_count, cell_count))

for j in range(theta.shape[0]):
    for i in range(pos.shape[0]):
        R = np.array([
            [np.cos(theta[j]), np.sin(theta[j])],
            [-np.sin(theta[j]), np.cos(theta[j])],
        ])
        body = (R @ vertices.T).T

        dist[i] = obs.sdf(pos[i], pos[i]+body)

    dist_2d = dist.reshape((cell_count, cell_count))
    sdf[j] = dist_2d

import pickle

pickle.dump([sdf, dim, cell_count, vertices, obs], open("rotational_experiment_data.p", "wb"))
