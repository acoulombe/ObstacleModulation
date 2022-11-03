import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM
from matplotlib.patches import Polygon

sdf, dim, cell_count, robot, obs = pickle.load(open("rotational_experiment_data.p", "rb"))

x = np.linspace(-dim/2, dim/2, cell_count)
y = np.linspace(-dim/2, dim/2, cell_count)
theta = np.linspace(-np.pi, np.pi, cell_count)
x, y = np.meshgrid(y, theta)

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(sdf < 0, edgecolor='k')
plt.show()

cell_size = dim / (cell_count - 1)
cell_mid_idx = cell_count // 2

idx = int(cell_count * 0.26)

fig, ax = plt.subplots(1,2)

R = np.array([
    [np.cos(theta[idx]), np.sin(theta[idx])],
    [-np.sin(theta[idx]), np.cos(theta[idx])],
])
R = np.eye(2)
points = (R @ robot.T).T + np.array([4,0])
# robot_model = Polygon(points, color='r')
# ax[0].add_patch(robot_model)
# obs.plot_environment(ax[0])
# ax[0].set_xlim(xmin=-dim/2, xmax=dim/2)
# ax[0].set_ylim(ymin=-dim/2, ymax=dim/2)
# ax[0].set_title(f"Environment")

# CT = ax[1].contourf(x, y, sdf[:,idx,:], 50, cmap='coolwarm')
# ax[1].contour(x, y, sdf[:,idx,:], 50, colors='k')
# fig.colorbar(CT)
# ax[1].set_title(f"SDF")
# plt.show()

import Collision
from scipy.spatial import ConvexHull as ConvHull

num_vertices = 10
rand_vertices = (np.random.random((num_vertices,2)) - 0.5) * 4 + np.array([5, 3])

sd, D, cache = Collision.signedDistance2d(points, rand_vertices)

def sdf(theta, shape1, shape2):
    R_new = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)],
    ])
    new_points = (R_new @ shape1.T).T + np.array([4,0])
    sd, D, cache = Collision.signedDistance2d(new_points, shape2)
    P1, P2 = Collision.closestPoints(D_new, cache)
    return sd, D, cache, P1, P2

P1 = cache.A[0]
P2 = cache.A[0] + D
err = float("inf")
sd_new = sd
D_new = D
new_theta = 0
h = 0.01
alpha = 0.01
max_iter = 500
iteration = 0
while err > 1e-6 and iteration < max_iter:
    sd_prev = sd_new

    sd_p, _, _, _, _ = sdf(new_theta + h/2, robot, rand_vertices)
    sd_n, _, _, _, _ = sdf(new_theta - h/2, robot, rand_vertices)
    new_theta = new_theta + alpha * (sd_p - sd_n) / h
    sd_new, _, _, _, _ = sdf(new_theta, robot, rand_vertices)

    err = np.abs(sd_new - sd_prev)
    iteration += 1

print(f"Old: {sd} | New {sd_new}")

N = 501
theta = np.linspace(-np.pi, np.pi, N)
dist = np.zeros(N)
for i in range(N):
    R_new = np.array([
        [np.cos(theta[i]), np.sin(theta[i])],
        [-np.sin(theta[i]), np.cos(theta[i])],
    ])
    tmp_points = (R_new @ robot.T).T + np.array([4,0])
    d, D, cache = Collision.signedDistance2d(tmp_points, rand_vertices)
    P1, P2 = Collision.closestPoints(D, cache)
    dist[i] = d


fig, ax = plt.subplots(1,3)
# draw the polygons of the convex hull
hull = ConvHull(points)
ax[0].fill(points[hull.vertices,0], points[hull.vertices,1], color="g")
hull = ConvHull(rand_vertices)
ax[0].fill(rand_vertices[hull.vertices,0], rand_vertices[hull.vertices,1], color="r")
ax[0].set_title("Init Angle 0")

ax[1].fill(rand_vertices[hull.vertices,0], rand_vertices[hull.vertices,1], color="r")
R_new = np.array([
    [np.cos(new_theta), np.sin(new_theta)],
    [-np.sin(new_theta), np.cos(new_theta)],
])
new_points = (R_new @ robot.T).T + np.array([4,0])
hull = ConvHull(new_points)
ax[1].fill(new_points[hull.vertices,0], new_points[hull.vertices,1], color="g")
ax[1].set_title(f"Final Angle {new_theta/np.pi:.2f}$\pi$")
radii = np.linalg.norm(robot, axis=1)
radii = np.unique(radii)
for r in radii:
    c = matplotlib.patches.Circle(np.array([4,0]), r, linestyle="--", fill=False)
    ax[1].add_patch(c)

offset = robot.flatten()
offset = np.unique(offset)
for o in offset:
    c = matplotlib.patches.Circle(np.array([4,0]), o, linestyle="--", fill=False)
    ax[1].add_patch(c)

ax[2].plot(theta, dist, color='b')
ax[2].plot(theta[:-1], (dist[1:]-dist[:-1])/(theta[1:]-theta[:-1]), color='g')
ax[2].scatter([0], [sd], color='g')
ax[2].scatter([new_theta], [sd_new], color='orange')
ax[2].set_title(f"SDF w.r.t. Angle | Iter {iteration}")
plt.show()
