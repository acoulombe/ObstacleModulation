import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

# Obstacle
env = 0
if env == 0:
    vertices = np.array([
        [1.5,1.5],
        [0, 2],
        [-1.5,1],
        [-2,0],
        [-1.5,-1],
        [0,-2],
        [1.5,-1.5],
        [2.5,0],
    ])
    octagon = OM.ConvexHull(np.array([3, 0]), vertices, np.eye(2), safety_factor=1, reactivity=1)

    vertices = np.array([
        [1.5,1.5],
        [-1.5,1],
        [-1.5,-1],
        [1.5,-1.5],
        [2.5,0],
    ])

    pentagon = OM.ConvexHull(np.array([-4, 5]), vertices, np.eye(2), safety_factor=1, reactivity=1)

    vertices = np.array([
        [2,1],
        [-2,1],
        [-2,-1],
        [2,-1],
    ])

    box = OM.ConvexHull(np.array([-5, -3]), vertices, np.eye(2), safety_factor=1, reactivity=1)

    def dynamics(pos):
        goal = np.array([[0,0]])
        v = -(pos - goal)
        return v

    obs_avoid = OM.ObstacleAvoidance()
    obs_avoid.add_obstacle(box)
    obs_avoid.add_obstacle(pentagon)
    obs_avoid.add_obstacle(octagon)

    obs_avoid.set_system_dynamics(dynamics)

elif env == 1:
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
cell_count = 151
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

# Get Normal Basis Vector
filt_dx = np.zeros((3,3))
filt_dx[:,0] = 1
filt_dx[:,-1] = -1

filt_dy = np.zeros((3,3))
filt_dy[0,:] = 1
filt_dy[-1,:] = -1

grad_x = convolve2d(dist, filt_dx, mode='same')
grad_y = convolve2d(dist, filt_dy, mode='same')

# Tangents to Obstacle Surface
def gram_schmidt(X):
    Q, R = np.linalg.qr(X)
    return Q

def get_initial_basis(v):
    S = np.zeros((v.shape[0], v.shape[1], v.shape[1]))
    S[:,:,0] = v
    for i in range(1, v.shape[1]):
        a = np.zeros(v.shape)
        a[np.isclose(v[:,0], 0), 0] = 1
        a[~np.isclose(v[:,0], 0), 0] = -v[~np.isclose(v[:,0], 0),i] / v[~np.isclose(v[:,0], 0),0]
        a[:,i] = 1
        S[:,:,i] = a

    return S

grad = np.array([grad_x.flatten(), grad_y.flatten()]).T
S = get_initial_basis(grad)
basis = gram_schmidt(S)
c = np.sum(basis[:,0] * grad, axis=1)
basis[c < 0, :] *= -1

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

# Get Extrema
minima = np.vstack((x[extrema==True], y[extrema==True])).T
minima_labels = DBSCAN(eps=0.5, min_samples=1).fit(minima)
ref_points = []
for i in range(0, np.max(minima_labels.labels_)+1):
    cluster = minima[minima_labels.labels_ == i]
    ref_points.append(np.mean(cluster, axis=0))

# Plot Extrema and cluster points
ref_points = np.array(ref_points)

ax.scatter(minima[:,0], minima[:,1], color="purple", s=10)
ax.scatter(ref_points[:,0], ref_points[:,1], color="orange", s=20)
ax.set_title("SDF Reference Point Discovery")

# Reconstruct Gamma Function Values
fig, ax = plt.subplots(1,2)
gamma_reconstruct = np.zeros(pos.shape[0])
dist_pos = dist.flatten()
for i in range(pos.shape[0]):
    dist_closest_ref = np.min(np.linalg.norm(pos[i] - ref_points, axis=1))
    gamma_reconstruct[i] = dist_closest_ref / (dist_closest_ref - dist_pos[i])

gamma_reconstruct = gamma_reconstruct.reshape(cell_count, cell_count)

CT = ax[0].contourf(x, y, gamma_reconstruct, 50, cmap='coolwarm')
ax[0].contour(x, y, gamma_reconstruct, 50, colors='k')
fig.colorbar(CT)
obs_avoid.plot_environment(ax[0])
ax[0].set_xlim(xmin=-dim/2, xmax=dim/2)
ax[0].set_ylim(ymin=-dim/2, ymax=dim/2)
ax[0].set_title("Gamma Function Reconstruction")

# True Gamma Function
CT = ax[1].contourf(x, y, gammas, 50, cmap='coolwarm')
ax[1].contour(x, y, gammas, 50, colors='k')
fig.colorbar(CT)
obs_avoid.plot_environment(ax[1])
ax[1].set_xlim(xmin=-dim/2, xmax=dim/2)
ax[1].set_ylim(ymin=-dim/2, ymax=dim/2)
ax[1].set_title("Gamma Function Baseline")


fig, ax = plt.subplots()
gamma_reconstruct = gamma_reconstruct.flatten()
dyn = np.zeros(pos.shape)
Ms = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))
Es = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))

for i in range(pos.shape[0]):
    lambda_r = 1 - 1 / gamma_reconstruct[i]
    lambda_e = 1 + 1 / gamma_reconstruct[i]
    D = np.diag([lambda_r, lambda_e])
    E = basis[i]
    ref_idx = np.argmin(np.linalg.norm(pos[i] - ref_points, axis=1))
    if not np.isclose(np.linalg.norm(pos[i] - ref_points[ref_idx]), 0):
        E[:,0] = (pos[i] - ref_points[ref_idx]) / np.linalg.norm(pos[i] - ref_points[ref_idx])

    M = E @ D @ np.linalg.pinv(E)
    dyn[i] = M @ -pos[i]

    Ms[i,:,:] = M
    Es[i,:,:] = E

pos_cp = np.copy(pos)
pos = pos.reshape(cell_count, cell_count, 2)
dyn = dyn.reshape(cell_count, cell_count, 2)

ax.streamplot(pos[:,:,0], pos[:,:,1], dyn[:,:,0], dyn[:,:,1], density=5, color='b')
obs_avoid.plot_environment(ax)
ax.scatter(0,0,color='r',s=50)

fig, ax = plt.subplots(1,2)
idx = np.argwhere(np.array(range(0, cell_count**2)) % 345 == 0)
ax[0].set_title("Modulation Frame")
ax[0].quiver(pos_cp[idx,0], pos_cp[idx,1], Ms[idx,0,0], Ms[idx,1,0], color="b")
ax[0].quiver(pos_cp[idx,0], pos_cp[idx,1], Ms[idx,0,1], Ms[idx,1,1], color="g")

ax[1].set_title("Basis Frame")
ax[1].quiver(pos_cp[idx,0], pos_cp[idx,1], Es[idx,0,0], Es[idx,1,0], color="b")
ax[1].quiver(pos_cp[idx,0], pos_cp[idx,1], Es[idx,0,1], Es[idx,1,1], color="g")
obs_avoid.plot_environment(ax[0])
obs_avoid.plot_environment(ax[1])

plt.show()
