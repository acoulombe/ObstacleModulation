import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Obstacle
theta_X = 0 * np.pi/180
theta_Y = 0 * np.pi/180
theta_Z = 0 * np.pi/180


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

box = OM.Cuboid(np.array([3, 0, 0]), R, np.array([3,2,1]), safety_factor=np.array([1.1, 1.15, 1.3]), reactivity=1)

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
path_length = 2000

# Get Starting points
pos = []
for i in range(splits):
    for j in range(splits):
        pos += [
            [x[-1], x[i], x[j]],
        ]

# Sample Trajectories
pos = np.array(pos)

traj = [pos]
for t in range(path_length):
    modulation = box.get_modulation_matrix(pos)
    dyn = dynamics(pos, goal).reshape(-1, 3, 1)
    mod_dyn = np.matmul(modulation, dyn).reshape(-1, 3)
    is_zero = np.where(np.all(np.isclose(mod_dyn, 0), axis=1))
    gammas = box.gamma_func(pos)
    flag = np.any(np.logical_or(np.all(np.isclose(mod_dyn, 0), axis=1), np.isclose(gammas, 1))) 
    if flag:
        dyn[is_zero] = np.random.random(dyn.shape)[is_zero]
        mod_dyn = np.matmul(modulation, dyn).reshape(-1, 3)
    pos = pos + mod_dyn * dt
    traj.append(pos)

# Plot the trajectories
fig = plt.figure()
ax = Axes3D(fig)

traj = np.array(traj).T
for path in range(pos.shape[0]):
    ax.scatter(traj[0,path,0], traj[1,path,0], traj[2,path,0], s=50, color='r')
    ax.scatter(traj[0,path,-1], traj[1,path,-1], traj[2,path,-1], s=50, color='g')
    ax.plot(traj[0,path,:], traj[1,path,:], traj[2,path,:], color='b')

# Plot the flow field
box.plot_obstacle(ax)
plt.show()
