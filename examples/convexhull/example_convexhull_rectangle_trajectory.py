import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
theta = 45 * np.pi/180
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

vertices = np.array([
    [2,1],
    [-2,1],
    [-2,-1],
    [2,-1],
])

box = OM.ConvexHull(np.array([3, 0]), vertices, R, safety_factor=1.2, reactivity=1)

splits = 25
x = np.linspace(-10, 10, splits)

# Dynamics
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

# Goal
goal = np.array([[0,0]])
dt = 1e-2
fig, ax = plt.subplots()

# Get Modulations
path_length = 2000
num_arrows = 5
path_percentage_for_arrow = 0.1


# Get Starting points
pos = []
for i in range(splits):
    pos += [
        [x[i], x[0]],
        [x[0], x[i]],
        [x[i], x[-1]],
        [x[-1], x[i]],
    ]

# Sample Trajectories
pos = np.array(pos)

traj = [pos]
for t in range(path_length):
    modulation = box.get_modulation_matrix(pos)
    dyn = dynamics(pos, goal).reshape(-1, 2, 1)
    mod_dyn = np.matmul(modulation, dyn).reshape(-1, 2)
    is_zero = np.where(np.all(np.isclose(mod_dyn, 0), axis=1))
    gammas = box.gamma_func(pos)
    flag = np.any(np.logical_or(np.all(np.isclose(mod_dyn, 0), axis=1), np.isclose(gammas, 1))) 
    if flag:
        dyn[is_zero] = np.random.random(dyn.shape)[is_zero]
        mod_dyn = np.matmul(modulation, dyn).reshape(-1, 2)
    pos = pos + mod_dyn * dt
    traj.append(pos)

# Plot the trajectories
traj = np.array(traj).T
for path in range(pos.shape[0]):
    ax.scatter(traj[0,path,0], traj[1,path,0], s=50, color='r')
    ax.scatter(traj[0,path,-1], traj[1,path,-1], s=50, color='g')
    ax.plot(traj[0,path,:], traj[1,path,:], color='b')

    for checkpoint in range(0, int(path_length * path_percentage_for_arrow-1), int(path_length * path_percentage_for_arrow/num_arrows)):
        v = traj[:,path,checkpoint+1] - traj[:,path,checkpoint]
        ax.arrow(traj[0,path,checkpoint], traj[1,path,checkpoint], v[0], v[1], shape='full', lw=0, length_includes_head=True, head_width=0.15) 

# Plot the flow field
box.plot_obstacle(ax)
plt.show()