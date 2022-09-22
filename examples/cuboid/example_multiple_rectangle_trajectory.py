import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
theta = 0 * np.pi/180
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)],
])

box1 = OM.Cuboid(np.array([3, 2]), R, np.array([4,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)
box2 = OM.Cuboid(np.array([3, -2]), R, np.array([4,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)
box3 = OM.Cuboid(np.array([-3, 0]), R, np.array([1,4]), safety_factor=np.array([1.2, 1.1]), reactivity=1)

def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(box1)
obs_avoid.add_obstacle(box2)
obs_avoid.add_obstacle(box3)
obs_avoid.set_system_dynamics(dynamics)


# Get Starting points
splits = 25
x = np.linspace(-10, 10, splits)

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
path_length = 2000
dt = 1e-2

traj = [pos]
for t in range(path_length):
    mod_dyn = obs_avoid.get_action(pos)#, dynamics(pos))
    pos = pos + mod_dyn * dt
    traj.append(pos)


# Plot the trajectories
fig, ax = plt.subplots()

traj = np.array(traj).T
for path in range(pos.shape[0]):
    ax.scatter(traj[0,path,0], traj[1,path,0], s=50, color='r')
    ax.scatter(traj[0,path,-1], traj[1,path,-1], s=50, color='g')
    ax.plot(traj[0,path,:], traj[1,path,:], color='b')

# Plot the flow field
obs_avoid.plot_environment(ax)
plt.show()

