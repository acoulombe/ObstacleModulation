import numpy as np
import ObstacleModulation as OM
import matplotlib.pyplot as plt

# Obstacle
R = np.eye(2)

division_wall   = OM.Cuboid(np.array([-2, 3]), R, np.array([16,2]), safety_factor=np.array([1.025, 1.2]), reactivity=1)
bottom_wall     = OM.Cuboid(np.array([3, -3]), R, np.array([14,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)
left_wall       = OM.Cuboid(np.array([-2, 3]), R, np.array([1,14]), safety_factor=np.array([1.2, 1.1]), reactivity=1)
right_wall      = OM.Cuboid(np.array([10, 3]), R, np.array([1,14]), safety_factor=np.array([1.2, 1.1]), reactivity=1)
top_wall        = OM.Cuboid(np.array([3, 9]), R, np.array([14,2]), safety_factor=np.array([1.1, 1.2]), reactivity=1)

def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(division_wall)
obs_avoid.add_obstacle(bottom_wall)
obs_avoid.add_obstacle(left_wall)
obs_avoid.add_obstacle(right_wall)
obs_avoid.add_obstacle(top_wall)
obs_avoid.set_system_dynamics(dynamics)

splits = 501
x = np.linspace(-2, 10, splits)
y = np.linspace(-2, 10, splits)
z = np.zeros((splits, splits, 2))

x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T

in_collision = obs_avoid.check_collision(pos)
pos_valide = pos[~in_collision]

# Sample Trajectories
num_sample = 50
pos = pos_valide[np.random.randint(pos_valide.shape[0], size=num_sample)]
path_length = 2000
dt = 1e-2

traj = [pos]
for t in range(path_length):
    mod_dyn = obs_avoid.get_action(pos, dynamics(pos))
    pos = pos + mod_dyn * dt
    traj.append(pos)


# Plot the trajectories
fig, ax = plt.subplots()

traj = np.array(traj).T
for path in range(pos.shape[0]):
    ax.plot(traj[0,path,:], traj[1,path,:], color='b')
    ax.scatter(traj[0,path,0], traj[1,path,0], s=50, color='r')
    ax.scatter(traj[0,path,-1], traj[1,path,-1], s=50, color='g')

# Plot the flow field
obs_avoid.plot_environment(ax)
ax.scatter(0, 0, s=50, color='r', label='Goal')
ax.set_xlim(-2,10)
ax.set_ylim(-3,9)
plt.show()
