import matplotlib.pyplot as plt
import numpy as np
import ObstacleModulation as OM
from matplotlib import scale

# Obstacle
env = 1
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
cell_count = 25

x = np.linspace(-dim/2, dim/2, cell_count)
y = np.linspace(-dim/2, dim/2, cell_count)
x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

# Get Modulations
pos = np.vstack((x_f,y_f)).T

M = box5.get_modulation_matrix(pos)
E = box5.get_basis_matrix(pos)
D = box5.get_diagonal_matrix(pos)
gammas = box5.gamma_func(pos).reshape(cell_count, cell_count)
dist = box5.sdf(pos).reshape(cell_count, cell_count)

fig, ax = plt.subplots(2, 3)

ax[0,0].set_title("Modulation Frame")
ax[0,0].quiver(pos[:,0], pos[:,1], M[:,0,0], M[:,1,0], color="b")
ax[0,0].quiver(pos[:,0], pos[:,1], M[:,0,1], M[:,1,1], color="g")

ax[0,1].set_title("Basis Frame")
ax[0,1].quiver(pos[:,0], pos[:,1], E[:,0,0], E[:,1,0], color="b")
ax[0,1].quiver(pos[:,0], pos[:,1], E[:,0,1], E[:,1,1], color="g")

ax[0,2].set_title("Gamma field")
ax[0,2].contourf(x, y, gammas, 50, cmap='coolwarm')
CS_g = ax[0,2].contour(x, y, gammas, 50, colors='k')
ax[0,2].clabel(CS_g, inline=True, fontsize=5)


ax[1,0].set_title("SDF")
ax[1,0].contourf(x, y, dist, 50, cmap='coolwarm')
CS_dist = ax[1,0].contour(x, y, dist, 50, colors='k')
ax[1,0].clabel(CS_dist, inline=True, fontsize=5)

ax[1,1].set_title("SVD Frame")
Vs = np.zeros((pos.shape[0], pos.shape[1], pos.shape[1]))
for i in range(pos.shape[0]):
    U, S, V = np.linalg.svd(M[i,:,:])
    Vs[i,:,:] = V

ax[1,1].quiver(pos[:,0], pos[:,1], Vs[:,0,0], Vs[:,1,0], color="b")
ax[1,1].quiver(pos[:,0], pos[:,1], Vs[:,0,1], Vs[:,1,1], color="g")

ax[1,2].set_title("")


# Plot Obstacles in each subplot
box5.plot_obstacle(ax[0,0])
box5.plot_obstacle(ax[0,1])
box5.plot_obstacle(ax[0,2])
box5.plot_obstacle(ax[1,0])
box5.plot_obstacle(ax[1,1])
box5.plot_obstacle(ax[1,2])

plt.show()
