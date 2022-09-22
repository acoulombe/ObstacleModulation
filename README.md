# `ObstacleModulation`

Dynamic Obstacle Avoidance through Obstacle Modulation to stretch space around objects to not collide with them and arrive at the goal.

## Installation
The package can be installed either by cloning the repository and adding it to your python packages, or directly adding the package to your python packages. To clone the repository and place it into your python site-packages, run
```
git clone https://github.com/acoulombe/ObstacleModulation.git
cd ObstacleModulation
pip3 install .
```

For directly adding the package to your python packages. run
```
pip3 install https://github.com/acoulombe/ObstacleModulation.git@main
```

To use the package and be able to develop with it, it is beneficial to add the package as an editable package in your python site-packages. This is done with the following commands:
```
git clone https://github.com/acoulombe/ObstacleModulation.git
cd ObstacleModulation
pip3 install -e .
```

Pre-requisites are handle by `setup.py` and installed into the site-packages as the package is added through pip.

## Quickstart
The package is composed on multiple object classes
- Obstacle
- Sphere
- Cuboid
- ConvexHull
- ConvexConvex
- SphereInternal
- CuboidInternal
- ObstacleAvoidance

The class `Obstacle` is the base class that the other classes inherit, with the exception of `ConvexConvex`. The class has all the generic methods implemented for all possible star-shaped obstacles and the abstract methods that must be implemented for new classes of star-shaped obstacles.

To import the package into a python script and use the various objects, they can be included by
```py
import numpy as np
import ObstacleModulation as OM

box = OM.Cuboid(np.zeros(2), np.eye(2), np.ones(2), safety_factor=np.array([1.1, 1.2]), reactivity=1)

internal_box = OM.CuboidInternal(np.zeros(2), np.eye(2), np.ones(2), safety_factor=np.array([1.1, 1.2]), reactivity=1)

sphere = OM.Sphere(np.zeros(2), 1, safety_factor=1.2, reactivity=1, repulsion_coeff=2)

internal_sphere = OM.SphereInternal(np.zeros(2), 1, safety_factor=1.2, reactivity=1)

vertices = np.array([
    [2,1],
    [-2,1],
    [-2,-1],
    [2,-1],
])
convex_obs = OM.ConvexHull(np.zeros(2), vertices, np.eye(2), safety_factor=1.2, reactivity=1)
```

### Single Obstacles

Obstacles can provide collision detection by using their `check_collision` method
```py
in_collison = sphere.check_collision(pos)
```

To use an obstacle to modulate a dynamic system motion field, the `get_modulation_matrix` is used to the matrix to modify the system dynamics. It can be used as follows:
```py
def dynamics(pos, goal):
    v = -(pos - goal)
    return v

dyn = dynamics(pos, goal)

modulation = sphere.get_modulation_matrix(pos)
mod_dyn = np.matmul(modulation, dyn)
```
or, to remove modulation artifacts and use shorter paths:
```py
modulation = sphere.get_modulation_matrix(pos, dyn, tail_effects=False)
mod_dyn = np.matmul(modulation, dyn)
```

A useful tool for visualizing the obstacles is the `plot_obstacle` method. 
```py
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
s.plot_obstacle(ax)
plt.show()
```
### Obstacle avoidance in complex environments
For obstacle avoidance in complex environments, the class `ObstacleAvoidance` is used. The class takes in a model of the environment composed of obstacles of the other obstacle classes. Constructing the environment is done through the `add_obstacle` method.
```py
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
octagon = OM.ConvexHull(np.array([3, 0]), vertices, R, safety_factor=1.2, reactivity=1)

vertices = np.array([
    [1.5,1.5],
    [-1.5,1],
    [-1.5,-1],
    [1.5,-1.5],
    [2.5,0],
])

pentagon = OM.ConvexHull(np.array([-4, 5]), vertices, R, safety_factor=1.2, reactivity=1)

vertices = np.array([
    [2,1],
    [-2,1],
    [-2,-1],
    [2,-1],
])

box = OM.ConvexHull(np.array([-5, -3]), vertices, R, safety_factor=1.2, reactivity=1)

obs_avoid = OM.ObstacleAvoidance()
obs_avoid.add_obstacle(box)
obs_avoid.add_obstacle(pentagon)
obs_avoid.add_obstacle(octagon)
```

The constructed environment has a helper function to visualize the environment called `plot_environment`. The method plots each of the individual obstacles in the object
```py
fig, ax = plt.subplots()
obs_avoid.plot_environment(ax)
plt.show()
```

To make the motion fields by the obstacle avoidance object, the dynamics of the system are required. This is done by passing a method for computing the system dynamics with one argument, which is the position of the robot.
```py
def dynamics(pos):
    goal = np.array([[0,0]])
    v = -(pos - goal)
    return v

obs_avoid.set_system_dynamics(dynamics)
```

Once the dynamics and the enivornment are provided, the robot actions can easily be computed through the `get_action` method
```py
mod_dyn = obs_avoid.get_action(pos)
```
Using the modulated dynamics, the robotic system will avoid collisions with anything modelled in the environment.

For plotting what the resulting motion field would look like, the following script produces a streamline plot of the environment
```py
splits = 151
x = np.linspace(-10, 10, splits)
y = np.linspace(-10, 10, splits)
z = np.zeros((splits, splits, 2))

x, y = np.meshgrid(x, y)
x_f = x.flatten()
y_f = y.flatten()

in_collision = obs_avoid.check_collision(pos)
pos_valide = pos[~in_collision]

mod_dyn = obs_avoid.get_action(pos_valide)

tmp = np.zeros(pos.shape)
tmp[~in_collision] = mod_dyn.reshape(-1, 2)
tmp[in_collision] = np.nan      # Make in collision positions NAN so they are ignored
mod_dyn = tmp.reshape(z.shape)

# Plot the flow field
fig, ax = plt.subplots()
ax.streamplot(x, y, mod_dyn[:,:,0], mod_dyn[:,:,1], density=5, color='b')
obs_avoid.plot_environment(ax)
ax.scatter(0, 0, s=50, color='r', label='Goal')
plt.show()
```
The resulting plot would look like something as follows:
![motion_field](/examples/images/Multiple_convexhull_flowfield.png)

More example motion field can be found under `./examples/images/`.


## Examples

Example for how to use the library can be found in `./examples/`. Examples are organized by the different obstacle classes, cluttered environments using the obstacle avoidance class and various experiments.


## References

>Khansari-Zadeh, Seyed Mohammad, and Aude Billard. "A dynamical system approach to realtime obstacle avoidance." Autonomous Robots 32.4 (2012): 433-454.

>Khansari-Zadeh, S. Mohammad, and Aude Billard. "Realtime avoidance of fast moving objects: A dynamical system-based approach." Electronic proc. of the Workshop on Robot Motion Planning: Online, Reactive, and in Real-Time [IROS'2012]. No. POST_TALK. 2012.

>Huber, Lukas, Aude Billard, and Jean-Jacques Slotine. "Avoidance of convex and concave obstacles with convergence ensured through contraction." IEEE Robotics and Automation Letters 4.2 (2019): 1462-1469.

>Huber, Lukas, Jean-Jacques Slotine, and Aude Billard. "Avoiding Dense and Dynamic Obstacles in Enclosed Spaces: Application to Moving in Crowds." IEEE Transactions on Robotics (2022).

>Huber, Lukas, Aude Billard, and Jean-Jacques Slotine. "Fast Obstacle Avoidance Based on Real-Time Sensing." arXiv preprint arXiv:2205.04928 (2022).

## Author
- Alexandre Coulombe - alexandre.coulombe@mail.mcgill.ca