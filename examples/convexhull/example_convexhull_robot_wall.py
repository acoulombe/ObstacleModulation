from cmath import isclose
import pybullet as p
import pybullet_data
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
exampledir = os.path.dirname(parentdir)
wrkdir = os.path.dirname(exampledir)
sys.path.append(wrkdir)
import numpy as np
from Kinematics import kinematics
import ObstacleModulation as OM

# Open pybullet simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

## Setup Simulation Stage
p.loadURDF("plane.urdf", [0, 0, 0])

# Set Environment
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

vertices = np.array([
    [0.1,0.025,0.25],
    [0.1,0.025,-0.25],
    [0.1,-0.025,0.25],
    [0.1,-0.025,-0.25],
    [-0.1,0.025,0.25],
    [-0.1,0.025,-0.25],
    [-0.1,-0.025,0.25],
    [-0.1,-0.025,-0.25],
]) * 1000

box = OM.ConvexHull(np.array([250,0,-10]), vertices, R, safety_factor=1.2, reactivity=1)
bullet_box = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.1,0.025,0.25], collisionFramePosition=[0.25,0,-0.01])
p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bullet_box)

# Spawn default robot
robotId = p.loadURDF("URDF/meca_500_r3_steven.urdf", [0, 0, 0], useFixedBase=1)

# Set tracking and loop variables 
p.resetBasePositionAndOrientation(robotId, [0, 0, 0], [0, 0, 0, 1])
num_joints = 6
num_actuator = p.getNumJoints(robotId)
endEffectorIndex = num_joints- 1

## Initialize robot position
for i in range(num_joints):
    p.resetJointState(robotId, i, 0)

jointStates = np.zeros(num_joints)

# Simulation Step and Tracking
p.setGravity(0, 0, -9.8)
t = 0
prevPose = [0, 0, 0]
hasPrevPose = 0
p.setRealTimeSimulation(1)
trailDuration = 15

goal = np.array([200, 100, 200])

while 1:
    # Check keyboard events for actions from user
    key_events = p.getKeyboardEvents()
    for key in key_events:
        if chr(key) == 'r':
            for i in range(num_joints):
                p.resetJointState(robotId, i, 0)
            t = 0
            jointStates = np.zeros(num_joints)

    curr_pose =  kinematics.Mat2Pose(kinematics.forwardKinematics(jointStates))[:3]
    dyn = goal - curr_pose
    modulation = box.get_modulation_matrix(curr_pose.reshape(-1, 3))
    dx = np.matmul(modulation, dyn).flatten()

    J, Jv, Jw = kinematics.Jacobian(jointStates)

    dtheta = kinematics.inverseVelocity(jointStates, dx)
    exceed_vel = dtheta / kinematics.jointSpeedLimits
    reducer = np.max([np.max(exceed_vel), 1])
    dtheta = dtheta / reducer

    jointStates = np.clip(jointStates + dtheta, kinematics.jointLimitsNegative, kinematics.jointLimitsPositive)

    if np.all(np.isclose(dyn, 0, atol=1e-2)):
        goal[1] *= -1


    for i in range(num_joints):
        p.setJointMotorControl2(bodyIndex=robotId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointStates[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)

    p.stepSimulation()

    ls = p.getLinkState(robotId, endEffectorIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, ls[4], [1, 0, 0], 1, trailDuration)
    prevPose = ls[4]
    hasPrevPose = 1

p.disconnect()