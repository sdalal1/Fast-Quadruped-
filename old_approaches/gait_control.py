import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
from math import *

# implement Bezier Curve

def factorial(n):
    return 1 if n == 0 else n * factorial(n - 1)

def binomial(n, i):
    return factorial(n) / (factorial(i) * factorial(n - i))

# def bezier(control_points, t):
#     n = len(control_points) - 1
#     return sum([binomial(n, i) * (1 - t)**(n - i) * t**i * control_points[i] for i in range(n + 1)])

def bezier(control_points, t):
    n = len(control_points) - 1
    # Initialize a result vector of the same length as the elements in control_points
    result = [0] * len(control_points[0])  # Assumes all control points have the same dimension
    for i in range(n + 1):
        binom = binomial(n, i) * (1 - t)**(n - i) * t**i
        # Update each dimension of the result vector
        for d in range(len(result)):
            result[d] += binom * control_points[i][d]
    return result


def getcalf_theta(theta, x, lc):
    return asin(-x/lc * sin(theta)) - theta

def get_angle(x,y,lt, lc):
    alpha = acos(sqrt(x**2 + y**2) / (2*lt))
    gamma = atan2(x, y)
    theta_thigh_back = gamma + alpha
    theta_calf_back = getcalf_theta(theta_thigh_back, x, lc)
    theta_thigh_front = gamma - alpha
    theta_calf_front = getcalf_theta(theta_thigh_front, x, lc)
    
    return theta_thigh_back, theta_calf_back, theta_thigh_front, theta_calf_front

def get_coordinates(theta_thigh, theta_calf, lt, lc):
    x = lt * sin(theta_thigh) + lc * sin(theta_thigh + theta_calf)
    y = lt * cos(theta_thigh) + lc * cos(theta_thigh + theta_calf)
    return x, y

def get_coordinates_from_control_points(control_points, lt, lc):
    gait_calf = []
    gait_thigh = []
    for i in range(len(control_points)):
        x, y = get_coordinates(control_points[i][0], control_points[i][1], lt, lc)
        gait_calf.append([x, y])
        x, y = get_coordinates(control_points[i][2], control_points[i][3], lt, lc)
        gait_thigh.append([x, y])
    
    return gait_calf, gait_thigh
    
    
model = mujoco_py.load_model_from_path("./model/half_cheetah_real_two_limbs.xml")
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)

# get position of the torso
torso_pos = sim.data.get_body_xpos('torso')
bthigh_pos = sim.data.get_body_xpos('bthigh')
bshin_pos = sim.data.get_body_xpos('bshin')
fthigh_pos = sim.data.get_body_xpos('fthigh')
fshin_pos = sim.data.get_body_xpos('fshin')
ffoot_pos = sim.data.get_body_xpos('ffoot')
bfoot_pos = sim.data.get_body_xpos('bfoot')
back_contact_pos = sim.data.get_site_xpos('back_contact')
front_contact_pos = sim.data.get_site_xpos('front_contact')

torso_angle = sim.data.get_body_xquat('torso')
bthigh_angle = sim.data.get_body_xquat('bthigh')
bshin_angle = sim.data.get_body_xquat('bshin')
fthigh_angle = sim.data.get_body_xquat('fthigh')
fshin_angle = sim.data.get_body_xquat('fshin')

bthigh_size = sim.model.geom_size[sim.model.geom_name2id('bthigh')]
bshin_size = sim.model.geom_size[sim.model.geom_name2id('bshin')]
fthigh_size = sim.model.geom_size[sim.model.geom_name2id('fthigh')]
fshin_size = sim.model.geom_size[sim.model.geom_name2id('fshin')]

# print("bthigh size", bthigh_size)
# print("bshin size", bshin_size)
# print("fthigh size", fthigh_size)
# print("fshin size", fshin_size)

floor_pos = sim.data.get_geom_xpos('floor')

# Define 6 control points for the Bezier curve


# print("floor position", (floor_pos+front_contact_pos)[0]/2)
control_points = [
    # ((front_contact_pos)[0]/2, (floor_pos + front_contact_pos)[2]/2) ,
    (ffoot_pos[0], ffoot_pos[2]),   
    (fshin_pos[0], fshin_pos[2]),
    (fthigh_pos[0], fthigh_pos[2]),
    (torso_pos[0], torso_pos[2]),
    (bthigh_pos[0], bthigh_pos[2]),
    (bshin_pos[0], bshin_pos[2]),
    (bfoot_pos[0], bfoot_pos[2]),
    # ((floor_pos + back_contact_pos)[0]/2, (floor_pos + back_contact_pos)[2]/2)
]

print("control points", control_points)
control_points = np.array(control_points)
plt.plot(control_points[:, 0], control_points[:, 1], 'ro')
plt.plot(control_points[:, 0], control_points[:, 1], 'r-')
plt.show()


detla_x = 0.205
detla_y = 0.108

for i in range(1000):
    t = i / 100.0
    # theta_thigh, theta_calf = bezier(control_points, t)
    x, y = bezier(control_points, t)
    theta_thigh_back, theta_calf_back, theta_thigh_front, theta_calf_front = get_angle(x, y, fthigh_size[1], fshin_size[1])
    print("angles", theta_thigh_back, theta_calf_back, theta_thigh_front, theta_calf_front)
    sim.data.ctrl[0] = theta_thigh_back
    sim.data.ctrl[1] = theta_calf_back
    sim.data.ctrl[2] = theta_thigh_front
    sim.data.ctrl[3] = theta_calf_front
    
    print("controlling")
        
    # Apply these angles to the model
    
    

    # print("theta_thigh", theta_thigh)
    # print("theta_calf", theta_calf)
    # Step the simulation
    sim.step()

    # Render the simulation
    viewer.render()

