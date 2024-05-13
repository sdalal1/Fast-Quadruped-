import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
from math import *
import modern_robotics as mr    

def get_transformation_matrix(rotation,pos):
    T = np.eye(4)
    T[:3,:3] = rotation
    T[:3,3] = pos
    return T

def theta_front_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x + (torso_length/2)
    
    if (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) > 1:
        theta2 = 0
    elif (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) < -1:
        theta2 = np.pi
    else:
        theta2 = np.pi - acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    return theta1, theta2

def theta_front_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x + (torso_length/2)
    # print(x, y, lt, lc, torso_length)
    if (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) > 1:
        theta2 = 0
    elif (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) < -1:
        theta2 = np.pi
    else:
        theta2 = np.pi + acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    
    return theta1, theta2

def theta_back_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x - (torso_length/2)
    if (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) > 1:
        theta2 = 0
    elif (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) < -1:
        theta2 = np.pi
    else:
        theta2 = np.pi - acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    theta2-= real_calf_angle
    theta1-= real_thigh_angle
    return theta1, theta2

def theta_back_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # x = x - (torso_length/2)
    if (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) > 1:
        theta2 = 0
    elif (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc) < -1:
        theta2 = np.pi
    else:
        theta2 = np.pi + acos((x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc))
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    theta2-= real_calf_angle
    theta1-= real_thigh_angle
    return theta1, theta2

def pick_case_front(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold, real_calf_angle, real_thigh_angle):
    # print(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold ) 
    # x = x - (torso_length/2)
    t11, t21 = theta_front_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    t12, t22 = theta_front_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if np.sqrt(t11**2 + t21**2) < np.sqrt(t12**2 + t22**2):
        return t11, t21
    else:
        return t12, t22
    
def pick_case_back(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold, real_calf_angle, real_thigh_angle):
    # x = x + (torso_length/2)
    t11, t21 = theta_back_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    t12, t22 = theta_back_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if np.sqrt(t11**2 + t21**2) < np.sqrt(t12**2 + t22**2):
        return t11, t21
    else:
        return t12, t22

def clip_angle(angle, min_angle, max_angle):
    return min(max(angle, min_angle), max_angle)

def normalize_angle(angle):
    if angle > np.pi:
        angle = angle - 2*np.pi
    if angle < -np.pi:
        angle = angle + 2*np.pi
    return angle

def map_angle_to_control(desired_angle, min_angle, max_angle, stiffness):
    # Calculate the range of joint angles
    angle_range = max_angle - min_angle
    
    # Map the desired joint angle to the control input range [-1, 1]
    control_input = 2 * (desired_angle - min_angle) / (stiffness * angle_range) - 1
    return control_input

def angle_from_points(x1,y1,x2,y2,x3,y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return angle

def bezier_curve(t, control_points):
    n = len(control_points) - 1
    points = np.copy(control_points)
    curve_points = np.zeros((len(t), 2))  # Initialize array to store curve points
    for i, ti in enumerate(t):
        b = np.zeros((n + 1, 2))  # Initialize array to store intermediate points
        b[:, :] = points  # Set first row of b to control points
        for r in range(1, n + 1):
            for j in range(n - r + 1):
                b[j] = (1 - ti) * b[j] + ti * b[j + 1]
        curve_points[i] = b[0]
    return curve_points

def stance(x, delta, y):
    res = []
    l = x[0] - x[-1]
    if l == 0:
        for i in range(len(x)):
            y_c = -1*delta*np.sin(np.pi*i/len(x))+y
            res.append(y_c)
    else:
        for i in range(len(x)):
            y_c = -1*delta*np.sin(np.pi*(x[i]-x[0])/l)+y
            res.append(y_c)
    return res



def convert_angle_to_control_using_PD(desired_angle, current_angle, stiffness, damping, prev_error):
    # Calculate the error
    error = desired_angle - current_angle
    # print("error", error)
    # Calculate the derivative of the error
    error_derivative = (error - prev_error)
    # print("error_derivative", error_derivative)
    # Calculate the control input
    control_input = stiffness * error + damping * error_derivative
    
    # control input can only be between -1 and  1
    # if control_input > 1:
    #     control_input = 1
    # if control_input < -1:
    #     control_input = -1
        
    return control_input, error


# def convert_angle_to_control_signal
# Load the model
model = mujoco_py.load_model_from_path("./model/half_cheetah_real_two_limbs.xml")

# Create a simulation environment
sim = mujoco_py.MjSim(model)

# Create a viewer instance
viewer = mujoco_py.MjViewer(sim)

# Get the time step from the simulation
dt = sim.model.opt.timestep

#Initialize x and y coordinates for each link

prev_error1 = 0
prev_error2 = 0
prev_error3 = 0
prev_error4 = 0  


# def H_c(x,y,c_x,c_y, a ,b):
#     return ((x-c_x)**2/a**2 + (y-c_y)**2/b**2)

# def del_H_c(x,y,c_x,c_y, a ,b):
#     return (2*(x-c_x)/a**2, 2*(y-c_y)/b**2)

# def x_i(omega, gamma, x, y, c_x, c_y, a, b, H_i):
#     return -2*omega(y-c_y) / b**2 + 2*gamma*(x-c_x) / a**2 - 2*H_i*(x-c_x)/a**2

# def y_i(omega, gamma, x, y, c_x, c_y, a, b, H_i):
#     return 2*omega(x-c_x) / a**2 + 2*gamma*(y-c_y) / b**2 - 2*H_i*(y-c_y)/b**2

# k = [[0,-1],[-1,0]]

# lamda = 0.1

# x = [-0.03,-0.05, -0.05, -0.05, 0.0, 0.0, 0.0, 0.15, 0.15, 0.13, 0.05]
# y = [0.0, -0.059, -0.059, -0.059, -0.059, -0.059, -0.09, -0.09, -0.09, -0.09, 0.0]

# desired_x = -0.03
# desired_y = 0
# 0.2032, -0.1786

ffoot_pos = sim.data.get_body_xpos('ffoot')
con_pos = np.array([ffoot_pos[0], ffoot_pos[2]])

control_points = np.array([ [ 0.   ,   0.    ], 
                            [-0.03  ,  0.    ],
                            [-0.05  ,  -0.059 ],
                            [-0.05  , -0.059 ],
                            [-0.05  ,  -0.059 ],
                            [ 0.0 ,  -0.059 ],
                            [ 0.0 ,  -0.059 ],
                            [ 0.0 ,  -0.09],
                            [ 0.15, -0.09],
                            [ 0.15, -0.09],
                            [ 0.13,  0.    ],
                            [ 0.05  ,  0.    ]])

# subtract the front contact position so the control point[0] becomes the front contact position
# control_points += con_pos

# print(control_points)

joint_idx = [model.joint_name2id("fthigh"), model.joint_name2id("fshin"), model.joint_name2id("bthigh"), model.joint_name2id("bshin")]

t = np.linspace(0, 1, 100)
a = 1

trajectory = bezier_curve(t, control_points)
sin_x = np.linspace(control_points[-1][0], control_points[0][0], 100)
sin_y = stance(sin_x, 0.01, -control_points[0][1])

mv_x = np.concatenate((trajectory[:, 0], sin_x))
mv_y = np.concatenate((trajectory[:, 1], sin_y))

#merge mv_x and mv_y for each point in the trajectory

x =[]
y = [] 

foot1_traj = []
foot2_traj = []

kp = 0.6
kd = 0.5
phase = 'left'

for i in range(0, len(mv_x)):
    # x.append(-trajectory[i][0])
    # y.append(-trajectory[i][1])
    x.append(mv_x[i])
    y.append(-mv_y[i])
    

# Run a simulation for 1000 steps
## wait for the model to touch ground
for _ in range(0, 200):
    sim.step()
    viewer.render()
for _ in range(0, 1000):
    desired_x = x[a]
    desired_y = y[a]
    
    
    torso_pos = sim.data.get_body_xpos('torso')
    bthigh_pos = sim.data.get_body_xpos('bthigh')
    bshin_pos = sim.data.get_body_xpos('bshin')
    fthigh_pos = sim.data.get_body_xpos('fthigh')
    fshin_pos = sim.data.get_body_xpos('fshin')
    back_contact_pos = sim.data.get_site_xpos('back_contact')
    front_contact_pos = sim.data.get_site_xpos('front_contact')
    ffoot_pos = sim.data.get_body_xpos('ffoot')
    bfoot_pos = sim.data.get_body_xpos('bfoot')

    torso_angle = sim.data.get_body_xquat('torso')
    bthigh_angle = sim.data.get_body_xquat('bthigh')
    bshin_angle = sim.data.get_body_xquat('bshin')
    fthigh_angle = sim.data.get_body_xquat('fthigh')
    fshin_angle = sim.data.get_body_xquat('fshin')

    bthigh_size = sim.model.geom_size[sim.model.geom_name2id('bthigh')]
    bshin_size = sim.model.geom_size[sim.model.geom_name2id('bshin')]
    fthigh_size = sim.model.geom_size[sim.model.geom_name2id('fthigh')]
    fshin_size = sim.model.geom_size[sim.model.geom_name2id('fshin')]

    floor_pos = sim.data.get_geom_xpos('floor')

    prev_x = sim.data.qpos[0]

    ## get foot cnotact posiution wrt to the torso
    front_contact_pos = sim.data.get_site_xpos('front_contact')
    back_contact_pos = sim.data.get_site_xpos('back_contact')


    # # Get the rotation matrix of the body
    rotation_matrix_ffoot = sim.data.body_xmat[model.body_name2id("ffoot")].reshape(3, 3)
    rotation_matrix_bfoot = sim.data.body_xmat[model.body_name2id("bfoot")].reshape(3, 3)
    rotation_matrix_torso = sim.data.body_xmat[model.body_name2id("torso")].reshape(3, 3)
    rotation_matrix_floor = sim.data.geom_xmat[model.geom_name2id("floor")].reshape(3, 3)  

    transformation_floor = get_transformation_matrix(rotation_matrix_floor, floor_pos)
    transformation_torso = get_transformation_matrix(rotation_matrix_torso, torso_pos)
    transformation_bfoot = get_transformation_matrix(rotation_matrix_bfoot, back_contact_pos)
    transformation_ffoot = get_transformation_matrix(rotation_matrix_ffoot, front_contact_pos)


    # get rotation matrix for the front foot wrt to the torso
    transform_torso_ffoot = np.linalg.inv(transformation_torso) @ transformation_ffoot
    transform_torso_bfoot = np.linalg.inv(transformation_torso) @ transformation_bfoot

    # get the position of the front foot wrt to the torso
    position_ffoot = transform_torso_ffoot[:3,3]
    position_bfoot = transform_torso_bfoot[:3,3]

    # get the rotation matrix of the front foot wrt to the torso
    rotation_matrix_ffoot_torso = transform_torso_ffoot[:3,:3]
    rotation_matrix_bfoot_torso = transform_torso_bfoot[:3,:3]

    
    thigh_real_front_angle = angle_from_points(torso_pos[0], torso_pos[2], fthigh_pos[0], fthigh_pos[2], fshin_pos[0], fshin_pos[2])
    thigh_real_back_angle = angle_from_points(torso_pos[0], torso_pos[2], bthigh_pos[0], bthigh_pos[2], bshin_pos[0], bshin_pos[2])
    
    calf_real_front_angle = angle_from_points(fthigh_pos[0], fthigh_pos[2], fshin_pos[0], fshin_pos[2], ffoot_pos[0], ffoot_pos[2])
    calf_real_back_angle = angle_from_points(bthigh_pos[0], bthigh_pos[2], bshin_pos[0], bshin_pos[2], bfoot_pos[0], bfoot_pos[2])
       
    
    # Get current position of the torso
    torso_pos = sim.data.get_body_xpos('torso')
    f_thigh_angle = (sim.data.get_joint_qpos('fthigh'))
    f_shin_angle = (sim.data.get_joint_qpos('fshin'))
    b_thigh_angle = (sim.data.get_joint_qpos('bthigh'))
    b_shin_angle = (sim.data.get_joint_qpos('bshin'))

    # print(desired_x, desired_y, fthigh_size[1], fshin_size[1], torso_pos[0], f_thigh_angle, f_shin_angle)
    theta_thigh, theta_calf = pick_case_front(desired_x, desired_y, fthigh_size[1], fshin_size[1],1 , f_thigh_angle, f_shin_angle, 1, calf_real_front_angle, thigh_real_front_angle)
    b_theta_thigh, b_theta_calf = pick_case_back(desired_x, desired_y, bthigh_size[1], bshin_size[1], 1, b_thigh_angle, b_shin_angle, 1, calf_real_back_angle, thigh_real_back_angle)
    
    # theta_thigh, theta_calf = theta_front_foot(-desired_x, -desired_y, fthigh_size[1], fshin_size[1], 1 , calf_real_front_angle, thigh_real_front_angle)
    # b_theta_thigh, b_theta_calf = theta_back_foot_case2(desired_x, -desired_y, bthigh_size[1], bshin_size[1], 1, calf_real_back_angle, thigh_real_back_angle)
    
    # print("angles before ", theta_thigh, theta_calf, b_theta_thigh, b_theta_calf)
    theta_thigh = clip_angle(theta_thigh, -1, 0.7)
    theta_calf = clip_angle(theta_calf, -1.2, 0.87)
    b_theta_thigh = clip_angle(b_theta_thigh, -0.52, 1.05)
    b_theta_calf = clip_angle(b_theta_calf, -0.785, 0.785)
    
    # theta_thigh = 3.0
    # theta_calf = 0  
    # b_theta_thigh = 0
    # b_theta_calf = 0
    
    # sim.data.ctrl[2] = map_angle_to_control(theta_thigh, -1, 0.7, 180)
    # sim.data.ctrl[3] = map_angle_to_control(theta_calf, -1.2, 0.87, 120)

    # print("theta thigh", theta_thigh)
    # print("theta calf", theta_calf)
    # sim.data.qpos[joint_idx[0]] = clip_angle(theta_thigh, -1, 0.7)
    # sim.data.qpos[joint_idx[1]] =   clip_angle(theta_calf, -1.2, 0.87) 
    # sim.data.qpos[joint_idx[2]] = clip_angle(b_theta_thigh, -0.52, 1.05)
    # sim.data.qpos[joint_idx[3]] = clip_angle(b_theta_calf, -0.785, 0.785)
    # print(theta_thigh, theta_calf, b_theta_thigh, b_theta_calf)
    
    ctrl1 = convert_angle_to_control_using_PD(theta_thigh, f_thigh_angle, kp,kd, prev_error1)
    ctrl2 = convert_angle_to_control_using_PD(theta_calf, f_shin_angle, kp,kd, prev_error2)
    ctrl3 = convert_angle_to_control_using_PD(b_theta_thigh, b_thigh_angle, 0.6,0.5, prev_error3)
    ctrl4 = convert_angle_to_control_using_PD(b_theta_calf, b_shin_angle, 0.6,0.5, prev_error4)
    
        # foot1_traj.append(ffoot_pos)
    # foot2_traj.append(bfoot_pos)
    foot1_traj.append(transform_torso_ffoot[:3,3])
    foot2_traj.append(transform_torso_bfoot[:3,3])
    
    # sim.data.ctrl[0] = b_theta_thigh
    # sim.data.ctrl[1] = b_theta_calf
    # sim.data.ctrl[2] = theta_thigh
    # sim.data.ctrl[3] = theta_calf
    
    if a == len(x) - 1:
        a = 0
        if phase == 'right':
            phase = 'left'
        elif phase == 'left':
            phase = 'right'
            # foot1_traj = np.array(foot1_traj)
            # foot2_traj = np.array(foot2_traj)
        #     print(foot1_traj)
        #     plt.plot(x, y, label="Desired Trajectory")
        #     plt.scatter(foot1_traj[:, 0][0], foot1_traj[:, 2][0], label="Front Foot")
        #     plt.scatter(foot2_traj[:, 0][0], foot2_traj[:, 2][0], label="Back Foot")
        #     # plt.plot(foot1_traj[:, 0], foot1_traj[:, 2], label="Front Foot")
        #     # plt.plot(foot2_traj[:, 0], foot2_traj[:, 2], label="Back Foot")
        #     plt
        #     plt.legend()
        #     plt.show()
        #     exit()
    else:  
        a += 1
    
    ## only send front leg till one cycle end and then send back leg
    if phase == 'right':
        sim.data.ctrl[0] = ctrl1[0]
        sim.data.ctrl[1] = ctrl2[0]
        sim.data.ctrl[2] = 0
        sim.data.ctrl[3] = 0
    elif phase == 'left':
        sim.data.ctrl[0] = 0
        sim.data.ctrl[1] = 0
        sim.data.ctrl[2] = ctrl3[0]
        sim.data.ctrl[3] = ctrl4[0]
        
    prev_error1 = ctrl1[1]
    prev_error2 = ctrl2[1]
    prev_error3 = ctrl3[1]
    prev_error4 = ctrl4[1]
    
    # print("error", ctrl1[1], ctrl2[1], ctrl3[1], ctrl4[1])
    
    # print(ffoot_pos, bfoot_pos)
    sim.step()
    
    # Render the viewer
    viewer.render()

# Close the viewer (the simulation will automatically close when the script ends)

# Plot the trajectory of the foot
foot1_traj = np.array(foot1_traj)
foot2_traj = np.array(foot2_traj)

# print(foot1_traj)
plt.scatter(x[20], y[20], label="Start")
# plt.scatter(x[-1], y[-1], label="End")
plt.plot(x, y, label="Desired Trajectory")
# fffot pos looks like and array of [x,y,z] so we can plot x and z
plt.scatter(foot1_traj[:, 0], foot1_traj[:, 2], label="Front Foot")
plt.scatter(foot2_traj[:, 0], foot2_traj[:, 2], label="Back Foot")
# plt.plot(foot1_traj[:, 0], foot1_traj[:, 2], label="Front Foot")
# plt.plot(foot2_traj[:, 0], foot2_traj[:, 2], label="Back Foot")

plt.legend()
plt.show()

viewer.close()