import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
from math import acos, atan2, sin, cos

def get_transformation_matrix(rotation, pos):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = pos
    return T

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

def stance(xcoords, delta, y_level):
  res = []
  length = xcoords[0] - xcoords[-1]
  if (length == 0):
    for i in range(0, len(xcoords)):
      ycoord = -1.0*delta*np.sin((np.pi/len(xcoords))*i) + y_level
      res.append(ycoord)
  else:
    for i in range(0, len(xcoords)):
      ycoord = -1.0*delta*np.cos((np.pi/length)*(xcoords[i]-stand_x)) + y_level
      res.append(ycoord)
  return res


def convert_angle_to_control_using_PD(desired_angle, current_angle, stiffness, damping, prev_error):
    # Calculate the error
    error = desired_angle - current_angle
    # Calculate the derivative of the error
    error_derivative = (error - prev_error)
    # Calculate the control input
    control_input = stiffness * error + damping * error_derivative
    return control_input, error

def clip_angle(angle, min_angle, max_angle):
    return min(max(angle, min_angle), max_angle)

def theta_front_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    print(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if lt == 0 or lc == 0:
        return 0, 0  # Prevent division by zero
    
    value = (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc)
    # value = np.clip(value, -1, 1)  # Ensure value is within [-1, 1]

    theta2 = np.pi - acos(value)
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    return theta1, theta2

def theta_front_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    # print(lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if lt == 0 or lc == 0:
        return 0, 0  # Prevent division by zero
    
    value = (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc)
    # value = np.clip(value, -1, 1)  # Ensure value is within [-1, 1]

    theta2 = np.pi + acos(value)
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    return theta1, theta2

def theta_back_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    if lt == 0 or lc == 0:
        return 0, 0  # Prevent division by zero
    
    value = (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc)
    # value = np.clip(value, -1, 1)  # Ensure value is within [-1, 1]

    theta2 = np.pi - acos(value)
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    return theta1, theta2

def theta_back_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle):
    if lt == 0 or lc == 0:
        return 0, 0  # Prevent division by zero
    
    value = (x**2 + y**2 - lt**2 - lc**2) / (2 * lt * lc)
    # value = np.clip(value, -1, 1)  # Ensure value is within [-1, 1]

    theta2 = np.pi + acos(value)
    theta1 = atan2(y, x) - atan2(lc * sin(theta2), lt + lc * cos(theta2))
    
    theta2 -= real_calf_angle
    theta1 -= real_thigh_angle
    return theta1, theta2

def pick_case_front(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold, real_calf_angle, real_thigh_angle):
    t11, t21 = theta_front_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    t12, t22 = theta_front_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if np.sqrt(t11**2 + t21**2) < np.sqrt(t12**2 + t22**2):
        return t11, t21
    else:
        return t12, t22

def pick_case_back(x, y, lt, lc, torso_length, current_theta1, current_theta2, threshold, real_calf_angle, real_thigh_angle):
    t11, t21 = theta_back_foot(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    t12, t22 = theta_back_foot_case2(x, y, lt, lc, torso_length, real_calf_angle, real_thigh_angle)
    if np.sqrt(t11**2 + t21**2) < np.sqrt(t12**2 + t22**2):
        return t11, t21
    else:
        return t12, t22

# Load the model
model = mujoco_py.load_model_from_path('/home/sdalal/mujoco_models/mujoco_menagerie/unitree_go1/scene.xml')

# Create a simulation environment
sim = mujoco_py.MjSim(model)

# Create a viewer instance
viewer = mujoco_py.MjViewer(sim)

# Get joint IDs
FL_thigh_joint_id = model.joint_name2id('FL_thigh_joint')
FL_calf_joint_id = model.joint_name2id('FL_calf_joint')
RL_thigh_joint_id = model.joint_name2id('RL_thigh_joint')
RL_calf_joint_id = model.joint_name2id('RL_calf_joint')
FR_thigh_joint_id = model.joint_name2id('FR_thigh_joint')
FR_calf_joint_id = model.joint_name2id('FR_calf_joint')
RR_thigh_joint_id = model.joint_name2id('RR_thigh_joint')
RR_calf_joint_id = model.joint_name2id('RR_calf_joint')

# Initialize x and y coordinates for each link
prev_error1 = 0
prev_error2 = 0
prev_error3 = 0
prev_error4 = 0
prev_error5 = 0
prev_error6 = 0
prev_error7 = 0
prev_error8 = 0

# Define control points for the Bézier curve


l = 0.213

calf_lo = -2.82
calf_hi = -0.89

thigh_lo = -0.69
thigh_hi =  4.50

hip_lo = -0.86
hip_hi =  0.86


lspan = 0.025
dx1 = 0.0125
dx2 = 0.0125
stand_floor = -0.8*(2*l)
swing_height = 0.04
dy = 0.0125
delta = 0.01
stand_x = -0.05

points_x = [stand_x + -1.0*lspan,
            stand_x + -1.0*lspan - dx1,
            stand_x + -1.0*lspan - dx1 - dx2,
            stand_x + -1.0*lspan - dx1 - dx2,
            stand_x + -1.0*lspan - dx1 - dx2,
            stand_x + 0.0,
            stand_x + 0.0,
            stand_x + 0.0,
            stand_x + lspan + dx1 + dx2,
            stand_x + lspan + dx1 + dx2,
            stand_x + lspan + dx1,
            stand_x + lspan]
points_y = [stand_floor,
            stand_floor,
            stand_floor + swing_height,
            stand_floor + swing_height,
            stand_floor + swing_height,
            stand_floor + swing_height,
            stand_floor + swing_height,
            stand_floor + swing_height + dy,
            stand_floor + swing_height + dy,
            stand_floor + swing_height + dy,
            stand_floor,
            stand_floor]

# # points_x = [-0.2, -0.2805, -0.300, -0.300, -0.300,   0.0,   0.0,   0.0, 0.3032, 0.3032, 0.2826, 0.200] 
# points_x = [-0.18, -0.2605, -0.280, -0.280, -0.280,   0.0,   0.0,   0.0, 0.2832, 0.2832, 0.2626, 0.180] 
# # points_y = [-0.5, -0.5, -0.3611, -0.3611, -0.3611, -0.3611, -0.3611, -0.3214, -0.3214, -0.3214, -0.5, -0.5]
# points_y = [-0.3, -0.3, -0.1611, -0.1611, -0.1611, -0.1611, -0.1611, -0.1214, -0.1214, -0.1214, -0.3, -0.3]
control_points = np.array([(points_x[i],points_y[i]) for i in range(len(points_x))])

# control_points = np.array([
#     [0.0, 0.0],
#     [-0.03, 0.0],
#     [-0.05, -0.059],
#     [-0.05, -0.059],
#     [-0.05, -0.059],
#     [0.0, -0.059],
#     [0.0, -0.059],
#     [0.0, -0.09],
#     [0.15, -0.09],
#     [0.15, -0.09],
#     [0.13, 0.0],
#     [0.05, 0.0]
# ])

t = np.linspace(0, 1, 100)
a = 1

trajectory = bezier_curve(t, control_points)
sin_x = np.linspace(control_points[-1][0], control_points[0][0], 100)
sin_y = stance(sin_x, 0.01, control_points[0][1])

mv_x = np.concatenate((trajectory[:, 0], sin_x))
mv_y = np.concatenate((trajectory[:, 1], sin_y))

# Merge mv_x and mv_y for each point in the trajectory
x = mv_x
y = mv_y

foot1_traj = []
foot2_traj = []
foot3_traj = []
foot4_traj = []

kp = 0.6
kd = 0.5
phase = 'left'

# Run a simulation for 1000 steps
# Wait for the model to touch ground
for _ in range(0, 200):
    sim.step()
    viewer.render()

for _ in range(0, 1000):
    desired_x = x[a]
    desired_y = y[a]

    torso_pos = sim.data.get_body_xpos('trunk')
    fthigh_pos = sim.data.get_body_xpos('FL_thigh')
    fshin_pos = sim.data.get_body_xpos('FL_calf')
    bthigh_pos = sim.data.get_body_xpos('RL_thigh')
    bshin_pos = sim.data.get_body_xpos('RL_calf')
    # ffoot_pos = sim.data.get_body_xpos('FL_calf')  # Updated
    # bfoot_pos = sim.data.get_body_xpos('RL_calf')  # Updated
    # ffoot is site
    ffoot_pos = sim.data.get_site_xpos('FL')
    bfoot_pos = sim.data.get_site_xpos('RL')
    
    

    f_thigh_angle = sim.data.get_joint_qpos('FL_thigh_joint')
    f_shin_angle = sim.data.get_joint_qpos('FL_calf_joint')
    b_thigh_angle = sim.data.get_joint_qpos('RL_thigh_joint')
    b_shin_angle = sim.data.get_joint_qpos('RL_calf_joint')

    fthigh_size = sim.model.geom_size[sim.model.geom_name2id('FL')]
    fshin_size = sim.model.geom_size[sim.model.geom_name2id('FL')]
    bthigh_size = sim.model.geom_size[sim.model.geom_name2id('RL')]
    bshin_size = sim.model.geom_size[sim.model.geom_name2id('RL')]

    # Assuming some arbitrary angles for now
    calf_real_front_angle = 0
    thigh_real_front_angle = 0
    calf_real_back_angle = 0
    thigh_real_back_angle = 0

    theta_thigh, theta_calf = pick_case_front(desired_x, desired_y, fthigh_size[1], fshin_size[1], 1, f_thigh_angle, f_shin_angle, 1, calf_real_front_angle, thigh_real_front_angle)
    b_theta_thigh, b_theta_calf = pick_case_back(desired_x, desired_y, bthigh_size[1], bshin_size[1], 1, b_thigh_angle, b_shin_angle, 1, calf_real_back_angle, thigh_real_back_angle)
    
    # theta_thigh = clip_angle(theta_thigh, -1, 0.7)
    # theta_calf = clip_angle(theta_calf, -1.2, 0.87)
    # b_theta_thigh = clip_angle(b_theta_thigh, -0.52, 1.05)
    # b_theta_calf = clip_angle(b_theta_calf, -0.785, 0.785)

    # ctrl1 = convert_angle_to_control_using_PD(theta_thigh, f_thigh_angle, kp, kd, prev_error1)
    # ctrl2 = convert_angle_to_control_using_PD(theta_calf, f_shin_angle, kp, kd, prev_error2)
    # ctrl3 = convert_angle_to_control_using_PD(b_theta_thigh, b_thigh_angle, kp, kd, prev_error3)
    # ctrl4 = convert_angle_to_control_using_PD(b_theta_calf, b_shin_angle, kp, kd, prev_error4)
    ctrl1 = [theta_thigh, 0]
    ctrl2 = [theta_calf, 0]
    
    print("theta thigh", theta_thigh)
    print("theta calf", theta_calf)
    

    foot1_traj.append([fthigh_pos[0], fthigh_pos[2]])
    foot2_traj.append([bthigh_pos[0], bthigh_pos[2]])

    # Process for right legs (FR and RR)
    r_fthigh_pos = sim.data.get_body_xpos('FR_thigh')
    r_fshin_pos = sim.data.get_body_xpos('FR_calf')
    r_bthigh_pos = sim.data.get_body_xpos('RR_thigh')
    r_bshin_pos = sim.data.get_body_xpos('RR_calf')
    r_ffoot_pos = sim.data.get_body_xpos('FR_calf')  # Updated
    r_bfoot_pos = sim.data.get_body_xpos('RR_calf')  # Updated

    r_f_thigh_angle = sim.data.get_joint_qpos('FR_thigh_joint')
    r_f_shin_angle = sim.data.get_joint_qpos('FR_calf_joint')
    r_b_thigh_angle = sim.data.get_joint_qpos('RR_thigh_joint')
    r_b_shin_angle = sim.data.get_joint_qpos('RR_calf_joint')

    r_fthigh_size = sim.model.geom_size[sim.model.geom_name2id('FR')]
    r_fshin_size = sim.model.geom_size[sim.model.geom_name2id('FR')]
    r_bthigh_size = sim.model.geom_size[sim.model.geom_name2id('RR')]
    r_bshin_size = sim.model.geom_size[sim.model.geom_name2id('RR')]

    r_theta_thigh, r_theta_calf = pick_case_front(desired_x, desired_y, 0.213, 0.213, 1, r_f_thigh_angle, r_f_shin_angle, 1, calf_real_front_angle, thigh_real_front_angle)
    r_b_theta_thigh, r_b_theta_calf = pick_case_back(desired_x, desired_y, 0.213, 0.213, 1, r_b_thigh_angle, r_b_shin_angle, 1, calf_real_back_angle, thigh_real_back_angle)

    # r_theta_thigh = clip_angle(r_theta_thigh, -1, 0.7)
    # r_theta_calf = clip_angle(r_theta_calf, -1.2, 0.87)
    # r_b_theta_thigh = clip_angle(r_b_theta_thigh, -0.52, 1.05)
    # r_b_theta_calf = clip_angle(r_b_theta_calf, -0.785, 0.785)

    ctrl5 = convert_angle_to_control_using_PD(r_theta_thigh, r_f_thigh_angle, kp, kd, prev_error5)
    ctrl6 = convert_angle_to_control_using_PD(r_theta_calf, r_f_shin_angle, kp, kd, prev_error6)
    ctrl7 = convert_angle_to_control_using_PD(r_b_theta_thigh, r_b_thigh_angle, kp, kd, prev_error7)
    ctrl8 = convert_angle_to_control_using_PD(r_b_theta_calf, r_b_shin_angle, kp, kd, prev_error8)

    foot3_traj.append([r_fthigh_pos[0], r_fthigh_pos[2]])
    foot4_traj.append([r_bthigh_pos[0], r_bthigh_pos[2]])

    if a == len(x) - 1:
        a = 0
        if phase == 'right':
            phase = 'left'
        elif phase == 'left':
            phase = 'right'
    else:
        a += 1
    
    # print(ctrl1, ctrl2, ctrl3, ctrl4, ctrl5, ctrl6, ctrl7, ctrl8)
    # Set control inputs by joint names
    sim.data.ctrl[FL_thigh_joint_id -1] = ctrl1[0]
    sim.data.ctrl[FL_calf_joint_id -1] = ctrl2[0]
    # sim.data.ctrl[RL_thigh_joint_id-1] = ctrl3[0]
    # sim.data.ctrl[RL_calf_joint_id-1] = ctrl4[0]
    # sim.data.ctrl[FR_thigh_joint_id-1] = ctrl5[0]
    # sim.data.ctrl[FR_calf_joint_id-1] = ctrl6[0]
    # sim.data.ctrl[RR_thigh_joint_id-1] = ctrl7[0]
    # sim.data.ctrl[RR_calf_joint_id-1] = ctrl8[0]

    prev_error1 = ctrl1[1]
    prev_error2 = ctrl2[1]
    # prev_error3 = ctrl3[1]
    # prev_error4 = ctrl4[1]
    # prev_error5 = ctrl5[1]
    # prev_error6 = ctrl6[1]
    # prev_error7 = ctrl7[1]
    # prev_error8 = ctrl8[1]

    sim.step()
    viewer.render()

# Plot the trajectory of the foot
foot1_traj = np.array(foot1_traj)
foot2_traj = np.array(foot2_traj)
foot3_traj = np.array(foot3_traj)
foot4_traj = np.array(foot4_traj)


# plot the foot as stick figure
#location of the foot joints
# foot1 = np.array([fthigh_pos[0], fthigh_pos[2]], [fshin_pos[0], fshin_pos[2]], [ffoot_pos[0], ffoot_pos[2]])
foot1_x = [fthigh_pos[0], fshin_pos[0], ffoot_pos[0]]
foot1_y = [fthigh_pos[2], fshin_pos[2], ffoot_pos[2]]

plt.scatter(x[20], y[20], label="Start")
plt.plot(x, y, label="Desired Trajectory")
# plt.scatter(foot1_traj[:, 0], foot1_traj[:, 1], label="Front Left Foot")
# plt.scatter(foot2_traj[:, 0], foot2_traj[:, 1], label="Back Left Foot")
#plot left and right foor joints
plt.plot(foot1_x, foot1_y, label="Front Left Foot")
plt.scatter(fthigh_pos[0], fthigh_pos[2], label="Front Left Thigh")
plt.scatter(fshin_pos[0], fshin_pos[2], label="Front Left Shin")
# plt.scatter(ffoot_pos[0], ffoot_pos[2], label="Front Left Foot")

# plt.scatter(foot3_traj[:, 0], foot3_traj[:, 1], label="Front Right Foot")
# plt.scatter(foot4_traj[:, 0], foot4_traj[:, 1], label="Back Right Foot")

plt.legend()
plt.show()

viewer.close()
