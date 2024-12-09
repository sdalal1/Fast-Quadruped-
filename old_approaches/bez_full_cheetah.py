import numpy as np
import matplotlib.pyplot as plt
import mujoco_py
from math import acos, atan2, sin, cos, pi
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bezier_curve(t, control_points):
    n = len(control_points) - 1
    curve_points = np.zeros((len(t), 2))
    for i, ti in enumerate(t):
        point = np.zeros(2)
        for j in range(n + 1):
            point += control_points[j] * (
                np.math.comb(n, j) * (1 - ti)**(n - j) * ti**j
            )
        curve_points[i] = point
    return curve_points

def inverse_kinematics(x, y, l1, l2):
    r = np.sqrt(x**2 + y**2)
    if r > l1 + l2:
        logging.warning(f"Target position ({x}, {y}) is out of reach. Adjusting to maximum extent.")
        x = x * (l1 + l2) / r
        y = y * (l1 + l2) / r
        r = l1 + l2

    cos_angle2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_angle2 = np.clip(cos_angle2, -1, 1)
    angle2 = acos(cos_angle2)

    angle1 = atan2(y, x) - atan2(l2 * sin(angle2), l1 + l2 * cos(angle2))

    return angle1, angle2

def plot_bezier_curve(control_points, curve_points):
    plt.figure(figsize=(10, 10))
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='Bezier Curve')
    plt.title('Bezier Curve for Cheetah Foot Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Load the model
try:
    model = mujoco_py.load_model_from_path('model/3D_cheetah.xml')
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
except Exception as e:
    logging.error(f"Failed to load the MuJoCo model: {str(e)}")
    exit(1)

# Get joint IDs
try:
    bthigh_id = sim.model.joint_name2id('rbthigh')
    bshin_id = sim.model.joint_name2id('rbshin')
    bfoot_id = sim.model.joint_name2id('rbfoot')
    fthigh_id = sim.model.joint_name2id('rfthigh')
    fshin_id = sim.model.joint_name2id('rfshin')
    ffoot_id = sim.model.joint_name2id('rffoot')
    print(bthigh_id, bshin_id, bfoot_id, fthigh_id, fshin_id, ffoot_id)
except ValueError as e:
    logging.error(f"Failed to find joint in the model: {str(e)}")
    exit(1)

# Define control points for the Bezier curve
control_points = np.array([
    [0.0, -0.5],    # Start point (stance)
    [0.1, -0.45],   # Control point
    [0.2, -0.3],    # Highest point (middle of swing)
    [0.3, -0.45],   # Control point
    [0.4, -0.5]     # End point (back to stance)
])

# Generate Bezier curve
t = np.linspace(0, 1, 100)
trajectory = bezier_curve(t, control_points)

# Plot the Bezier curve
plot_bezier_curve(control_points, trajectory)

# Simulation parameters
step = 0
phase = 0
kp = 200.0
kd = 20.0

# Link lengths (based on the provided model)
l1 = 0.23  # Length of thigh (from model: size="0.046 .133")
l2 = 0.23  # Length of shin (from model: size="0.046 .106")

# Main simulation loop
try:
    while True:
        # Get current position in trajectory
        x, y = trajectory[step % len(trajectory)]
        
        # Adjust for stance phase
        if phase >= 50:  # Stance phase
            y = -0.5  # Fixed y position during stance
        
        # Compute inverse kinematics
        try:
            theta1, theta2 = inverse_kinematics(x, y, l1, l2)
        except ValueError as e:
            logging.error(f"Inverse kinematics failed: {str(e)}")
            continue
        
        # Log joint angles for debugging
        logging.debug(f"Step {step}: theta1 = {theta1}, theta2 = {theta2}")
        
        # Apply control to joints
        for joint_id, target_angle in zip([bthigh_id, bshin_id, fthigh_id, fshin_id],
                                          [theta1, theta2, -theta1, -theta2]):
            print(len(sim.data.ctrl))
            print(len(sim.data.qpos))
            print(len(sim.data.qvel))
            current_angle = sim.data.qpos[joint_id]
            current_velocity = sim.data.qvel[joint_id]
            control = kp * (target_angle - current_angle) - kd * current_velocity
            sim.data.ctrl[joint_id -3] = np.clip(control, -1, 1)  # Assuming control range of [-1, 1]
            
            # Log control signals for debugging
            logging.debug(f"Joint {joint_id}: target = {target_angle}, current = {current_angle}, control = {control}")
        
        # Keep feet parallel to the ground
        sim.data.ctrl[bfoot_id-3] = kp * (-theta1 - theta2 - sim.data.qpos[bfoot_id]) - kd * sim.data.qvel[bfoot_id]
        sim.data.ctrl[ffoot_id-3] = kp * (theta1 + theta2 - sim.data.qpos[ffoot_id]) - kd * sim.data.qvel[ffoot_id]
        
        # Step the simulation
        sim.step()
        viewer.render()
        
        # Log body position and orientation periodically
        if step % 100 == 0:
            body_pos = sim.data.get_body_xpos('torso')
            body_ori = sim.data.get_body_xquat('torso')
            logging.info(f"Step {step}: Body position = {body_pos}, orientation = {body_ori}")
        
        # Update counters
        step += 1
        phase = (phase + 1) % 100

except KeyboardInterrupt:
    logging.info("Simulation stopped by user")
except Exception as e:
    logging.error(f"Simulation failed at step {step}: {str(e)}")
finally:
    # Close the viewer
    viewer.close()
    logging.info("Simulation ended")