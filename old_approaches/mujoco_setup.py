# import mujoco_py
# import numpy as np

# # Load the model
# model = mujoco_py.load_model_from_path("./model/half_cheetah_real.xml")

# # Create a simulation environment
# sim = mujoco_py.MjSim(model)

# # Create a viewer instance
# viewer = mujoco_py.MjViewer(sim)

# # Run a simulation for 1000 steps
# for _ in range(10000):
#     # Generate random actions for the actuators
#     # action = np.random.uniform(low=-5, high=5, size=sim.model.nu)
#     # Apply the actions to the actuators
#     # sim.data.ctrl[:] = action
    
#     # Step the simulation
#     sim.step()
    
#     # Render the viewer
#     viewer.render()

# # Close the viewer (the simulation will automatically close when the script ends)
# viewer.close()


# import mujoco_py
# import numpy as np

# # Load the model
# model = mujoco_py.load_model_from_path("./model/half_cheetah_real_two_limbs.xml")

# # model = mujoco_py.load_model_from_path("./model/half_cheetah.xml")

# # Create a simulation environment
# sim = mujoco_py.MjSim(model)

# # Create a viewer instance
# viewer = mujoco_py.MjViewer(sim)

# # Get the time step from the simulation
# dt = sim.model.opt.timestep

# # Initialize the previous x-coordinate
# prev_x = sim.data.qpos[0]

# # Run a simulation for 1000 steps
# for _ in range(10000):
#     # Generate random actions for the actuators with bias towards forward movement
#     forward_bias = 15.0  # Increase this value for more forward bias
#     action = np.random.uniform(low=-5, high=5, size=sim.model.nu)
#     # action =  np.array([1,1,1,1])
#     action += forward_bias * np.clip(prev_x - sim.data.qpos[0], 0, np.inf)
    
#     # Apply the actions to the actuators
#     sim.data.ctrl[:] = action
    
#     # Step the simulation
#     sim.step()
    
#     # Get the current x-coordinate of the cheetah
#     current_x = sim.data.qpos[0]
    
#     print("current state:", sim.data.qpos)    
#     # # Calculate the forward movement reward
#     # forward_reward_weight = 10.0  # Adjust as needed
#     # forward_reward = forward_reward_weight * max(0, current_x - prev_x) / dt
    
#     # # Calculate the control cost
#     # ctrl_cost_weight = -1.0  # Adjust as needed
#     # ctrl_cost = ctrl_cost_weight * np.sum(action ** 2)
    
#     # # Combine the rewards and costs
#     # total_reward = forward_reward - ctrl_cost
    
#     # # Update the previous x-coordinate for the next iteration
#     # prev_x = current_x
    
#     # # Print rewards and costs for debugging
#     # print("Forward Reward:", forward_reward)
#     # print("Control Cost:", ctrl_cost)
#     # print("Total Reward:", total_reward)
    
#     # Render the viewer
#     viewer.render()

# # Close the viewer (the simulation will automatically close when the script ends)
# viewer.close()

import mujoco_py
import numpy as np

# Load the model
model = mujoco_py.load_model_from_path("./model/half_cheetah_real_two_limbs.xml")
# model = mujoco_py.load_model_from_path("./model/half_cheetah.xml")


# Create a simulation environment
sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)
# Get number of actuators
num_actuators = sim.model.nu

# Proportional control gains
kp = 1.0

# Run a simulation for 1000 steps
for _ in range(1000):
    
    # Get current position of the torso
    torso_pos = sim.data.get_body_xpos('torso')
    
    # Calculate desired velocity in x-direction (e.g., 1 m/s)
    desired_vel_x = 1.0
    
    # Calculate error in x-direction
    error_x = desired_vel_x - torso_pos[0]
    
    # Calculate control signal (torque) for each actuator
    # ctrl_signal = kp * error_x * np.ones(num_actuators)
    ctrl_signal = kp * error_x * np.random.uniform(low=-1, high=1, size=num_actuators)
    # Apply the control signal to the actuators
    sim.data.ctrl[:] = ctrl_signal
    # sim.data.qvel[0] = 0.2
    
    # # Step the simulation
    print("current state:", sim.data.qpos)
    print("current velocity:", sim.data.qvel)
    print("number of actuator", num_actuators)
    #read from sensors
    print("sensor data", sim.data.sensordata)
    print("link size", sim.model.nbody)

    sim.step()
    
    viewer.render()

viewer.close()  


# bezier curve
