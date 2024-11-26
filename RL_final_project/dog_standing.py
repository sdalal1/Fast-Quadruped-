import gym
import os
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import mujoco_py
import numpy as np

# Register the custom environment
register(
    id='QuadrupedStand-v0',
    entry_point='__main__:QuadrupedStandEnv',
)

class QuadrupedStandEnv(gym.Env):
    def __init__(self):
        super(QuadrupedStandEnv, self).__init__()
        
        # Load MuJoCo model
        self.model = mujoco_py.load_model_from_path("./muj_models/3D_cheetah_flexible_back_8_2.xml")
        self.sim = mujoco_py.MjSim(self.model)
        
        # Add a viewer for visualization
        self.viewer = mujoco_py.MjViewer(self.sim)
        
        # Define action and observation space
        n_actuators = self.model.nu
        self.action_space = gym.spaces.Box(low=-20.0, high=10.0, shape=(n_actuators,), dtype=np.float32)
        
        n_observations = self.sim.data.qpos.size + self.sim.data.qvel.size
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_observations,), dtype=np.float32)

    def reset(self):
        # Reset the simulation to initial state
        self.sim.reset()
        self.sim.step()  # Take a step to settle
        
        obs = self._get_obs()
        return obs
    
    def step(self, action):
        # Apply the action
        self.sim.data.ctrl[:] = action
        self.sim.step()
        
        # Render the simulation
        self.viewer.render()
        
        # Calculate reward
        reward = self._compute_reward()
        done = self._is_done()
        
        obs = self._get_obs()
        return obs, reward, done, {}
    
    def _get_obs(self):
        # Combine position and velocity observations
        # return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat], dtype=np.float64)
    
    def _compute_reward(self):
        # Reward for staying upright and close to the initial x-position
        torso_position = self.sim.data.get_body_xpos("torso_1")
        torso_height = torso_position[2]
        torso_x_position = torso_position[0]
        
        # Set desired standing height and initial x-position (standing still target)
        target_height = 0.45  # Adjust based on your model
        initial_x_position = 0.0  # Assuming it starts at x = 0

        # Reward for height
        height_reward = max(0, torso_height - target_height)
        
        # Penalty for moving away from the initial x-position
        stability_penalty = -abs(torso_x_position - initial_x_position)

        # Combined reward: encouraging standing upright without moving forward/backward
        reward = height_reward + stability_penalty
        return reward
    
    def _is_done(self):
        # Terminate if the model falls below a threshold height
        torso_height = self.sim.data.get_body_xpos("torso_1")[2]
        return torso_height < 0.1

# Save directory setup
save_dir = "quadruped_stand_model"
os.makedirs(save_dir, exist_ok=True)

# Callback to save the model every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_dir, name_prefix="ppo_quadruped_stand")

# Instantiate the registered environment
env = gym.make('QuadrupedStand-v0')
env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for SB3

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./quadruped_stand_tensorboard/")

# Train the agent indefinitely
model.learn(total_timesteps=float('inf'), callback=checkpoint_callback)

# Save the final model (optional if you stop manually)
model.save(os.path.join(save_dir, "final_quadruped_stand_model"))
