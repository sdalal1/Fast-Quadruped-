import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mujoco_py import MjSim, MjViewer, load_model_from_path

class CustomCheetahEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, xml_path, render_mode=None):
        super(CustomCheetahEnv, self).__init__()
        
        # Load the MuJoCo model
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        self.viewer = None
        
        self.render_mode = render_mode
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sim.data.qpos.size + self.sim.data.qvel.size,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the simulation
        self.sim.reset()
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        # Apply action
        self.sim.data.ctrl[:] = action
        self.sim.step()
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if done
        terminated = self._is_done()
        # truncated = False  # Update this if you have conditions for truncation
        # truncated when the episode length is greater than 1000 and the z of torso is less than a number
        truncated = self.sim.data.get_body_xpos('torso')[2] < 0.5 and self.sim.data.time > 1000
        
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = MjViewer(self.sim)
            self.viewer.render()
    
    def _get_obs(self):
        # Return the current observation
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat]).astype(np.float32)
    
    def _compute_reward(self):
        # Define your reward function
        return self.sim.data.qvel[0]
    
    def _is_done(self):
        return False
    
    def close(self):
        if self.viewer is not None:
            self.viewer = None
