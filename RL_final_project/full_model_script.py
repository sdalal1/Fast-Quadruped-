import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
import argparse
import time
from mujoco_py import MjSim, MjViewer, load_model_from_path
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

class FullCheetahEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, xml_path, render_mode=None):
        super(FullCheetahEnv, self).__init__()
        
        # Load the MuJoCo model
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        # self.viewer = None
        self.viewer = MjViewer(self.sim)
        
        # self.render_mode = render_mode
        
        # Define action and observation space
        # 12 actuators for 4 legs (3 joints per leg)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-20, high=20, shape=(8,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)
        
        
        
        
        # Observation space includes joint positions and velocities
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
        if self.viewer is not None:
            self.viewer.render()    
        # Apply action
        self.sim.data.ctrl[:] = action
        # print("ACTION: ", action)
        self.sim.step()
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if done
        terminated = self._is_done()
        truncated = self._is_truncated()
             
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
        # Speed Reward
        speed_reward = 2 * self.speed_reward()

        # Upright Reward
        # upright_reward = self.upright_reward()ls

        # # Joint Position Reward
        # position_reward = self.joint_position_reward()

        # # Joint Effort Penalty
        # effort_penalty = self.joint_effort_penalty()

        total_reward = (
            speed_reward
            # + upright_reward
            # + position_reward
            # + effort_penalty
        )
        
        # Penalize if any leg is too low
        if (self.sim.data.get_body_xpos('lfthigh')[2] < 0.1 or
            self.sim.data.get_body_xpos('rfthigh')[2] < 0.1 or
            self.sim.data.get_body_xpos('lbthigh')[2] < 0.1 or
            self.sim.data.get_body_xpos('rbthigh')[2] < 0.1):
            total_reward -= 100.

        return total_reward

    def speed_reward(self):
        forward_velocity = self.sim.data.qvel[0]
        return forward_velocity

    def upright_reward(self):
        upright_position = 0.10
        current_z_pos = self.sim.data.get_body_xpos('torso')[2]
        return -abs(current_z_pos - upright_position)

    def joint_position_reward(self):
        joint_ranges = {
            'lfthigh': [-1.0, 0.7], 'lfshin': [-1.2, 0.87], 'lffoot': [-0.5, 0.5],
            'rfthigh': [-1.0, 0.7], 'rfshin': [-1.2, 0.87], 'rffoot': [-0.5, 0.5],
            'lbthigh': [-0.52, 1.05], 'lbshin': [-0.785, 0.785], 'lbfoot': [-0.4, 0.785],
            'rbthigh': [-0.52, 1.05], 'rbshin': [-0.785, 0.785], 'rbfoot': [-0.4, 0.785]
        }
        
        reward = 0
        for joint_name, (min_val, max_val) in joint_ranges.items():
            joint_angle = self.sim.data.get_joint_qpos(joint_name)
            if min_val <= joint_angle <= max_val:
                reward += 1.0
            else:
                reward -= 1.0
        
        return reward

    def joint_effort_penalty(self):
        max_torque = {
            'lfthigh': 120, 'lfshin': 60, 'lffoot': 30,
            'rfthigh': 120, 'rfshin': 60, 'rffoot': 30,
            'lbthigh': 120, 'lbshin': 90, 'lbfoot': 60,
            'rbthigh': 120, 'rbshin': 90, 'rbfoot': 60
        }
        
        penalty = 0
        for joint_name, max_tor in max_torque.items():
            joint_torque = self.sim.data.actuator_force[self.model.actuator_name2id(joint_name)]
            if abs(joint_torque) > max_tor:
                penalty -= 0.1 * (abs(joint_torque) - max_tor)
        
        return penalty
    
    # def joint_position_reward(self):
    #     joint_ranges = {
    #         'lfthigh': [-1.0, 0.7], 'lfshin': [-1.2, 0.87],
    #         'rfthigh': [-1.0, 0.7], 'rfshin': [-1.2, 0.87],
    #         'lbthigh': [-0.52, 1.05], 'lbshin': [-0.785, 0.785],
    #         'rbthigh': [-0.52, 1.05], 'rbshin': [-0.785, 0.785],
    #     }
        
    #     reward = 0
    #     for joint_name, (min_val, max_val) in joint_ranges.items():
    #         joint_angle = self.sim.data.get_joint_qpos(joint_name)
    #         if min_val <= joint_angle <= max_val:
    #             reward += 1.0
    #         else:
    #             reward -= 1.0
        
    #     return reward

    # def joint_effort_penalty(self):
    #     max_torque = {
    #         'lfthigh': 120, 'lfshin': 60,
    #         'rfthigh': 120, 'rfshin': 60,
    #         'lbthigh': 120, 'lbshin': 90,
    #         'rbthigh': 120, 'rbshin': 90,        }
        
    #     penalty = 0
    #     for joint_name, max_tor in max_torque.items():
    #         joint_torque = self.sim.data.actuator_force[self.model.actuator_name2id(joint_name)]
    #         if abs(joint_torque) > max_tor:
    #             penalty -= 0.1 * (abs(joint_torque) - max_tor)
        
    #     return penalty
    
    def _is_done(self):
        return False
    
    def _is_truncated(self):
        if self.sim.data.time > 1000:
            return True
        if self.sim.data.get_body_xpos('torso')[2] < 0.1:
            return True
        if self.sim.data.get_body_xpos('torso')[0] > 20.0:
            return True
        if self.sim.data.get_body_xpos('torso')[0] < -1.0:
            return True
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer = None
    
    def get_total_energy(self):
        g = 9.81
        
        potential_energy = 0
        for i in range(self.model.nbody):
            mass = self.model.body_mass[i]
            height = self.sim.data.body_xpos[i][2]
            potential_energy += mass * g * height
        
        kinetic_energy = 0
        for i in range(self.model.nbody):
            mass = self.model.body_mass[i]
            velocity = self.sim.data.cvel[i]
            kinetic_energy += 0.5 * mass * np.dot(velocity, velocity)
        
        total_energy = potential_energy + kinetic_energy
        return total_energy
    
    def get_motor_power(self):
        motor_power = np.sum(np.abs(self.sim.data.ctrl * self.sim.data.qvel[:len(self.sim.data.ctrl)]))
        return motor_power

def register_custom_env():
    register(
        id='FullCheetah-v0',
        entry_point='__main__:FullCheetahEnv',
        # kwargs={'xml_path': 'muj_models/3D_cheetah.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_no_foot_workup_imp.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_no_foot_workup_imp_2.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_no_foot_workup_imp_3.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_no_foot_workup_imp_4.xml'}
        kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_8_1_3D_no_cons_1_link_back.xml'}        
        
        
        
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_8_1_3D_cons_3.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_2.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_no_foot_2.xml'}
    )

def train(env_id, algorithm, fname):
    print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
    env = gym.make(env_id)
    log_dir = "real_test_no_link_logs/"
    model_dir = "real_test_no_link_model/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    normalize_kwargs = {'norm_obs': True, 'norm_reward': False}

    if algorithm == 'PPO':
        model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=0,
        normalize_advantage=True,
        device='cpu'
    )
        # model = PPO(
        # policy='MlpPolicy',
        # env=env,
        # verbose=0,
        # tensorboard_log=log_dir,
        # batch_size=256,  # Increased for stability
        # clip_range=0.2,  # Slightly more aggressive clipping
        # ent_coef=0.01,   # Increased for better exploration
        # gae_lambda=0.95, # Slightly increased for better advantage estimation
        # gamma=0.99,      # Slightly increased for longer-term rewards
        # learning_rate=3e-4,  # Start higher and use scheduler
        # max_grad_norm=0.8,
        # n_steps=2048,    # Increased for better sample efficiency
        # n_epochs=10,     # Reduced to prevent overfitting
        # vf_coef=0.5,     # Standard value
        # policy_kwargs=policy_kwargs,
        # normalize_advantage=True,
        # device='cuda'
        # )
        # model = PPO(
        #     policy='MlpPolicy',
        #     env=env,
        #     verbose=0,
        #     tensorboard_log=log_dir,
        #     # batch_size=64,
        #     batch_size=128,
        #     clip_range=0.1,
        #     ent_coef=0.000401762,
        #     gae_lambda=0.92,
        #     gamma=0.98,
        #     learning_rate=2.0633e-05,
        #     max_grad_norm=0.8,
        #     n_steps=512,
        #     n_epochs=20,
        #     vf_coef=0.58096,
        #     policy_kwargs=policy_kwargs,
        #     normalize_advantage=True,
        #     device='cuda'
        # )
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', env, verbose=0)
        print("Using SAC")
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=0)
        print("Using A2C")
    else:
        print("Algorithm not found")
        return
    
    TIMESTEPS = 1000000
    iters = 0
    while True:
        iters += 1
        print(f"Starting iteration {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
        model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}_{fname}")
        print(f"Completed iteration {iters}, model saved")

def test(env_id, algorithm, path_to_model):
    print(f"Starting testing with environment: {env_id}, algorithm: {algorithm}, model path: {path_to_model}")
    
    env = gym.make(env_id, render_mode='human')
    
    if algorithm == 'PPO':
        model = PPO.load(path_to_model, env=env)
    else:
        print("Algorithm not found")
        return

    obs, _ = env.reset()
    done = False
    extra_steps = 500

    energy_data = []
    time_data = []
    motor_energy_data = []
    motor_power_data = []
    velocity_data = []
    total_motor_energy = 0
    
    start_time = time.time()
    previous_time = start_time
    while env.unwrapped.sim.data.get_body_xpos('torso')[0] < 20.0:
        action, _ = model.predict(obs)
        obs, _, done, truncated, _ = env.step(action)
        
        total_energy = env.unwrapped.get_total_energy()
        motor_power = env.unwrapped.get_motor_power()

        current_time = time.time()
        time_elapsed = current_time - previous_time
        previous_time = current_time

        total_motor_energy += motor_power * time_elapsed
        
        energy_data.append(total_energy)
        time_data.append(current_time - start_time)
        motor_energy_data.append(total_motor_energy)
        motor_power_data.append(motor_power)
        velocity_data.append(env.unwrapped.sim.data.qvel[0])

        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Testing completed in {total_time} seconds")
    print(f"Total energy expended by motors: {total_motor_energy} units")
    print(f"MAXIMUM VELOCITY: {max(velocity_data)}")

    fig , ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    
    ax1.plot(time_data, energy_data, label='Total Energy')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Total Energy')
    ax1.set_title('Total Energy vs Time during Testing')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(time_data, motor_energy_data, label='Motor Energy Expended', linestyle='--')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Motor Energy Expended')
    ax2.set_title('Motor Energy Expended vs Time during Testing')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(time_data, motor_power_data, label='Motor Power', color='red')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Motor Power')
    ax3.set_title('Motor Power vs Time during Testing')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(time_data, velocity_data, label='Velocity', color='green')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Velocity')
    ax4.set_title('Velocity vs Time during Testing')
    ax4.legend()
    ax4.grid(True)
    
    total_time_text = f"Total Time: {total_time:.2f} seconds"
    total_motor_energy_text = f"Total Motor Energy: {total_motor_energy:.2f} units"
    fig.text(0.5, 0.02, total_time_text, ha='center', fontsize=12)
    fig.text(0.5, 0.01, total_motor_energy_text, ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{path_to_model}_energy_consumed.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gym environment id')
    parser.add_argument('sb3_algo', help='Stable Baselines3 algorithm')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-f', '--fname', help='Name of the file')
    args = parser.parse_args()

    register_custom_env()

    if args.train:
        train(args.gymenv, args.sb3_algo, args.fname)
    
    if args.test:
        if os.path.isfile(args.test):
            test(args.gymenv, args.sb3_algo, args.test)
        else:
            print(f'{args.test} not found.')