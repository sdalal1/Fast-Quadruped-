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

import gymnasium as gym
from gymnasium import spaces
from mujoco_py import MjSim, MjViewer, load_model_from_path
import numpy as np

class FullCheetahEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, xml_path, render_mode=None):
        super(FullCheetahEnv, self).__init__()
        
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        # self.viewer = None if render_mode != 'human' else MjViewer(self.sim)
        self.viewer = MjViewer(self.sim)
        
        self.render_mode = render_mode
        
        # 12 actuators for 4 legs (3 joints per leg)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        
        # Observation space includes joint positions and velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sim.data.qpos.size + self.sim.data.qvel.size,), dtype=np.float32
        )
        
        # Define torso links
        self.torso_links = ['torso_1', 'torso_2', 'torso_3', 'torso_4']
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim.reset()
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        if self.viewer is not None:
            self.viewer.render()    
        self.sim.data.ctrl[:] = action
        self.sim.step()
        
        obs = self._get_obs()
        reward = self._compute_reward()
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
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat]).astype(np.float32)
    
    def _compute_reward(self):
        speed_reward = 2 * self.speed_reward()
        upright_reward = self.upright_reward()
        position_reward = self.joint_position_reward()
        effort_penalty = self.joint_effort_penalty()

        total_reward = speed_reward + upright_reward + position_reward + effort_penalty
        
        # Penalize if any leg is too low
        if (self.sim.data.get_body_xpos('lfthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('rfthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('lbthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('rbthigh')[2] < 0.2):
            total_reward -= 100.

        return total_reward

    def speed_reward(self):
        return self.sim.data.qvel[0]  # Forward velocity

    def upright_reward(self):
        upright_position = 0.4
        total_deviation = 0
        for torso_link in self.torso_links:
            current_z_pos = self.sim.data.get_body_xpos(torso_link)[2]
            total_deviation += abs(current_z_pos - upright_position)

            # print(f"Torso link: {torso_link}, Current Z pos: {current_z_pos}, Deviation: {abs(current_z_pos - upright_position)}")
        # Calculate average deviation and convert to a reward
        avg_deviation = total_deviation / len(self.torso_links)
        return -avg_deviation  # Negative because less deviation is better
        # upright_position = 0.6

        #there is a negative reward for torso position less than 0.6 
        #and positive reward for torso position greater than 0.6
        
        # total_deviation = 0
        # for torso_link in self.torso_links:
        #     current_z_pos = self.sim.data.get_body_xpos(torso_link)[2]
        #     if current_z_pos < upright_position:
        #         total_deviation += 1.0
        #     else:
        #         total_deviation -= 1.0
                
        # return -total_deviation

    def joint_position_reward(self):
        # joint_ranges = {
        #     'lfthigh': [-1.0, 0.7], 'lfshin': [-1.2, 0.87], 'lffoot': [-0.5, 0.5],
        #     'rfthigh': [-1.0, 0.7], 'rfshin': [-1.2, 0.87], 'rffoot': [-0.5, 0.5],
        #     'lbthigh': [-0.52, 1.05], 'lbshin': [-0.785, 0.785], 'lbfoot': [-0.4, 0.785],
        #     'rbthigh': [-0.52, 1.05], 'rbshin': [-0.785, 0.785], 'rbfoot': [-0.4, 0.785]
        # }
        
        joint_ranges = {
            'lfthigh': [-1.0, 0.7], 'lfshin': [-1.2, 0.87],
            'rfthigh': [-1.0, 0.7], 'rfshin': [-1.2, 0.87],
            'lbthigh': [-0.52, 1.05], 'lbshin': [-0.785, 0.785],
            'rbthigh': [-0.52, 1.05], 'rbshin': [-0.785, 0.785],
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
        # max_torque = {
        #     'lfthigh': 120, 'lfshin': 60, 'lffoot': 30,
        #     'rfthigh': 120, 'rfshin': 60, 'rffoot': 30,
        #     'lbthigh': 120, 'lbshin': 90, 'lbfoot': 60,
        #     'rbthigh': 120, 'rbshin': 90, 'rbfoot': 60
        # }
        
        max_torque = {
            'lfthigh': 120, 'lfshin': 60,
            'rfthigh': 120, 'rfshin': 60,
            'lbthigh': 120, 'lbshin': 90,
            'rbthigh': 120, 'rbshin': 90
        }
        
        penalty = 0
        for joint_name, max_tor in max_torque.items():
            joint_torque = self.sim.data.actuator_force[self.model.actuator_name2id(joint_name)]
            if abs(joint_torque) > max_tor:
                penalty -= 0.1 * (abs(joint_torque) - max_tor)
        
        return penalty
    
    def _is_done(self):
        return False
    
    def _is_truncated(self):
        if self.sim.data.time > 1000:
            return True
        # if any(self.sim.data.get_body_xpos(link)[2] < 0.4 for link in self.torso_links):
        #     return True
        if any(self.sim.data.get_body_xpos(link)[2] < 0.3 for link in self.torso_links):
            return True
        if any(self.sim.data.get_body_xpos(link)[0] > 20.0 for link in self.torso_links):
            return True
        if any(self.sim.data.get_body_xpos(link)[0] < -1.0 for link in self.torso_links):
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
    
    def get_body_mass(self):
        # body name and mass as a dictionary
        body_mass = {self.model.body_id2name(i): self.model.body_mass[i] for i in range(self.model.nbody)}
        return body_mass

# def register_custom_env():
#     register(
#         id='FullCheetah-v0',
#         entry_point='full_model_script:FullCheetahEnv',
#         kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_2.xml'}
#     )

# def train(env_id, algorithm, fname):
#     print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
#     env = gym.make(env_id)
#     log_dir = "full_flex_dog_logs/"
#     model_dir = "full_flex_dog_models/"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)
    
#     policy_kwargs = dict(
#         log_std_init=-2,
#         ortho_init=False,
#         activation_fn=nn.ReLU,
#         net_arch=[dict(pi=[256, 256], vf=[256, 256])]
#     )
#     normalize_kwargs = {'norm_obs': True, 'norm_reward': False}

#     if algorithm == 'PPO':
#         model = PPO(
#             policy='MlpPolicy',
#             env=env,
#             verbose=0,
#             tensorboard_log=log_dir,
#             # batch_size=64,
#             batch_size=128,
#             clip_range=0.1,
#             ent_coef=0.000401762,
#             gae_lambda=0.92,
#             gamma=0.98,
#             learning_rate=2.0633e-05,
#             max_grad_norm=0.8,
#             n_steps=512,
#             n_epochs=20,
#             vf_coef=0.58096,
#             policy_kwargs=policy_kwargs,
#             normalize_advantage=True,
#         )
#     else:
#         print("Algorithm not found")
#         return
    
#     if torch.cuda.is_available():
#         print("Using CUDA")
#         model.policy.to('cuda')
#     else:
#         print("CUDA not available, using CPU")
    
#     TIMESTEPS = 100000
#     iters = 0
#     while True:
#         iters += 1
#         print(f"Starting iteration {iters}")
#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#         model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}_{fname}")
#         print(f"Completed iteration {iters}, model saved")

# def test(env_id, algorithm, path_to_model):
#     print(f"Starting testing with environment: {env_id}, algorithm: {algorithm}, model path: {path_to_model}")
    
#     env = gym.make(env_id, render_mode='human')
    
#     if algorithm == 'PPO':
#         model = PPO.load(path_to_model, env=env)
#     else:
#         print("Algorithm not found")
#         return

#     obs, _ = env.reset()
#     done = False
#     extra_steps = 500

#     energy_data = []
#     time_data = []
#     motor_energy_data = []
#     motor_power_data = []
#     velocity_data = []
#     total_motor_energy = 0
    
#     start_time = time.time()
#     previous_time = start_time
#     while env.unwrapped.sim.data.get_body_xpos('torso')[0] < 20.0:
#         action, _ = model.predict(obs)
#         obs, _, done, truncated, _ = env.step(action)
        
#         total_energy = env.unwrapped.get_total_energy()
#         motor_power = env.unwrapped.get_motor_power()

#         current_time = time.time()
#         time_elapsed = current_time - previous_time
#         previous_time = current_time

#         total_motor_energy += motor_power * time_elapsed
        
#         energy_data.append(total_energy)
#         time_data.append(current_time - start_time)
#         motor_energy_data.append(total_motor_energy)
#         motor_power_data.append(motor_power)
#         velocity_data.append(env.unwrapped.sim.data.qvel[0])

#         if done:
#             extra_steps -= 1
#             if extra_steps < 0:
#                 break

#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Testing completed in {total_time} seconds")
#     print(f"Total energy expended by motors: {total_motor_energy} units")
#     print(f"MAXIMUM VELOCITY: {max(velocity_data)}")

#     fig , ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    
#     ax1.plot(time_data, energy_data, label='Total Energy')
#     ax1.set_xlabel('Time (seconds)')
#     ax1.set_ylabel('Total Energy')
#     ax1.set_title('Total Energy vs Time during Testing')
#     ax1.legend()
#     ax1.grid(True)
    
#     ax2.plot(time_data, motor_energy_data, label='Motor Energy Expended', linestyle='--')
#     ax2.set_xlabel('Time (seconds)')
#     ax2.set_ylabel('Motor Energy Expended')
#     ax2.set_title('Motor Energy Expended vs Time during Testing')
#     ax2.legend()
#     ax2.grid(True)

#     ax3.plot(time_data, motor_power_data, label='Motor Power', color='red')
#     ax3.set_xlabel('Time (seconds)')
#     ax3.set_ylabel('Motor Power')
#     ax3.set_title('Motor Power vs Time during Testing')
#     ax3.legend()
#     ax3.grid(True)
    
#     ax4.plot(time_data, velocity_data, label='Velocity', color='green')
#     ax4.set_xlabel('Time (seconds)')
#     ax4.set_ylabel('Velocity')
#     ax4.set_title('Velocity vs Time during Testing')
#     ax4.legend()
#     ax4.grid(True)
    
#     total_time_text = f"Total Time: {total_time:.2f} seconds"
#     total_motor_energy_text = f"Total Motor Energy: {total_motor_energy:.2f} units"
#     fig.text(0.5, 0.02, total_time_text, ha='center', fontsize=12)
#     fig.text(0.5, 0.01, total_motor_energy_text, ha='center', fontsize=12)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#     plt.savefig(f"{path_to_model}_energy_consumed.png")
#     plt.show()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train or test model.')
#     parser.add_argument('gymenv', help='Gym environment id')
#     parser.add_argument('sb3_algo', help='Stable Baselines3 algorithm')
#     parser.add_argument('-t', '--train', action='store_true')
#     parser.add_argument('-s', '--test', metavar='path_to_model')
#     parser.add_argument('-f', '--fname', help='Name of the file')
#     args = parser.parse_args()

#     register_custom_env()

#     if args.train:
#         train(args.gymenv, args.sb3_algo, args.fname)
    
#     if args.test:
#         if os.path.isfile(args.test):
#             test(args.gymenv, args.sb3_algo, args.test)
#         else:
#             print(f'{args.test} not found.')


def register_custom_env():
    register(
        id='FullCheetah-v0',
        entry_point='__main__:FullCheetahEnv',
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_2.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_3.xml'}
        kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_4.xml'}
        
        
    )

def train(env_id, algorithm, fname):
    print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
    env = gym.make(env_id)
    log_dir = "full_flex_dog_logs/"
    model_dir = "full_flex_dog_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        # net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        net_arch=dict(pi=[256, 256], vf=[256, 256])
        
    )

    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=128,
        clip_range=0.1,
        ent_coef=0.000401762,
        gae_lambda=0.92,
        gamma=0.98,
        learning_rate=2.0633e-05,
        max_grad_norm=0.8,
        n_steps=512,
        n_epochs=20,
        vf_coef=0.58096,
        policy_kwargs=policy_kwargs,
        normalize_advantage=True,
    )
    
    if torch.cuda.is_available():
        print("Using CUDA")
        model.policy.to('cuda')
    else:
        print("CUDA not available, using CPU")
    
    TIMESTEPS = 100000
    iters = 0
    while True:
        iters += 1
        print(f"Starting iteration {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
        model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}_{fname}")
        print(f"Completed iteration {iters}, model saved")
        
def test(env_id, algorithm, path_to_model):
    
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
    
    mass = env.unwrapped.get_body_mass()
    while env.unwrapped.sim.data.get_body_xpos('torso_2')[0] < 20:
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
    print(f"Mass of the body parts: {mass}")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
    
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
    parser = argparse.ArgumentParser(description='Train FullCheetah model.')
    parser.add_argument('gymenv', help='Gym environment id')
    parser.add_argument('sb3_algo', help='Stable Baselines3 algorithm')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-f','--fname', help='Name of the file', default='full_cheetah_model')
    args = parser.parse_args()

    register_custom_env()
    # train('FullCheetah-v0', 'PPO', args.fname)
    
    if args.train:
        train(args.gymenv, args.sb3_algo, args.fname)
        
    if args.test:
        if os.path.isfile(args.test):
            test(args.gymenv, args.sb3_algo, args.test)
        else:
            print(f'{args.test} not found)')