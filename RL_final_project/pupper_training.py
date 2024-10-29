import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import os
import argparse
import time
from mujoco_py import MjSim, MjViewer, load_model_from_path
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

class PupperEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, xml_path, render_mode=None):
        super(PupperEnv, self).__init__()
        
        # Load the MuJoCo model
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        # self.viewer = None if render_mode != 'human' else MjViewer(self.sim)
        self.viewer = MjViewer(self.sim)
        
        self.render_mode = render_mode
        
        # Define action and observation space
        # 8 actuators for 4 legs (2 joints per leg)
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        
        # Observation space includes joint positions and velocities
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.sim.data.qpos.size + self.sim.data.qvel.size,), dtype=np.float32
        )
        
        # self.torso_links = ['torso_1', 'torso_2', 'torso_3']
        self.torso_links = ['torso']
        
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
        # Compute reward based on forward velocity and upright position
        forward_velocity = self.sim.data.qvel[0]
        upright_reward = -abs(self.sim.data.get_body_xpos('torso')[2] - 0.3)
        # joint_upright_reward = self.sim.data.get_body_xpos('fr')[2] - 0.3
        # upright_reward = self.sim.data.get_body_xpos('torso')[2] - 0.3
        # # Penalize excessive joint movements
        # joint_penalty = -0.01 * np.sum(np.square(self.sim.data.qvel[1:]))
        
        # return forward_velocity + upright_reward + joint_penalty
        return forward_velocity + upright_reward

    def _is_done(self):
        return False
    
    # def _is_truncated(self):
    #     # End episode if pupper falls or moves too far
    #     if self.sim.data.time > 1000:
    #         return True
    #     if self.sim.data.get_body_xpos('torso')[2] < 0.1:
    #         return True
    #     if abs(self.sim.data.get_body_xpos('torso')[0]) > 100.0:
    #         return True
    #     return False
    
    def _is_truncated(self):
        if self.sim.data.time > 1000:
            return True
        # if any(self.sim.data.get_body_xpos(link)[2] < 0.4 for link in self.torso_links):
        #     return True
        if any(self.sim.data.get_body_xpos(link)[2] < 0.1 for link in self.torso_links):
            return True
        if any(self.sim.data.get_body_xpos(link)[0] > 100.0 for link in self.torso_links):
            return True
        if any(self.sim.data.get_body_xpos(link)[0] < -1.0 for link in self.torso_links):
            return True
        if(self.sim.data.get_body_xpos('fr')[2] < 0.1 or self.sim.data.get_body_xpos('fl')[2] < 0.1 or self.sim.data.get_body_xpos('br')[2] < 0.1 or self.sim.data.get_body_xpos('bl')[2] < 0.1):
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
        # motor_power = np.sum(np.abs(self.sim.data.ctrl * self.sim.data.qvel[:len(self.sim.data.ctrl)]))
        # return motor_power
        motor_1_power = np.abs(self.sim.data.ctrl[0] * self.sim.data.qvel[0])
        motor_2_power = np.abs(self.sim.data.ctrl[1] * self.sim.data.qvel[1])
        motor_3_power = np.abs(self.sim.data.ctrl[2] * self.sim.data.qvel[2])
        motor_4_power = np.abs(self.sim.data.ctrl[3] * self.sim.data.qvel[3])
        motor_5_power = np.abs(self.sim.data.ctrl[4] * self.sim.data.qvel[4])
        motor_6_power = np.abs(self.sim.data.ctrl[5] * self.sim.data.qvel[5])
        motor_7_power = np.abs(self.sim.data.ctrl[6] * self.sim.data.qvel[6])
        motor_8_power = np.abs(self.sim.data.ctrl[7] * self.sim.data.qvel[7])
        
        return motor_1_power, motor_2_power, motor_3_power, motor_4_power, motor_5_power, motor_6_power, motor_7_power, motor_8_power

def register_custom_env():
    register(
        id='Pupper-v0',
        entry_point='__main__:PupperEnv',
        # kwargs={'xml_path': 'muj_models/pupper.xml'}  # Update this path
        kwargs={'xml_path': 'muj_models/pupper_2.xml'}  # Update this path
        # kwargs={'xml_path': 'muj_models/pupper_flex.xml'}  # Update this path
        
    
    )

def train(env_id, algorithm, fname):
    print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
    env = gym.make(env_id)
    log_dir = "pupper_logs/"
    model_dir = "pupper_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    if algorithm == 'PPO':
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            batch_size=128,
            clip_range=0.1,
            ent_coef=0.0004,
            gae_lambda=0.92,
            gamma=0.98,
            learning_rate=2e-5,
            max_grad_norm=0.8,
            n_steps=512,
            n_epochs=20,
            vf_coef=0.58,
            policy_kwargs=policy_kwargs,
            normalize_advantage=True,
        )
    else:
        print("Algorithm not found")
        return
    
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
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}_{fname}")
        print(f"Completed iteration {iters}, model saved")

def test(env_id, algorithm, path_to_model):
    def save_list_as_csv(data, filename):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write a header row with joint names
            # header = joint_names
            num_joints = len(data[0])
            header = [f"joint_{i}" for i in range(num_joints)]
            
            writer.writerow(header)
            # print(data)
            
            # Write each time step's joint data as a new row, rounded to two decimal places
            for row in data:
                # print(row)
                # rounded_row =   # Rounding each value to 2 decimal places
                writer.writerow([round(val, 4) for val in row])

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
    # motor_power_data = []
    motor_power_1_data = []
    motor_power_2_data = []
    motor_power_3_data = []
    motor_power_4_data = []
    motor_power_5_data = []
    motor_power_6_data = []
    motor_power_7_data = []
    motor_power_8_data = []
    velocity_data = []
    joint_data = []
    total_motor_energy = 0
    
    start_time = time.time()
    previous_time = start_time
    while env.unwrapped.sim.data.get_body_xpos('torso')[0] < 5.0:
        action, _ = model.predict(obs)
        obs, _, done, truncated, _ = env.step(action)
        
        total_energy = env.unwrapped.get_total_energy()
        # motor_power = env.unwrapped.get_motor_power()
        m1, m2, m3, m4, m5, m6, m7, m8 = env.unwrapped.get_motor_power()
        current_time = time.time()
        time_elapsed = current_time - previous_time
        previous_time = current_time

        # total_motor_energy += motor_power * time_elapsed
        
        energy_data.append(total_energy)
        time_data.append(current_time - start_time)
        motor_energy_data.append(total_motor_energy)
        # motor_power_data.append(motor_power)
        motor_power_1_data.append(m1)
        motor_power_2_data.append(m2)
        motor_power_3_data.append(m3)
        motor_power_4_data.append(m4)
        motor_power_5_data.append(m5)
        motor_power_6_data.append(m6)
        motor_power_7_data.append(m7)
        motor_power_8_data.append(m8)
        velocity_data.append(env.unwrapped.sim.data.qvel[0])
        data = (env.unwrapped.sim.data.qpos[3:]).copy()
        
        joint_data.append(data)
        # print(joint_data)
        #joint name
        print("name", env.unwrapped.sim.model.joint_names[3:])
        # print(env.unwrapped.sim.data.qpos[3:])

        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Testing completed in {total_time} seconds")
    # print(f"Total energy expended by motors: {total_motor_energy} units")
    print(f"MAXIMUM VELOCITY: {max(velocity_data)}")

    save_list_as_csv(joint_data, f"{path_to_model}_joint_data.csv")
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