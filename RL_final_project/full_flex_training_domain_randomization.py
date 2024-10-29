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

class FullCheetahEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, xml_path, render_mode=None):
        super(FullCheetahEnv, self).__init__()
        
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        # self.viewer = MjViewer(self.sim) if render_mode == 'human' else None
        self.viewer = MjViewer(self.sim)
        self.render_mode = render_mode
        
        # Action space remains the same
        self.action_space = spaces.Box(low=-20, high=20, shape=(8,), dtype=np.float32)
        
        # Extend observation space to include IMU data (linear acceleration and angular velocity)
        self.torso_links = ['torso_1', 'torso_2', 'torso_3', 'torso_4']
        imu_dim = 6  # 3 for linear acceleration + 3 for angular velocity
        obs_dim = self.sim.data.qpos.size + self.sim.data.qvel.size + len(self.torso_links) * 4 + imu_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Ground randomization parameters
        self.friction_range = (0.3, 0.8)  # Range for ground friction
        # self.elasticity_range = (0.2, 0.8)  # Range for ground elasticity
        
    def _randomize_ground(self):
        """Randomize ground properties"""
        # Randomize friction
        friction = np.random.uniform(*self.friction_range)
        # Get ground geom ID and modify its parameters
        floor_id = self.model.geom_name2id('floor')
        self.model.geom_friction[floor_id][0] = friction  # Sliding friction
        self.model.geom_friction[floor_id][1] = friction * 0.1  # Torsional friction
        self.model.geom_friction[floor_id][2] = friction * 0.1  # Rolling friction
        
        # Randomize elasticity (restitution)
        # elasticity = np.random.uniform(*self.elasticity_range)
        # self.model.geom_solref[floor_id][0] = elasticity
        
        print(f"Ground friction: {friction}")
        
    # def _get_imu_data(self):
    #     """Get IMU readings (linear acceleration and angular velocity)"""
    #     # Get linear acceleration of the imu_site
    #     linear_acc = self.sim.data.get_body_xacc('imu_site')
    #     # linear_acc = self.sim.data.get_body_xacc('imu_site')
        
    #     # Get angular velocity of the main torso
    #     # angular_vel = self.sim.data.get_body_xvelr('torso_1')
    #     angular_vel = self.sim.data.get_body_xvelr('imu_site')
        
    #     print(f"Linear acceleration: {linear_acc}")
    #     print(f"Angular velocity: {angular_vel}")
    #     # exit()
        
    #     return np.concatenate([linear_acc, angular_vel]).astype(np.float32)
    
    def get_imu_data(self):
        """Get IMU readings (linear acceleration and angular velocity)"""
        # Ensure sensors are in the correct order in your XML model (accelerometer first, gyro second)

        # Access the linear acceleration and angular velocity directly from sensordata
        # Assuming first three entries are acceleration, and the next three are angular velocity
        linear_acc = self.sim.data.sensordata[:3]  # Linear acceleration from imu_site accelerometer
        angular_vel = self.sim.data.sensordata[3:6]  # Angular velocity from imu_site gyro

        # Print values for debugging
        # print(f"Linear acceleration: {linear_acc}")
        # print(f"Angular velocity: {angular_vel}")

        # Concatenate and return as a single array
        return np.concatenate([linear_acc, angular_vel]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.sim.reset()
        self._randomize_ground()  # Randomize ground properties on reset
        obs = self.get_obs()
        info = {}
        return obs, info
    
    def get_obs(self):
        qpos = self.sim.data.qpos.flat
        qvel = self.sim.data.qvel.flat
        torso_orientations = []
        for torso_link in self.torso_links:
            torso_orientations.extend(self.sim.data.get_body_xquat(torso_link))
        
        # Add IMU data to observations
        imu_data = self.get_imu_data()
        # exit()
        
        return np.concatenate([qpos, qvel, torso_orientations, imu_data]).astype(np.float32)
    
    def _compute_reward(self):
        speed_reward = self.speed_reward()
        upright_reward = self.upright_reward()
        stability_reward = self._get_stability_reward()  # New stability reward based on IMU
        
        total_reward = 2 * speed_reward + upright_reward + stability_reward
        
        # Penalize if any leg is too low
        if (self.sim.data.get_body_xpos('lfthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('rfthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('lbthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('rbthigh')[2] < 0.2):
            total_reward -= 100.
        
        return total_reward
    
    def _get_stability_reward(self):
        """Compute reward based on IMU readings"""
        imu_data = self.get_imu_data()
        linear_acc = imu_data[:3]
        angular_vel = imu_data[3:]
        
        # Penalize excessive accelerations and angular velocities
        acc_penalty = -0.01 * np.sum(np.square(linear_acc))
        ang_vel_penalty = -0.05 * np.sum(np.square(angular_vel))
        
        return acc_penalty + ang_vel_penalty
    
    def step(self, action):
        if self.viewer is not None:
            self.viewer.render()
        self.sim.data.ctrl[:] = action
        self.sim.step()
        
        obs = self.get_obs()
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
    
    # def _get_obs(self):
    #     qpos = self.sim.data.qpos.flat
    #     qvel = self.sim.data.qvel.flat
    #     torso_orientations = []
    #     for torso_link in self.torso_links:
    #         torso_orientations.extend(self.sim.data.get_body_xquat(torso_link))
    #     return np.concatenate([qpos, qvel, torso_orientations]).astype(np.float32)
    
    def _compute_reward(self):
        # speed_reward = 2 * self.speed_reward()
        # upright_reward = self.upright_reward()
        # position_reward = self.joint_position_reward()
        # effort_penalty = self.joint_effort_penalty()

        # total_reward = 2 * (speed_reward + 2 * upright_reward) + position_reward + effort_penalty
        total_reward = 0
        # # Penalize if any leg is too low
        if (self.sim.data.get_body_xpos('lfthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('rfthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('lbthigh')[2] < 0.2 or
            self.sim.data.get_body_xpos('rbthigh')[2] < 0.2):
            total_reward -= 100.

        # return total_reward
        speed_reward = self.speed_reward()
        # print(f"Speed reward: {speed_reward}")
        return 2 * speed_reward + total_reward


    def speed_reward(self):
        return self.sim.data.qvel[0]  # Forward velocity

    def upright_reward(self):
        upright_position = 0.5
        total_deviation = 0
        for torso_link in self.torso_links:
            current_z_pos = self.sim.data.get_body_xpos(torso_link)[2]
            deviation = abs(current_z_pos - upright_position)
            total_deviation += deviation if deviation < 0.2 else 0.2  # Cap the penalty
        return -total_deviation

    def joint_position_reward(self):
        joint_ranges = {
            'lfthigh': [-1.0, 0.7], 'lfshin': [-1.2, 0.87],
            'rfthigh': [-1.0, 0.7], 'rfshin': [-1.2, 0.87],
            'lbthigh': [-0.52, 1.05], 'lbshin': [-0.785, 0.785],
            'rbthigh': [-0.52, 1.05], 'rbshin': [-0.785, 0.785],
            # 'torso_joint_1': [-0.5, 0.5], 'torso_joint_2': [-0.5, 0.5], 'torso_joint_3': [-0.5, 0.5]
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
            'lfthigh': 120, 'lfshin': 60,
            'rfthigh': 120, 'rfshin': 60,
            'lbthigh': 120, 'lbshin': 90,
            'rbthigh': 120, 'rbshin': 90,
            # 'torso_joint_1': 50, 'torso_joint_2': 50, 'torso_joint_3': 50
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
        if any(self.sim.data.get_body_xpos(link)[2] < 0.1 for link in self.torso_links):
            return True
        if any(self.sim.data.get_body_xpos(link)[0] > 20.0 for link in self.torso_links):
            return True
        if any(self.sim.data.get_body_xpos(link)[0] < -1.0 for link in self.torso_links):
            return True
        if self.sim.data.get_body_xpos('rbthigh')[2] < 0.1 or self.sim.data.get_body_xpos('lbthigh')[2] < 0.1 or self.sim.data.get_body_xpos('rfthigh')[2] < 0.1 or self.sim.data.get_body_xpos('lfthigh')[2] < 0.1:
            return True
        return False

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            
    def get_total_energy(self):
        total_energy = 0
        potential_energy = 0
        kinetic_energy = 0
        
        for i in range(self.model.nbody):
            body = self.sim.model.body_id2name(i)
            mass = self.sim.model.body_mass[i]
            pos = self.sim.data.get_body_xpos(body)
            vel = self.sim.data.get_body_xvelp(body)
            potential_energy += mass * pos[2] * 9.81
            kinetic_energy += 0.5 * mass * np.dot(vel, vel)
        
        total_energy = potential_energy + kinetic_energy
        return total_energy
    
    def get_motor_power(self):
        # motor_power = np.sum(np.abs(self.sim.data.ctrl * self.sim.data.qvel[:len(self.sim.data.ctrl)]))
        # return motor_power
        m1 = self.sim.data.ctrl[0]
        m2 = self.sim.data.ctrl[1]
        m3 = self.sim.data.ctrl[2]
        m4 = self.sim.data.ctrl[3]
        m5 = self.sim.data.ctrl[4]
        m6 = self.sim.data.ctrl[5]
        m7 = self.sim.data.ctrl[6]
        m8 = self.sim.data.ctrl[7]
        
        return m1, m2, m3, m4, m5, m6, m7, m8
    
    
def register_custom_env():
    register(
        id='FullCheetah-v0',
        entry_point='__main__:FullCheetahEnv',
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_3.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_4.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_6.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_7.xml'}
        # kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_8.xml'}
        kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_8_1.xml'}
        

    )

def train(env_id, algorithm, fname):
    print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
    env = gym.make(env_id)
    log_dir = "random_ground_logs/"
    model_dir = "random_ground_models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh
    )
    # policy_kwargs = dict(
    #     log_std_init=-2,
    #     ortho_init=False,
    #     activation_fn=nn.ReLU,
    #     # net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    #     net_arch=dict(pi=[256, 256], vf=[256, 256])
        
    # )

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
        normalize_advantage=True
    )
    # model = PPO(
    #     policy='MlpPolicy',
    #     env=env,
    #     verbose=1,
    #     tensorboard_log=log_dir,
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
    # )
    
    if torch.cuda.is_available():
        print("Using CUDA")
        model.policy.to('cuda')
    else:
        print("CUDA not available, using CPU")
    
    TIMESTEPS = 1000000
    iters = 0
    
    while True:
        iters += 1
        print(f"Starting iteration {iters}")
        # print action space
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
        model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}_{fname}")
        print(f"Completed iteration {iters}, model saved")
        

def test(env_id, algorithm, path_to_model):
    
    env = gym.make(env_id, render_mode='human')
    
    if algorithm == 'PPO':
        model = PPO.load(path_to_model, env=env)
    else:
        raise ValueError("Unknown algorithm")
    
    obs,_ = env.reset()
    done = False
    extra_steps = 500
    
    energy_data = []
    time_data = []
    motor_energy_data = []
    # motor_power_data = []
    m1_data = []
    m2_data = []
    m3_data = []
    m4_data = []
    m5_data = []
    m6_data = []
    m7_data = []
    m8_data = []
    velocity_data = []
    total_motor_energy = 0
    
    start_time = time.time()
    previous_time = start_time
    
    while env.unwrapped.sim.data.get_body_xpos('torso_2')[0] < 20:
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
        m1_data.append(abs(m1))
        m2_data.append(abs(m2))
        m3_data.append(abs(m3))
        m4_data.append(abs(m4))
        m5_data.append(abs(m5))
        m6_data.append(abs(m6))
        m7_data.append(abs(m7))
        m8_data.append(abs(m8))
        
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
    print(f"motor1 max power: {max(m1_data)}, motor1_avg_power: {np.mean(m1_data)}")
    print(f"motor2 max power: {max(m2_data)}, motor2_avg_power: {np.mean(m2_data)}")
    print(f"motor3 max power: {max(m3_data)}, motor3_avg_power: {np.mean(m3_data)}")
    print(f"motor4 max power: {max(m4_data)}, motor4_avg_power: {np.mean(m4_data)}")
    print(f"motor5 max power: {max(m5_data)}, motor5_avg_power: {np.mean(m5_data)}")
    print(f"motor6 max power: {max(m6_data)}, motor6_avg_power: {np.mean(m6_data)}")
    print(f"motor7 max power: {max(m7_data)}, motor7_avg_power: {np.mean(m7_data)}")
    print(f"motor8 max power: {max(m8_data)}, motor8_avg_power: {np.mean(m8_data)}")
    
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
    
    # ax3.plot(time_data, motor_power_data, label='Motor Power', color='red')
    # ax3.set_xlabel('Time (seconds)')
    # ax3.set_ylabel('Motor Power')
    # ax3.set_title('Motor Power vs Time during Testing')
    # ax3.legend()
    # ax3.grid(True)
    
    ax3.plot(time_data, m1_data, label='Motor 1 Power', color='red')
    ax3.plot(time_data, m2_data, label='Motor 2 Power', color='blue')
    ax3.plot(time_data, m3_data, label='Motor 3 Power', color='orange')
    ax3.plot(time_data, m4_data, label='Motor 4 Power', color='purple')
    ax3.plot(time_data, m5_data, label='Motor 5 Power', color='black')
    ax3.plot(time_data, m6_data, label='Motor 6 Power', color='brown')
    ax3.plot(time_data, m7_data, label='Motor 7 Power', color='pink')
    ax3.plot(time_data, m8_data, label='Motor 8 Power', color='cyan')
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
    parser.add_argument('-f','--fname', help='Name of the file', default='full_cheetah_model')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    register_custom_env()
    # train('FullCheetah-v0', 'PPO', args.fname)
    
    if args.train:
        train(args.gymenv, args.sb3_algo, args.fname)
        
    if args.test:
        if os.path.isfile(args.test):
            test('FullCheetah-v0', 'PPO', args.test)
        else:
            print("Invalid path to model file")