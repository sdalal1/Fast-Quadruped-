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

class CustomCheetahEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, xml_path, render_mode=None):
        super(CustomCheetahEnv, self).__init__()
        
        # Load the MuJoCo model
        self.model = load_model_from_path(xml_path)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        # self.viewer = None
        
        self.render_mode = render_mode
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        
        # self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)        
        
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
        self.sim.step()
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if done
        terminated = self._is_done()
        # truncated = False  # Update this if you have conditions for truncation
        # truncated = self.sim.data.get_body_xpos('torso')[2] < 0.5
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
        # Define your reward function
        b_shin = self.sim.data.get_body_xpos('bshin')
        b_thigh = self.sim.data.get_body_xpos('bthigh')
        f_shin = self.sim.data.get_body_xpos('fshin')
        f_thigh = self.sim.data.get_body_xpos('fthigh')
        
        joint_penalty = 0.0
        
        if b_shin[2] < 0.1:
            joint_penalty += -1
        
        if b_thigh[2] < 0.05:
            joint_penalty += -1
        
        if f_shin[2] < 0.1:
            joint_penalty += -1
        
        if f_thigh[2] < 0.05:
            joint_penalty += -1
        
        # return self.sim.data.qvel[0] #- self.get_motor_power()
        return self.sim.data.qvel[0] + joint_penalty
    
    # def _compute_reward(self):
    # # Example rewards based on different criteria:
        
    #     # Reward for moving forward (x-direction velocity)
    #     forward_reward = 3 * self.sim.data.qvel[0]
    #     # forward_reward1 = self.sim.data.get_body_xpos('torso')[0]
        
    #     # Reward for staying upright (height of torso)
    #     height = self.sim.data.get_body_xpos('torso')[2]
    #     height1 = self.sim.data.get_body_xpos('midtorso')[2]
    #     upright_reward = 1.0 if height > 0.2 else -2.0
    #     upright_reward1 = 1.0 if height1 > 0.2 else -2.0
        
    #     b_shin = self.sim.data.get_body_xpos('bshin')
    #     b_thigh = self.sim.data.get_body_xpos('bthigh')
    #     f_shin = self.sim.data.get_body_xpos('fshin')
    #     f_thigh = self.sim.data.get_body_xpos('fthigh')
        
    #     joint_penalty = 0.0
        
    #     if b_shin[2] < 0.1:
    #         joint_penalty += -1
        
    #     if b_thigh[2] < 0.05:
    #         joint_penalty += -1
        
    #     if f_shin[2] < 0.1:
    #         joint_penalty += -1
        
    #     if f_thigh[2] < 0.05:
    #         joint_penalty += -1
        
    #     # Penalty for using too much control effort (energy efficiency)
    #     control_effort = np.sum(np.square(self.sim.data.ctrl))
    #     control_penalty = -0.1 * control_effort
        
    #     # Total reward (you can adjust the weights of each component)
    #     reward = forward_reward + upright_reward + control_penalty + joint_penalty + upright_reward1

    #     print(reward)
                
    #     return reward

    
    def _is_done(self):
        # if self.sim.data.get_body_xpos('torso')[2] < 0.2:
        #     return True
        # else:
            return False
    
    def _is_truncated(self):
        b_shin = self.sim.data.get_body_xpos('bshin')
        b_thigh = self.sim.data.get_body_xpos('bthigh')
        f_shin = self.sim.data.get_body_xpos('fshin')
        f_thigh = self.sim.data.get_body_xpos('fthigh')
        # episode length is 1000
        if self.sim.data.time > 1000:
            return True
        if self.sim.data.get_body_xpos('torso')[2] < 0.2:
            # print(self.sim.data.get_body_xpos('torso'))
            return True
       
        #get shin and thigh position
        
        # if b_shin[2] < 0.2 or b_thigh[2] < 0.2 or f_shin[2] < 0.2 or f_thigh[2] < 0.2:
        #     return True
    
        else:
           return False
    def close(self):
        if self.viewer is not None:
            self.viewer = None
    
    def get_total_energy(self):
        # Gravitational constant
        g = 9.81
        
        # Calculate potential energy (PE)
        potential_energy = 0
        for i in range(self.model.nbody):
            mass = self.model.body_mass[i]
            height = self.sim.data.body_xpos[i][2]
            potential_energy += mass * g * height
        
        # Calculate kinetic energy (KE)
        kinetic_energy = 0
        for i in range(self.model.nbody):
            mass = self.model.body_mass[i]
            velocity = self.sim.data.cvel[i]
            kinetic_energy += 0.5 * mass * np.dot(velocity, velocity)
        
        total_energy = potential_energy + kinetic_energy
        return total_energy
    
    def get_motor_power(self):
        # Power is torque (control input) times angular velocity
        motor_power = np.sum(np.abs(self.sim.data.ctrl * self.sim.data.qvel[:len(self.sim.data.ctrl)]))
        return motor_power

def register_custom_env():
    register(
        id='CustomCheetah-v0',
        entry_point='__main__:CustomCheetahEnv',
        kwargs={'xml_path': 'muj_models/half_cheetah_real_two_limbs.xml'}
        # kwargs={'xml_path': 'muj_models/half_cheetah_with_joint.xml'} 
        # kwargs={'xml_path': 'muj_models/half_cheetah_real.xml'}
        
    )

# def train(env_id, algorithm, fname):
#     print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
#     env = gym.make(env_id)
#     log_dir = "logs/"
#     model_dir = "models/"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)
#     # self.viewer = None
#     a = env.unwrapped.a
    
#     if algorithm == 'SAC':
#         model = SAC('MlpPolicy', env, verbose=9, tensorboard_log=log_dir)
#     elif algorithm == 'TD3':
#         model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
#     elif algorithm == 'A2C':
#         model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
#     elif algorithm == 'PPO':
#         model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
#     else:
#         print("Algorithm not found")
#         return
    
    
#     TIMESTEPS = 25000
#     iters = 0
#     while True:
#         iters += 1
#         print(f"Starting iteration {iters}")
#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#         model.save(f"{model_dir}/{a}_{algorithm}_{TIMESTEPS * iters}_{fname}")
#         print(f"Completed iteration {iters}, model saved")

def train(env_id, algorithm, fname):
    print(f"Starting training with environment: {env_id} and algorithm: {algorithm}")
    
    env = gym.make(env_id)
    log_dir = "logs/"
    model_dir = "models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define policy and normalization hyperparameters
    policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
    normalize_kwargs = {'norm_obs': True, 'norm_reward': False}

    if algorithm == 'PPO':
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
            tensorboard_log=log_dir,
            batch_size=64,
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
    else:
        print("Algorithm not found")
        return
    
    TIMESTEPS = 100000
    iters = 0
    while True:
        iters += 1
        print(f"Starting iteration {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}_{fname}")
        print(f"Completed iteration {iters}, model saved")

def test(env_id, algorithm, path_to_model):
    print(f"Starting testing with environment: {env_id}, algorithm: {algorithm}, model path: {path_to_model}")
    
    env = gym.make(env_id, render_mode='human')
    
    if algorithm == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif algorithm == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif algorithm == 'A2C':
        model = A2C.load(path_to_model, env=env)
    elif algorithm == 'PPO':
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
    total_motor_energy = 0

    x = 0
    # while x < 100:
    #     env.step([0, 0, 0])
    #     x += 1 
    
    
    start_time = time.time()
    previous_time = start_time
    while env.unwrapped.sim.data.get_body_xpos('torso')[0] < 10.0:
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

        if done or truncated:
            extra_steps -= 1
            if extra_steps < 0:
                break

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Testing completed in {total_time} seconds")
    print(f"Total energy expended by motors: {total_motor_energy} units")

    # Plotting the energy data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))
    
    # Plot for total energy
    ax1.plot(time_data, energy_data, label='Total Energy')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Total Energy')
    ax1.set_title('Total Energy vs Time during Testing')
    ax1.legend()
    ax1.grid(True)
    
    # Plot for motor energy expended
    ax2.plot(time_data, motor_energy_data, label='Motor Energy Expended', linestyle='--')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Motor Energy Expended')
    ax2.set_title('Motor Energy Expended vs Time during Testing')
    ax2.legend()
    ax2.grid(True)

    # Plot for motor power
    ax3.plot(time_data, motor_power_data, label='Motor Power', color='red')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Motor Power')
    ax3.set_title('Motor Power vs Time during Testing')
    ax3.legend()
    ax3.grid(True)
    
    total_time_text = f"Total Time: {total_time:.2f} seconds"
    total_motor_energy_text = f"Total Motor Energy: {total_motor_energy:.2f} units"
    fig.text(0.5, 0.02, total_time_text, ha='center', fontsize=12)
    fig.text(0.5, 0.01, total_motor_energy_text, ha='center', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    #save with model name
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
