# import gymnasium as gym
# from gymnasium.envs.registration import register
# from custom_cheetah_env import CustomCheetahEnv
# from stable_baselines3 import SAC, TD3, A2C
# import os
# import argparse
# import time

# register(
#     id='CustomCheetah-v0',
#     entry_point='custom_cheetah_env:CustomCheetahEnv',
#     kwargs={'xml_path': 'muj_models/half_cheetah_real_two_limbs.xml'}
# )

# def train(env_id, algorithm):
#     env = gym.make(env_id, render_mode=None)
#     log_dir = "logs/"
#     model_dir = "models/"
#     os.makedirs(log_dir, exist_ok=True)
#     os.makedirs(model_dir, exist_ok=True)
    
#     if algorithm == 'SAC':
#         model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
#     elif algorithm == 'TD3':
#         model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
#     elif algorithm == 'A2C':
#         model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
#     else:
#         print("Algorithm not found")
#         return
    
#     TIMESTEPS = 25000
#     iters = 0
#     while True:
#         iters += 1
#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
#         model.save(f"{model_dir}/{algorithm}_{TIMESTEPS * iters}")

# def test(env_id, algorithm, path_to_model):
#     env = gym.make(env_id, render_mode='human')
    
#     if algorithm == 'SAC':
#         model = SAC.load(path_to_model, env=env)
#     elif algorithm == 'TD3':
#         model = TD3.load(path_to_model, env=env)
#     elif algorithm == 'A2C':
#         model = A2C.load(path_to_model, env=env)
#     else:
#         print("Algorithm not found")
#         return

#     obs, _ = env.reset()
#     done = False
#     extra_steps = 500
#     while True:
#         action, _ = model.predict(obs)
#         obs, _, done, truncated, _ = env.step(action)
#         if done or truncated:
#             extra_steps -= 1
#             if extra_steps < 0:
#                 break
#         time.sleep(0.1)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train or test model.')
#     parser.add_argument('gymenv', help='Gym environment id')
#     parser.add_argument('sb3_algo', help='Stable Baselines3 algorithm')
#     parser.add_argument('-t', '--train', action='store_true')
#     parser.add_argument('-s', '--test', metavar='path_to_model')
#     args = parser.parse_args()

#     if args.train:
#         gymenv = gym.make(args.gymenv, render_mode=None)
#         train(gymenv, args.sb3_algo)
    
#     if args.test:
#         if os.path.isfile(args.test):
#             gym.make(args.gymenv, render_mode='human')
#             test(args.gymenv, args.sb3_algo, args.test)
#         else:
#             print(f'{args.test} not found.')


import gym
from gym.envs.registration import register
from stable_baselines3 import SAC
import os


# Register the custom environment
register(
    id='CustomCheetah-v0',
    entry_point='custom_cheetah_env:CustomCheetahEnv',
    kwargs={'xml_path': 'muj_models/half_cheetah_real_two_limbs.xml'}
)

# Create the environment
env = gym.make('CustomCheetah-v0')

# Define the model
model = SAC('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("custom_cheetah_sac")

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()
env.close()
