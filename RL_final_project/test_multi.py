from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from gymnasium.envs.registration import register
from full_model_script_multi_processs import FullCheetahEnv
import os
import torch.nn as nn

# # Register the custom environment
# def register_custom_env():
#     register(
#         id='FullCheetah-v0',  # Unique environment ID
#         entry_point='full_model_script_multi_processs:FullCheetahEnv',  # Full path to the custom environment class
#         kwargs={'xml_path': 'muj_models/3D_cheetah_flexible_back_8_1_3D_no_cons_1_link_back.xml'},
#         max_episode_steps=1000,
#     )

# Function to create each environment instance
# def make_env():
#     def _init():
#         env = gym.make('FullCheetah-v0')
#         return env
#     return _init

# def make_env():
#     return make("FullCheetah-v0")

def main():
    # Register the custom environment before instantiating it
    # register_custom_env()
    
    # Use the registered environment ID
    env_id = 'FullCheetah-v0'
    
    env = FullCheetahEnv(xml_path='muj_models/3D_cheetah_flexible_back_8_1_3D_no_cons_1_link_back.xml')
    # Number of parallel environments
    num_envs = 30  # Adjust this based on your CPU capacity
    
    # Create the vectorized environment using SubprocVecEnv for parallel training
    env = SubprocVecEnv([lambda: env for _ in range(num_envs)])
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh
    )
    log_dir = "subproc_test_logs/"
    model_dir = "subproc_test_model/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    # Initialize and train the model
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
        verbose=1,
        normalize_advantage=True,
        device='cpu'
    )
    TIMESTEPS = 1000000
    iters = 0
    while True:
        iters += 1
        print(f"Starting iteration {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
        model.save(f"{model_dir}/{PPO}_{iters}_subproc_test")
        print(f"Completed iteration {iters}, model saved")

if __name__ == "__main__":
    main()
