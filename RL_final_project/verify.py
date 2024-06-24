import gymnasium as gym

from stable_baselines3 import PPO
import torch.nn as nn

env = gym.make("HalfCheetah-v4", render_mode="human")

policy_kwargs = dict(
        log_std_init=-2,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=1,
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
            device='cuda',
        )
model.learn(total_timesteps=1000000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()