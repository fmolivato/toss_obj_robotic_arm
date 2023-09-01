import os
from datetime import datetime

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import VecVideoRecorder
from gymnasium.wrappers.record_video import RecordVideo

from thrower import ThrowerEnv
from pusher_v4 import PusherEnv

# Virtual display
from pyvirtualdisplay import Display


SEED = 97120
now = datetime.now()  # current date and time

cwd = f"./videos/{now.strftime('%Y%m%d')}"
os.makedirs(cwd, exist_ok=True)

max_episode_steps = 192
total_timesteps = 36870000
model_batch_size = 64

env_name = "Thrower-v0"
# Register your custom environment with Gym
gym.register(
    id=env_name,
    entry_point="thrower:ThrowerEnv",
    max_episode_steps=max_episode_steps,
)


def train():
    global max_episode_steps
    global total_timesteps
    global model_batch_size
    global cur_steps_trigger

    assert model_batch_size < max_episode_steps
    assert max_episode_steps < total_timesteps

    # Create the vectorized environment
    env = make_vec_env(env_name, n_envs=16)

    # vec_env = gym.make(env_name, render_mode="human")

    env = VecVideoRecorder(
        env,
        "videos",
        lambda i: i % (max_episode_steps * 100) == 0 and i != 0,
        video_length=200,
        name_prefix="vec-train",
        total_timesteps=total_timesteps,
    )

    env.reset()

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=max_episode_steps,
        batch_size=model_batch_size,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        # tensorboard_log="PPO_thrower_tensorboard",
    )

    # Train it for 1,000,000 timesteps
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save(env_name + ".zip")


def eval():
    # We create a separate environment for evaluation
    eval_env = gym.make(env_name, render_mode="human")
    model = PPO.load("Thrower-Curriculum-v0.zip", print_system_info=True)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    # train()
    eval()
