import os

import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from envs import MotionEquation, GRNEnv
from plotting import plot_reward
from utils import parse_args, set_seed, get_file_name


def get_env(env_name):
    if env_name == "motion":
        return MotionEquation()
    raise ValueError("Invalid env name: {}".format(env_name))


def get_algorithm(algorithm, **kwargs):
    if algorithm == "ppo":
        return PPO(**kwargs)
    elif algorithm == "rppo":
        return RecurrentPPO(**kwargs)
    raise ValueError("Invalid algorithm name: {}".format(algorithm))


def train(seed, task, algorithm, policy, num_steps=int(2e5)):
    file_name = get_file_name(seed=seed, task=task, algorithm=algorithm, policy=policy)
    env = Monitor(env=get_env(env_name=task),
                  filename=os.path.join("output", "monitor.csv"))
    model = get_algorithm(algorithm=algorithm, seed=seed, policy=policy, env=env, verbose=1)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(os.path.join("models", file_name))
    os.rename(os.path.join("output", "monitor.csv"), os.path.join("output", file_name + ".csv"))
    return model


def evaluate(model):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    return mean_reward, std_reward


def save_rendering(model, num_steps=1000):
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)
    agent = train(seed=args.seed,
                  task=args.task,
                  algorithm=args.algorithm,
                  policy=args.policy)
    plot_reward(file_name=get_file_name(seed=args.seed,
                                        task=args.task,
                                        algorithm=args.algorithm,
                                        policy=args.policy))
    if args.render:
        save_rendering(model=agent)
