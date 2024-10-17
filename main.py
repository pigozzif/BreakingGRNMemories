import os

import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from envs import MotionEquation, GRNEnv, LotkaVolterraEquation, SchrodingerEquation
from plotting import plot_reward
from utils import parse_args, set_seed, get_file_name


def get_env(env_name, seed):
    if env_name == "motion":
        return MotionEquation()
    elif env_name == "lotkavolterra":
        return LotkaVolterraEquation()
    elif env_name == "schrodinger":
        return SchrodingerEquation()
    elif "-" in env_name and all(map(lambda x: x.isnumeric(), env_name.split("-"))):
        ids = env_name.split("-")
        return GRNEnv(seed=seed,
                      biomodel_idx=int(ids[0]),
                      obs_dim=3,
                      r=ids[1],
                      ucs=ids[2],
                      cs=ids[3])
    raise ValueError("Invalid env name: {}".format(env_name))


def get_algorithm(algorithm, **kwargs):
    if algorithm == "ppo":
        return PPO(**kwargs)
    elif algorithm == "rppo":
        return RecurrentPPO(**kwargs)
    raise ValueError("Invalid algorithm name: {}".format(algorithm))


def train(seed, task, algorithm, policy, num_steps=int(1e2)):
    file_name = get_file_name(seed=seed, task=task, algorithm=algorithm, policy=policy)
    env = Monitor(env=get_env(env_name=task, seed=seed),
                  filename=os.path.join("output", "monitor.csv"))
    model = get_algorithm(algorithm=algorithm, seed=seed, policy=policy, env=env, verbose=1)
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(os.path.join("models", file_name))
    os.rename(os.path.join("output", "monitor.csv"), os.path.join("output", file_name + ".csv"))
    return model


def evaluate(model):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    return mean_reward, std_reward


def save_rendering(model, file_name, num_steps=1):
    vec_env = model.get_env()
    obs = vec_env.reset()
    data = np.zeros((num_steps, vec_env.observation_space.shape[0]))
    actions = np.zeros((num_steps, vec_env.action_space.shape[0]))
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    env = vec_env.envs[0].env
    fig, axes = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)
    for i in range(num_steps):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        episode_starts = dones
        data[i] = np.array(obs)
        actions[i] = np.delete(env._map_actions(actions=action), env.r)
    axes[0].plot(data, label=env.get_species_names())
    axes[0].set_title("species", fontsize=15)
    axes[1].plot(actions, label=env.get_action_names())
    axes[1].set_title("actions", fontsize=15)
    for ax in axes:
        ax.set_xlabel("sim. time [s]", fontsize=10)
        ax.legend()
    plt.savefig(os.path.join("render", file_name))
    plt.close()


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
        save_rendering(model=agent, file_name=".".join([get_file_name(seed=args.seed,
                                                                      task=args.task,
                                                                      algorithm=args.algorithm,
                                                                      policy=args.policy), "png"]))
