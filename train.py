import os

import gymnasium.spaces
import numpy as np
from matplotlib import pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from algorithms import SingleExhaustiveSolver, GeneticAlgorithmCombinatorics, GeneticAlgorithmNumerical
from envs import GRNEnv
from grn import GeneRegulatoryNetwork
from utils import parse_args, set_seed, get_file_name, create_system_rollout_module, discrete2continuous


def get_env(env_name, seed):
    ids = env_name.split("-")
    grn = GeneRegulatoryNetwork.create(biomodel_idx=int(ids[0]))
    return GRNEnv(seed=seed,
                  exp=ids[-1],
                  grn=grn,
                  obs_dim=len(create_system_rollout_module(grn.config).grn_step.y_indexes),
                  r=ids[1],
                  idx=ids[2])


def get_algorithm(algorithm, **kwargs):
    if algorithm == "ppo":
        return PPO(**kwargs)
    elif algorithm == "rppo":
        return RecurrentPPO(seed=kwargs["seed"],
                            policy=kwargs["policy"],
                            env=kwargs["env"],
                            verbose=kwargs["verbose"])
    elif algorithm == "single":
        return SingleExhaustiveSolver(env=kwargs["env"],
                                      seed=kwargs["seed"])
    elif algorithm == "ga":
        return GeneticAlgorithmCombinatorics(env=kwargs["env"],
                                             seed=kwargs["seed"],
                                             file_name=kwargs["file_name"],
                                             num_workers=kwargs["num_workers"])
    elif algorithm == "es":
        return GeneticAlgorithmNumerical(env=kwargs["env"],
                                         seed=kwargs["seed"],
                                         file_name=kwargs["file_name"],
                                         num_workers=kwargs["num_workers"])
    raise ValueError("Invalid algorithm name: {}".format(algorithm))


def train(seed, task, algorithm, policy, num_workers, num_steps=int(1e5)):
    file_name = get_file_name(seed=seed, task=task, algorithm=algorithm, policy=policy)
    env = get_env(env_name=task, seed=seed)
    if algorithm != "ga" and algorithm != "es":
        env = Monitor(env=env,
                      filename=os.path.join("output", file_name + ".csv"),
                      info_keywords=("is_broken",))
    model = get_algorithm(algorithm=algorithm,
                          seed=seed,
                          policy=policy,
                          env=env,
                          file_name=".".join([file_name, "csv"]),
                          num_workers=num_workers,
                          verbose=1)
    # model.load(os.path.join("models", file_name))
    # return model
    model.learn(total_timesteps=num_steps, progress_bar=True)
    model.save(os.path.join("models", file_name))
    if algorithm != "ga" and algorithm != "es":
        os.rename(os.path.join("output", file_name + ".csv.monitor.csv"), os.path.join("output", file_name + ".csv"))
    return model


def evaluate(model):
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
    return mean_reward, std_reward


def save_rendering(model, file_name, num_steps=1):
    vec_env = model.get_env()
    obs = vec_env.reset()
    data = []
    actions = []
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    env = vec_env.envs[0].env
    fig, axes = plt.subplots(figsize=(8, 10), nrows=2, ncols=1)
    for i in range(num_steps):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        episode_starts = dones
        data.append(info[0]["full_obs"])
        if isinstance(env.action_space, gymnasium.spaces.Discrete):
            action = discrete2continuous(action=action,
                                         env=env,
                                         num_steps=info[0]["full_obs"].shape[1])
        actions.append(action)
    axes[0].plot(np.hstack(data).T, label=env.get_species_names())
    axes[0].set_title("species", fontsize=15)
    axes[1].plot(np.hstack(actions), label=env.get_action_names())
    axes[1].set_title("actions", fontsize=15)
    for ax in axes:
        ax.set_xlabel("sim. time [s]", fontsize=10)
        ax.legend()
    plt.savefig(os.path.join("render", file_name))
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    set_seed(s=args.seed)
    os.makedirs(os.path.join("pop", ".".join([str(args.seed), str(args.task), str(args.algorithm), str(args.policy)])),
                exist_ok=True)
    os.makedirs("envs", exist_ok=True)
    agent = train(seed=args.seed,
                  task=args.task,
                  algorithm=args.algorithm,
                  policy=args.policy,
                  num_workers=args.np)
    os.system("rm {}/*".format(os.path.join("pop", ".".join([str(args.seed), str(args.task), str(args.algorithm),
                                                             str(args.policy)]))))
    os.system("rm {}/*".format(os.path.join("envs")))
    # plot_reward(file_name=get_file_name(seed=args.seed,
    #                                     task=args.task,
    #                                     algorithm=args.algorithm,
    #                                     policy=args.policy))
    # if args.render:
    #     save_rendering(model=agent, file_name=".".join([get_file_name(seed=args.seed,
    #                                                                   task=args.task,
    #                                                                   algorithm=args.algorithm,
    #                                                                   policy=args.policy), "png"]))
