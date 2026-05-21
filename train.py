import os

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from algorithms import SingleExhaustiveSolver, GeneticAlgorithmCombinatorics, GeneticAlgorithmNumerical
from envs import GRNEnv
from grn import GeneRegulatoryNetwork
from utils import parse_args, set_seed, get_file_name, create_system_rollout_module


def get_env(env_name, seed, dt=0.1):
    ids = env_name.split("-")
    grn = GeneRegulatoryNetwork.create(biomodel_idx=int(ids[0]), deltaT=dt)
    return GRNEnv(seed=seed,
                  exp=ids[-1],
                  grn=grn,
                  obs_dim=len(create_system_rollout_module(grn.config).grn_step.y_indexes),
                  r=ids[1],
                  idx=ids[2],
                  dt=dt)


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


def train(seed, task, algorithm, policy, num_workers, num_steps=int(1e4), max_steps=int(1e4)):
    file_name = get_file_name(seed=seed, task=task, algorithm=algorithm, policy=policy)
    env = get_env(env_name=task, seed=seed)
    if algorithm != "ga" and algorithm != "es":
        env = Monitor(env=env,
                      filename=os.path.join("new_output", file_name + ".csv"),
                      info_keywords=("is_broken",))
    model = get_algorithm(algorithm=algorithm,
                          seed=seed,
                          policy=policy,
                          env=env,
                          file_name=".".join([file_name, "csv"]),
                          num_workers=num_workers,
                          verbose=1)
    past_models = [file for file in os.listdir("models") if file_name in file]
    if past_models:
        curr_number_steps = max([int(file.split("_")[1]) for file in past_models if file.endswith("zip")])
        print(f"load {curr_number_steps}")
        model.load([os.path.join("models", file.replace(".zip", "")) for file in os.listdir("models")
                    if file.split("_")[1] == str(curr_number_steps)][0])
        if curr_number_steps >= max_steps:
            return model
    # return model
    model.learn(total_timesteps=num_steps,
                progress_bar=True)
    # model.save(os.path.join("models", file_name))
    if algorithm != "ga" and algorithm != "es":
        os.rename(os.path.join("new_output", file_name + ".csv.monitor.csv"),
                  os.path.join("new_output", file_name + ".csv"))
    return model


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
