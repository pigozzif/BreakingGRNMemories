import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback


class ExhaustiveSolver(BaseAlgorithm):

    def __init__(self, env, seed):
        super().__init__(None, env, 0.0, seed=seed)
        self.best = None
        self.best_reward = float("-inf")

    def learn(self: SelfBaseAlgorithm, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> SelfBaseAlgorithm:
        env = self.env.envs[0]
        env.reset()
        for action in range(env.action_space.n):
            obs, reward, terminated, truncated, info = env.step(action)
            if reward > self.best_reward:
                self.best = action
                self.best_reward = reward
            if terminated or truncated:
                env.reset()
        return self

    def predict(self, observation, state=None,  episode_start=None, deterministic=False):
        return np.array(self.best), None

    def _setup_model(self):
        return

    def save(self, path, exclude=None, include=None):
        return

    def load(self, path, env=None, device="auto", custom_objects=None, print_system_info=False, force_reset=True,
             **kwargs):
        return self
