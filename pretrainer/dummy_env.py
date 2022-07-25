import numpy as np
from gym import Env, spaces


class DummyEnv(Env):

    def __init__(self,z_dim=32, n_commands=2, n_command_history=20):
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(z_dim + (n_commands * n_command_history),),
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]),
                                       high=np.array([1, 1]), dtype=np.float32)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass