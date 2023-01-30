import gym
from gym import ObservationWrapper
from gym.wrappers.flatten_observation import FlattenObservation

from rl.wrappers.single_precision import SinglePrecision
from rl.wrappers.universal_seed import UniversalSeed



class DictToBoxWrapper(ObservationWrapper):
    r"""Observation wrapper that selects one of the keys from a
        dict obs space to expose to higher level consumers."""

    def __init__(self, env: gym.Env, obs_key: str):
        super(DictToBoxWrapper, self).__init__(env)
        self.obs_key = obs_key
        self.observation_space = env.observation_space[self.obs_key]

    def observation(self, observation):
        return observation[self.obs_key]


def wrap_gym(env: gym.Env, rescale_actions: bool = True, obs_key: str = None) -> gym.Env:
    env = SinglePrecision(env)
    env = UniversalSeed(env)
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        if obs_key is None:
            env = FlattenObservation(env)
        else:
            env = DictToBoxWrapper(env, obs_key)

    env = gym.wrappers.ClipAction(env)

    return env