from typing import Optional, Union, Iterable, Tuple

from flax.core import frozen_dict
import gym
import gym.spaces
import numpy as np


from walk_in_the_park.rl.data.dataset import Dataset, DatasetDict
from walk_in_the_park.rl.data.image_buffer import DiskImageReplayBuffer


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict,
                        insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 next_observation_space: Optional[gym.Space] = None,
                 image_shape: Optional[Tuple[int]] = None,
                 image_disk_save_path: Optional[str] = None):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space,
                                                  capacity)

        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
        )

        # if image_shape is not None:
        #     # unlike the other things stored, this will depend on disk storage space
        #     # since 1_000_000 images in RAM isn't great
        #     self.image_replay_buffer = DiskImageReplayBuffer(capacity=capacity, img_shape=image_shape, save_path=image_disk_save_path)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        # if "images" in data_dict:
        #     assert self._insert_index == self.image_replay_buffer.insert_idx
        #     self.image_replay_buffer.add(data_dict["images"])
        #     del data_dict["images"]

        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    # def sample(self,
    #            batch_size: int,
    #            keys: Optional[Iterable[str]] = None,
    #            indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
    #     if indx is None:
    #         if hasattr(self.np_random, 'integers'):
    #             indx = self.np_random.integers(len(self), size=batch_size)
    #         else:
    #             indx = self.np_random.randint(len(self), size=batch_size)

    #     frozen_dict_batch = super().sample(batch_size, keys, indx)

    #     import time
    #     start = time.time()
    #     images = [self.image_replay_buffer[idx] for idx in indx]
    #     end = time.time()
    #     print(f"{end - start} seconds to do image samples")


    #     return frozen_dict_batch, images
