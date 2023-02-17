'''
h5py disk backed replay buffer for images
'''
from typing import Tuple
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

class ImageReplayBuffer(Dataset):
    def __init__(self, capacity: int, img_shape: Tuple[int], save_path: str = None, read_only_if_exists: bool = False, should_print: bool = True):
        if os.path.exists(save_path):
            if read_only_if_exists:
                if should_print:
                    print(f"{save_path} already exists! loading this file instead. you will NOT be able to add to it.")
                self.f = h5py.File(save_path, "r")
            else:
                if should_print:
                    print(f"{save_path} already exists! loading this file instead. you will be able to add to it.")
                self.f = h5py.File(save_path, "r+")

            assert capacity == self.f["rgb"].shape[0]
            self.length = self.f["length"]
            self.insert_idx = self.f["insert_idx"]

            if should_print:
                print(f"{save_path} already has {self.length} images!")
            self.created = False
        else:
            if should_print:
                print(f"creating new dataset at {save_path}")
            self.f = h5py.File(save_path, "w")
            self.created = True

            self.dset = self.f.create_dataset("rgb", shape=(capacity, *img_shape), dtype=np.uint8)
            self.l_dset = self.f.create_dataset("length", shape=(1,), dtype=np.float32)
            self.idx_dset = self.f.create_dataset("insert_idx", shape=(1,), dtype=np.float32)

            self.length = 0
            self.insert_idx = 0

        self.save_path = save_path
        self.read_only_if_exists = read_only_if_exists
        self.rgb_shape = None
        self.capacity = capacity
        self.add_count = 0

    def __del__(self):
        self.f.close()

    def __len__(self):
        return self.length # this is the number of images stored

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError

        return self.dset[idx]

    def add(self, rgb_image):
        self.dset[self.insert_idx] = rgb_image
        self.insert_idx =  self.insert_idx + 1 % self.capacity
        self.length = min(self.length + 1, self.capacity)

        self.add_count += 1

        if self.add_count % 100 == 0:
            self.l_dset[0] = self.length
            self.idx_dset[0] = self.insert_idx
            self.f.flush()
            self.add_count = 0

if __name__ == "__main__":
    pass