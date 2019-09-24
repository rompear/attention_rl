from app.util.transformer.transformer import Transformer
from app.data.data_structure.transition import Transition
from app.config.config_factory import ConfigFactory
import torch
import numpy as np
import random
import pickle
import os
from typing import List, Set, Dict, Tuple, Optional,Union, Any, List, Optional, cast

class Replay:
    def __init__(self, config: ConfigFactory, phase: str) -> None:
        self.capacity = config.hyperparameters.capacity
        self.memory = []
        self.position = 0
        self.transformer = Transformer
        self.phase = phase
        self.config = config

        self.batch = 0
        self.pickle_dumps = 0

    def push(self, transition) -> None:
        self.memory.append(transition)

        if len(self.memory) > self.capacity:
            del self.memory[0]

        self.save_batch()

    def sample(self, batch_size) -> Tuple(Union[torch.Tensor, np.ndarray]):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    def save_batch(self) -> None:
        self.batch += 1

        # not turned on at the moment (remove False)
        if self.batch % self.capacity == 0 and len(self.memory) == self.capacity and False:
            self.make_path()
            self.write_batch()
            self.pickle_dumps += 1

    def write_batch(self) -> None:
        with open(self.config.save_state_path() + 'batch_' + str(self.pickle_dumps) + '.p', 'wb') as fp:
            pickle.dump(self.memory, fp)

    def make_path(self) -> None:
        dir = self.config.save_state_path()
        if not os.path.exists(dir):
            os.mkdir(dir)