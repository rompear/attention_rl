from app.util.transformer.transformer import Transformer
from app.config.config_factory import ConfigFactory
import torch
import numpy as np
import random
import pickle
import os
from typing import List, Set, Dict, Tuple, Optional,Union, Any, List, Optional, cast
import sys

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
        self.pretrain_states = {}

        if config.is_pretrain():
            self.fill_memory_with_pretrain_data()

    def push(self, transition, pretrain=False) -> None:
        transition = (transition[0].to(torch.half), transition[1], transition[2].to(torch.half), transition[3], transition[4], torch.from_numpy(np.array([pretrain])))
        self.memory.append(transition)

        if len(self.memory) > self.capacity:
            if len(self.pretrain_states) > 0:
                self.remove_item_from_dict()

            del self.memory[0]

        self.save_batch()

    def sample(self, batch_size) -> Tuple[Union[torch.Tensor, np.ndarray]]:
        transitions = random.sample(self.memory, batch_size)
        return transitions

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> None:
        self.pretrain_states = {}
        for transition in self.memory:
            key = transition[0]
            if key not in self.pretrain_states:
                self.pretrain_states[key] = [transition[1]]
            else:
                self.pretrain_states[key].append(transition[1])

    def remove_item_from_dict(self) -> None:
        # if there are more actions for this state, remove items from array in dict
        if len(self.pretrain_states[self.memory[0][0]]) > 1:
            del self.pretrain_states[self.memory[0][0]][self.memory[0][1]]

        # else: if only one acttion, remove the key
        else:
            del self.pretrain_states[self.memory[0][0]]

    def fill_memory_with_pretrain_data(self) -> None:
        for i in range(0, 5):
            transitions = self.load_data_from_disk()
            for transition in transitions:

                self.push(transition, pretrain=True)
        self.state_dict()

    def save_batch(self) -> None:
        self.batch += 1

        # not turned on at the moment (remove 'and False')
        if self.batch % self.capacity == 0 and len(self.memory) == self.capacity and False:
            self.make_path()
            self.write_batch_to_disk()
            self.pickle_dumps += 1
            print("saved bacth")


    def load_data_from_disk(self) -> List:
        with open(self.config.load_pretrain(), 'rb') as fp:
            return pickle.load(fp)

    def write_batch_to_disk(self) -> None:
        with open(self.config.save_state_path() + 'batch_' + str(self.pickle_dumps) + '.p', 'wb') as fp:
            pickle.dump(self.memory, fp)

    def make_path(self) -> None:
        dir = self.config.save_state_path()
        if not os.path.exists(dir):
            os.mkdir(dir)