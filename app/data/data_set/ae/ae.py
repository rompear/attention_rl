from app.util.transformer.transformer import Transformer
from app.config.config_factory import ConfigFactory
import pickle
from torch.utils.data import Dataset
import torch
from typing import List, Set, Dict, Tuple, Optional,Union, Any, List, Optional, cast


class AE(Dataset):
    def __init__(self, config: ConfigFactory, phase: str) -> None:
        self.capacity = config.hyperparameters.capacity
        self.phase = phase
        self.length = self.get_dataset_length()
        self.position = 0
        self.transformer = Transformer
        self.config = config
        self.dir = self.config.ae_path()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple(torch.Tensor):
        (state, action, next_state, reward, done) = self.get_object_from_file(idx)

        return (state.squeeze(0), action.squeeze(-1).to(torch.float), next_state.squeeze(0), reward, done)

    def get_object_from_file(self, idx) -> Tuple(torch.Tensor):
        file_idx = idx // 1000
        path = self.get_pickle_path(file_idx)
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
            return obj[idx]

    def get_pickle_path(self, file_idx) -> str:
        return  self.dir + "/" + self.phase + "/batch_" + str(file_idx) + ".p"

    def get_dataset_length(self) -> int:
        return 28*1000 if self.phase == 'train' else 2 * 1000