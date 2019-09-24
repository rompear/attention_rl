import torch
import gym
import app.util.wrapers_atari.atari_wrappers as Atari_Wrappers
import numpy as np
from  app.config.config_factory import ConfigFactory
from app.util.logger.logger import Logger
from typing import Dict, Any, Tuple, List, Union
from app.util.wrapers_atari.atari_wrappers import LazyFrames

class Trainer:
    def __init__(self, config: ConfigFactory, model: Dict[str:Any], logger: Logger) -> None:
        self.model = model
        self.config = config
        self.phases = config.phase
        self.pred_dict = {} # used for inference and test
        self.iterations = 0 # used for train and val.
        self.logger = logger
        self.create_env()

    def create_env(self) -> None:
        self.env = gym.make(self.config.env_name)
        if self.config.env_name != "CartPole-v0":
            self.env = Atari_Wrappers.wrap_deepmind(self.env, frame_stack=True, scale= not self.config.ram, clip_rewards=True)

    def set_phase_configuration(self, phase: str) -> None:
        if self.config.name == 'ae':
            if phase == 'train':
                self.model['ae'].train()
            else:
                self.model['ae'].eval()
        else:
            if phase == 'train':
                try:
                    self.model['scheduler'].step()
                except:
                    pass
                self.model['policy'].train()
                self.model['target'].eval()
            else:
                self.model['policy'].eval()
                self.model['target'].eval()

    def if_debug(self) -> bool:
        if self.config.debug:
            self.logger.comet.close()
            print("Configuration are in DEBUG mode.")

        return self.config.debug

    def inference(self) -> Dict[str: Any]:
        with torch.no_grad():
            pred_dict = {}
            for phase in ['inference']:
                self.set_phase_configuration(phase)
                pred_dict[phase] = []
                self.get_inference(phase)

        return self.pred_dict['inference']

    def test(self) -> Dict[str: Any]:
        self.model.eval()
        with torch.no_grad():
            self.pred_dict = {}
            for phase in ['test']:
                self.set_phase_configuration(phase)
                self.pred_dict[phase] = []
                self.get_test(phase)

        return self.pred_dict['test']

    def train_and_val(self) -> None:
        self.iterations = 0

        for epoch in range(self.config.hyperparameters.epoch):
            self.epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.config.hyperparameters.epoch - 1))
            print('-' * 10)

            for phase in ['train']:
                self.set_phase_configuration(phase)
                self.get_train(phase)

    def perform_train_step(self, loss: torch.optim, retain_graph: bool = False) -> None:
        loss.backward(retain_graph=retain_graph)
        self.model['optimizer'].step()

    def items_to_device(self, items: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        return tuple([item.to(self.config.device) for item in items])

    def transition_to_tensor(self, state: torch.Tensor,
                             next_state: Union[np.ndarray, LazyFrames],
                             action: torch.Tensor,
                             reward: torch.Tensor,
                             done: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reward = torch.tensor([reward])

        next_state = self.state_to_tensor(next_state)

        done = torch.tensor([done])
        done = done.to(dtype=torch.bool)

        return (state, action, next_state, reward, done)

    def state_to_tensor(self, state: Union[np.ndarray, LazyFrames]) -> torch.Tensor:
        if self.config.env_name == "CartPole-v0":
            state = torch.from_numpy(np.array(state))
        else:
            state = torch.from_numpy(np.array(state._frames))
        state = state.squeeze(-1)
        state = state.to(dtype=torch.float)
        state = state.unsqueeze(0)
        return state

    def get_test(self, phase: str) -> None:
        pass

    def get_inference(self, phase: str) -> None:
        pass

    def get_train(self, phase: str) -> None:
        pass
