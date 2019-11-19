from app.worker.base_worker.trainer import Trainer
from app.data.data_structure.transition import Transition
from app.util.processing.processing import Processing
from itertools import count

from app.config.config_factory import ConfigFactory
from app.util.logger.logger import Logger
from app.data.data_loader.data_loader_factory import DataLoaderFactory
from app.util.saver.saver import Savers
from typing import Dict, Any, Tuple

import torch
import random
import math
import numpy as np


class DDQN(Trainer):
    def __init__(self, config: ConfigFactory,
                 model: Dict[str, Any],
                 logger: Logger,
                 dataloaders: Dict[str, DataLoaderFactory],
                 savers: Savers) -> None:
        super().__init__(config, model, logger)

        self.logger = logger
        self.dataloaders = dataloaders
        self.savers = savers

        self.steps_done = 0

    def get_inference(self, phase: str) -> None:
        pass

    def get_train(self, phase: str) -> None:
        state = self.env.reset()
        state = self.state_to_tensor(state)

        ######################
        ##  Administration  ##
        ######################
        total_reward, total_loss, total_steps = 0, 0, 0
        chosen_actions = np.array([0] * (self.config.n_actions))
        reward_signals = np.array([0] * self.config.number_of_lives)

        metadata = self.logger.log_metadata(phase, total_steps)

        if self.config.is_pretrain():
            while(self.steps_done < self.config.hyperparameters.k):
                self.optimize_model(phase)
                self.steps_done += 1


        pretrain = False
        self.memory = []
        for t in count():
            action = self.select_action(state, metadata)

            next_state, reward, done, info = self.env.step(action.item())

            transition = self.transition_to_tensor(state, next_state, action, reward, done, pretrain)

            self.dataloaders[phase].dataset_phase.push(transition)

            loss = 0
            if self.train_network():
                loss = self.optimize_model(phase)

            state = self.state_to_tensor(next_state)

            if self.update_target_network():
                self.model['target'].load_state_dict(self.model['policy'].state_dict())

            if self.perform_test_step():
                self.get_test('test')

            metadata = self.logger.log_metadata(phase, total_steps, info=info)

            ######################
            ##  Administration  ##
            ######################
            chosen_actions[action.item()] += 1
            reward_signals[metadata["current_lives"] - 1] += reward
            total_reward += reward
            total_loss += loss

            self.steps_done += 1
            total_steps += 1

            if done:
                break

        data_stamp = {"reward": total_reward, "duration": total_steps, "loss": total_loss / total_steps, "epsilon": self.calculate_eps()}
        histo_stamp = {"actions": chosen_actions, "rewards": reward_signals}

        if self.epoch % 1 == 0:
            print(data_stamp)
            self.logger.stamp_data(data_stamp, self.epoch, phase, histogram_dict=histo_stamp)

    def get_test(self, phase: str) -> None:
        self.model['target'].eval()
        self.model['policy'].eval()

        state = self.env.reset()
        state = self.state_to_tensor(state)


        chosen_actions = np.array([0]*self.config.n_actions)
        reward_signals = np.array([0]*self.config.number_of_lives)
        total_reward, total_steps = 0, 0

        metadata = self.logger.log_metadata(phase, total_steps)

        frames = []
        for t in count():
            screen = self.env.render(mode='rgb_array')
            if len(frames) < 800:
                frames.append(screen.transpose((2,0,1)))

            action = self.select_action(state, metadata, test=True)
            next_state, reward, done, info = self.env.step(action.item())

            state = self.state_to_tensor(next_state)

            metadata = self.logger.log_metadata(phase, total_steps, info=info)

            ######################
            ##  Administration  ##
            ######################
            reward_signals[metadata["current_lives"] - 1] += reward
            total_reward += reward
            chosen_actions[action.item()] += 1
            total_steps += 1

            if done:
                break

        self.env.close()

        self.model['target'].eval()
        self.model['policy'].train()

        data_stamp = {"reward": total_reward, "duration": total_steps}
        histo_stamp = {"actions": chosen_actions, "rewards": reward_signals}

        # Video information
        frames = np.stack(frames)
        frames = np.expand_dims(frames, axis=0)
        video_stamp = {"plays": frames}

        self.logger.stamp_data(data_stamp, self.epoch, phase, histogram_dict=histo_stamp, video_dict=video_stamp)
        self.savers.store_model(self.steps_done, phase, self.model['policy'])

    def get_current_lives(self, info: Dict[str, Any]) -> int:
        return info['ale.lives']

    def optimize_model(self, phase: str) -> int:
        if self.steps_done < self.config.hyperparameters.start_learning_after and not self.config.is_pretrain():
            return 0

        original_transitions = self.dataloaders[phase].dataset_phase.sample(self.config.hyperparameters.batch_size)

        batch_state, batch_action, batch_next_state, batch_reward, done, pretrain = self.preprocess_tuple(original_transitions)

        margin_loss = self.loss_margin_function(phase, original_transitions, batch_state)

        ddqn_loss = self.ddqn_loss(batch_state, batch_action, batch_next_state, batch_reward, done)

        self.model['optimizer'].zero_grad()
        loss = torch.mean(ddqn_loss) + torch.mean(margin_loss)
        loss.backward()

        self.clamp_gradient()
        self.model['optimizer'].step()

        return loss.item()

    def select_action(self, state: torch.Tensor, metadata: Dict[str, Any], test: bool=False) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.calculate_eps()
        if test:
            eps_threshold = 0.01

        if sample > eps_threshold:
            with torch.no_grad():
                #  pick action with the larger expected reward.
                action =  self.model['policy'](state.to(device=self.config.device)).max(1)[1].view(1, 1).to(dtype=torch.long).cpu()
        else:
            action_number = random.randrange(self.config.n_actions)
            action = torch.tensor([[action_number]], dtype=torch.long)

        return action

    def calculate_eps(self) -> float:
        return self.config.hyperparameters.eps_end + (self.config.hyperparameters.eps_start  - self.config.hyperparameters.eps_end ) * \
                                  math.exp(-1. * self.steps_done / self.config.hyperparameters.eps_decay)

    def update_target_network(self) -> bool:
        return self.steps_done % self.config.hyperparameters.target_update == 0

    def perform_test_step(self) -> bool:
        return self.steps_done % self.config.hyperparameters.show_performance == 0

    def train_network(self) -> bool:
        return self.steps_done % self.config.hyperparameters.training_frequency == 0

    def preprocess_tuple(self, transitions: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        transitions = [(transition[0].to(torch.float), transition[1], transition[2].to(torch.float), transition[3], transition[4], transition[5]) for transition in transitions]
        batch_state, batch_action, batch_next_state, batch_reward, done, pretrain = zip(*transitions)

        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_next_state = torch.cat(batch_next_state)
        done = torch.cat(done)
        pretrain = torch.cat(pretrain)

        return self.items_to_device((batch_state, batch_action, batch_next_state, batch_reward, done, pretrain))

    def ddqn_loss(self, batch_state: torch.Tensor, batch_action, batch_next_state, batch_reward, done) -> torch.Tensor:
        current_q_values = self.model['policy'](batch_state).gather(1, batch_action)

        action_new = self.model['policy'](batch_next_state).detach().max(1)[1].view(-1, 1)

        max_next_q_values = self.model['target'](batch_next_state).detach().gather(1, action_new)
        max_next_q_values[done] = 0
        expected_q_values = batch_reward.view(-1, 1) + (self.config.hyperparameters.gamma * max_next_q_values)

        return current_q_values- expected_q_values.reshape(-1,1)**2

    def loss_margin_function(self, phase: str, transitions: Tuple[torch.Tensor, ...], batch_state: torch.Tensor) -> torch.Tensor:
        current_q_action_values = self.model['policy'](batch_state)

        losses = []
        for idx, transition in enumerate(transitions):
            if transition[0] in self.dataloaders[phase].dataset_phase.pretrain_states:
                pretrain_actions = self.dataloaders[phase].dataset_phase.pretrain_states[transition[0]]
                current_q_action_values[idx, :] += 0.8
                for actions in pretrain_actions:
                    current_q_action_values[idx, actions] -= 0.8

                max_q_value = max(current_q_action_values[idx, :])
                taken_q_value = current_q_action_values[idx, pretrain_actions[0]]
                losses.append(max_q_value - taken_q_value)
            else:
                losses.append(0.0)

        return self.items_to_device((torch.tensor(losses).to(torch.float)))[0]

    def clamp_gradient(self, min=-1, max= 1):
        for param in self.model['policy'].parameters():
            param.grad.data.clamp_(min, max)