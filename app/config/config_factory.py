import json
import os
import copy
from datetime import datetime
from typing import List, Set, Dict, Tuple, Optional,Union, Any, List, Optional, cast


class HyperParameters:
    def __init__(self, hyperparameter_dict: Dict[str, Union[str, Dict[str, str]]]) -> None:
        self.__dict__.update(hyperparameter_dict)
        self.hyperparameter_dict: Dict[str, Union[str, Dict[str, str]]] = hyperparameter_dict

class ConfigFactory:
    router:Dict[str, str] = {
        'ddqn': os.path.dirname(__file__) + '/default/ddqn.json',
        'default': os.path.dirname(__file__) + '/default/default.json',
    }

    def __init__(self, run_dict: Dict[str, Union[str, Dict[str, str]]]) -> None:
        self.name = run_dict['name']
        self.phase = run_dict['phase']
        self.model = run_dict['model']

        self.debug: bool = None
        self.dataset_directory:str = None
        self.output_directory: str = None

        self.hyperparameters: Any = None

        if 'config' in run_dict:
            self.config = self.get_config(run_dict['config'])
        else:
            self.config = self.get_config({})

        self.config_dict = copy.deepcopy(self.config)
        self.config_to_variable()
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%s")

    def config_to_variable(self) -> None:
        self.__dict__.update(self.config)
        self.hyperparameters = HyperParameters(self.hyperparameters)

    def set_debug_mode(self) -> None:
        self.comet_project_name = "debug"
        self.hyperparameters.epoch = 1
        self.dataloader_num_workers = 0

    def get_config(self, custom_config: Dict[str, Union[str, Dict[str, str]]]) \
            -> Dict[str, Union[str, Dict[str, str]]]:
        # always first load default mapper
        config = self.read_json(self.router['default'])
        model_config = self.get_model_config()

        default_model_phase_config = self.upgrade_config(config, model_config)
        if len(custom_config) == 0:
            return default_model_phase_config
        else:
            return self.upgrade_config(default_model_phase_config, custom_config)

    def get_model_config(self):
        default_model_config = self.read_json(self.router[self.model])

        # only get phases json config
        default_phase_config = copy.copy(default_model_config)
        default_phase_config = default_phase_config['phases'][self.phase]

        # remove phases part of json
        del default_model_config["phases"]

        return self.upgrade_config(default_model_config, default_phase_config)

    # We need to this later for the Comet app.
    def add_device(self) -> None:
        import torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def is_comet_key_set(self) -> bool:
        return 'comet_key' in self.config_dict

    def is_train_phase(self) -> bool:
        return self.phase == 'train'

    def is_test_phase(self) -> bool:
        return self.phase == 'test'

    def is_inference_phase(self) -> bool:
        return self.phase == 'inference'

    def is_debug(self) -> bool:
        return self.debug

    def is_pretrain(self) -> bool:
        return self.hyperparameters.pretrain

    def tensorboard_path(self) -> str:
        if self.debug:
            return self.output_directory + '/tensorboard/debug/' + self.start_time + '/'

        return self.output_directory + '/tensorboard/experiments/' + self.start_time + '/'

    def saver_path(self) -> str:
        if self.debug:
            return self.output_directory + '/models/debug/' + self.start_time + '/'

        return self.output_directory + '/models/experiments/' + self.start_time + '/'

    def save_state_path(self) -> str:
        if self.debug:
            return self.output_directory + '/states/debug/' + self.start_time + '/'

        return self.output_directory + '/states/experiments/' + self.start_time + '/'

    def load_pretrain(self) -> str:
        return self.output_directory + '/data/pretrain.p'


    def special_game_enc(self) -> str:
        return "Breakout-ram-v4"

    @staticmethod
    def read_json(path: str) -> Any:
        with open(path) as json_data_file:
            config = json.load(json_data_file)
        return config

    @staticmethod
    def upgrade_config(head_config: Dict[str, Union[str, Dict[str, str]]], new_config: Dict[str, Union[str, Dict[str, str]]]) \
            -> Dict[str, Union[str, Dict[str, str]]]:
        for key, value in new_config.items():
            if type(value) == dict:
                for key2, value2 in value.items():
                    head_config[key][key2] = value2

                    if type(value2) == dict:
                        raise TypeError
            else:
                head_config[key] = value

        return head_config
