from app.data.data_set.replay.replay import Replay
from app.data.data_set.ae.ae import AE

from app.config.config_factory import ConfigFactory

class DataSetFactory:
    router = {
        'ddqn': {
            'train': Replay
        },
        'ae': {
            'train': AE
        }
    }

    def __init__(self, config: ConfigFactory) -> None:
        self.config = config
        self.dataset_pointer = self.router[config.model][config.phase]
        self.dataset = {}

        if config.is_train_phase():
            self.dataset['train'] = self.dataset_pointer(config, phase='train')
            self.dataset['val'] = self.dataset_pointer(config, phase='val')
        else:
            self.dataset[config.phase] = self.dataset_pointer(config, phase=config.phase)