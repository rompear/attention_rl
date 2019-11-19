from app.model.factory_helper.ddqn import DDQNHelper

from typing import Union

class ModelFactory:
    def __init__(self, config) -> None:
        self.config = config

        self.router = {
            'ddqn': {
                'train': DDQNHelper(self.config).DDQN_train,
            },
        }

        self.model = self.router[self.config.model][self.config.phase]()
        self.model = self.model_to_device()

    def model_to_device(self) -> Union[DDQNHelper]:
        for key in self.model.keys():
            if key == 'policy' or key == 'target':
                self.model[key].to(self.config.device)
        return self.model
