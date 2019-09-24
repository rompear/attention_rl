import torch
import os
from app.config.config_factory import ConfigFactory

class Savers:
    def __init__(self, config: ConfigFactory):
        self.best_acc = {'train': 0, 'val': 0, 'test': 0}
        self.path = config.saver_path()
        os.mkdir(self.path)
        self.config = config

    def store_model(self, iterations, phase, model) -> None:
        path = self.path + '_' + phase + '_' + str(iterations) + '.p'

        torch.save(model.state_dict(), path)