from app.worker.ddqn.ddqn import DDQN
from app.worker.ae.ae import AE

from app.config.config_factory import ConfigFactory
from app.util.logger.logger import Logger
from app.data.data_loader.data_loader_factory import DataLoaderFactory
from app.util.saver.saver import Savers
from typing import Dict, Any

class WorkerFactory:
    router = {
        'ddqn': {
            'train': DDQN,
            'test': DDQN,
            'inference': DDQN,
        },
        'ae': {
            'train': AE,
            'test': AE,
            'inference': AE,
        },
    }

    def __init__(self, config: ConfigFactory,
                 model: Dict[str: Any],
                 logger: Logger,
                 dataloader: Dict[str: DataLoaderFactory],
                 savers: Savers) -> None:
        self.config = config
        self.worker = self.router[config.model][config.phase]
        self.worker = self.worker(config, model, logger, dataloader, savers)

    def __call__(self, phase: str) -> Any:
        if phase == 'train':
            return self.worker.train_and_val()

        if phase == 'test':
            return self.worker.test()

        if phase == 'inference':
            return self.worker.inference()