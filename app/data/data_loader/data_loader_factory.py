from torch.utils.data import DataLoader, Dataset
from app.config.config_factory import ConfigFactory

class DataLoaderFactory:
    def __init__(self, config: ConfigFactory, dataset_phase: Dataset, phase: str) -> None:
        self.router = {
            'ddqn': {
                'train': self.default,
                'test': self.default,
                'inference': self.default,
            },
        }

        self.dataset_phase = dataset_phase
        self.phase = phase
        self.config = config
        self.dataloader = self.router[config.model][config.phase]()

    def default(self) -> DataLoader:
        return DataLoader(self.dataset_phase, batch_size=self.config.hyperparameters.batch_size, shuffle=self.config.hyperparameters.dataloader_shuffle, num_workers=self.config.dataloader_num_workers)
