from app.config.config_factory import ConfigFactory
from app.util.logger.logger import Logger
from app.util.saver.saver import Savers

from app.model.model_factory import ModelFactory
from app.data.data_set.data_set_factory import DataSetFactory
from app.data.data_loader.data_loader_factory import DataLoaderFactory
from app.worker.worker_factory import WorkerFactory


from app.util.logger.comet.comet import Comet


import json
import argparse

from typing import Callable, Iterator, Union, Optional, Union, Any, List, Optional, cast

class RunFactory:
    def __init__(self, path: str) -> None:
        self.path = path
        self.configs: List[ConfigFactory] = []
        self.run_config: Any = None
        self.parse_runs()
        self.execute_runs()

    def parse_runs(self) -> None:
        run_config = self.read_json(self.path)
        self.run_config = run_config

        for idx, run in enumerate(run_config["runs"]):
            current_config = ConfigFactory(run)
            self.configs.append(current_config)

    def execute_runs(self) -> None:
        for idx, run in enumerate(self.run_config["runs"]):
            current_config = self.configs[idx]

            comet = Comet(current_config)
            current_config.add_device()

            logger = Logger(current_config)
            logger.comet = comet

            savers = Savers(current_config)

            dataset = DataSetFactory(current_config).dataset
            dataloader = {phase: DataLoaderFactory(current_config, dataset_phase, phase) for (phase, dataset_phase) in dataset.items()}

            model = ModelFactory(current_config).model

            trainer = WorkerFactory(current_config, model, logger, dataloader, savers)
            output = {phase: trainer(phase) for phase in dataset.keys()}

    def was_prev_run_train(self, prev_idx) -> bool:
        if prev_idx >= 0:
            return self.configs[prev_idx].is_train_phase()
        return False

    @staticmethod
    def read_json(path) -> Any:
        with open(path) as json_data_file:
            run_config = json.load(json_data_file)
        return run_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--run_path', type=str, default='./run_files/ddqn.json')
    args = vars(parser.parse_args())
    config = RunFactory(args['run_path'])