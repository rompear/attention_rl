from comet_ml import Experiment, ExistingExperiment
from app.config.config_factory import ConfigFactory

class Comet:
    api_key = "key"
    username = "username"

    def __init__(self, config: ConfigFactory) -> None:
        self.config = config

        if 'comet_key' in self.config.config_dict:
            self.comet_key = config.comet_key
            self.experiment = self.register_existing_experiment()
        else:
            self.name = config.comet_project_name
            self.experiment = self.register_experiment()
            self.config.comet_key = self.experiment.get_key()
            self.add_hyperparameters()
            self.experiment.add_tag(config.phase)
            self.experiment.set_name(self.name)
            self.experiment.log_other('model_directory', self.config.model)
            self.experiment.log_other('internal_ref', self.config.start_time)

    def register_existing_experiment(self) -> ExistingExperiment:
        return ExistingExperiment(api_key=self.api_key, previous_experiment=self.comet_key)


    def register_experiment(self) -> Experiment:
        return Experiment(api_key=self.api_key, project_name=self.name, workspace=self.username)

    def add_hyperparameters(self) -> None:
        self.experiment.log_parameters(self.config.hyperparameters.hyperparameter_dict)

    def close(self) -> None:
        self.experiment.end()

    def log_metric(self, name, value, step=None, include_context=True) -> None:
        self.experiment.log_metric(name, value, step=step, include_context=include_context)

    def log_data(self, data, key)-> None:
        for idx in range(len(data['x'])):
            self.log_metric(key, data['y'][idx], step=data['x'][idx])
