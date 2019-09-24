import numpy as np
from typing import List, Set, Dict, Tuple, Optional,Union, Any, List, Optional, cast
from app.config.config_factory import ConfigFactory

from app.util.logger.comet.comet import Comet
from app.util.logger.tensorboard.tensorboard import Tensorboard
from app.config.config_factory import ConfigFactory

class Logger:
    def __init__(self, config: ConfigFactory) -> None:
        self.current_phase = 'null'

        self.config = config
        self.tensorboard_logger = Tensorboard(config)

        self.BEGIN_VALUE = 0
        self.LAST_ITEM = -1

    def stamp_data(self, scalar_dict: Dict[str: Any], iteration, phase, histogram_dict: Dict[str: Any] = None, video_dict: Dict[str: Any] = None) -> None:
        for key, value in scalar_dict.items():
            self.tensorboard_logger.write_scalar(key + '_' + phase, value, iteration)
            self.comet.log_metric(key + '_' + phase, value, step=iteration)

        if histogram_dict:
            for key, value in histogram_dict.items():
                self.tensorboard_logger.write_histogram(key + '_' + phase, value, iteration)

        if video_dict:
            for key, value in video_dict.items():
                self.tensorboard_logger.write_video(key + '_' + phase, value, iteration)

    def is_new_phase(self, phase: str) -> bool:
        return phase != self.current_phase

    def is_first_log(self, iteration: int) -> bool:
        return iteration == self.BEGIN_VALUE

    def log_metadata(self, phase: str, steps: int, info = None) -> Dict[str:any]:
        if self.config.env_name == self.config.special_game_enc():
            if steps == 0:
                new_ball = True
                if phase == 'train':
                    self.current_lives = {'train' : self.config.number_of_lives, 'test': self.config.number_of_lives}

            else:
                ######################
                ## Ask for new Ball ##
                ######################
                temp_lives = info['ale.lives']
                if temp_lives != self.current_lives[phase]:
                    new_ball = True
                    self.current_lives[phase] = temp_lives
                else:
                    new_ball = False

            return {"new_ball": new_ball,  "current_lives": self.current_lives[phase]}

        return {"current_lives": 1}