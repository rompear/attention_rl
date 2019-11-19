from typing import Dict, Union
from torch.utils.tensorboard import SummaryWriter
from app.config.config_factory import ConfigFactory


class Tensorboard:
    def __init__(self, config: ConfigFactory) -> None:
        self.config = config
        self.writer = SummaryWriter(self.config.tensorboard_path())

    def write_scalar(self, title, value, iteration) -> None:
        self.writer.add_scalar(title, value, iteration)

    def write_video(self, title, value, iteration) -> None:
        self.writer.add_video(title, value, global_step=iteration, fps=self.config.fps)

    def write_image(self, title, value, iteration) -> None:
        self.writer.add_image(title, value, global_step=iteration, dataformats='CHW')

    def write_histogram(self, title, value, iteration) -> None:
        self.writer.add_histogram(title, value, iteration)
        
    def write_embedding(self, all_embeddings, metadata, images) -> None:
        self.writer.add_embedding(all_embeddings, metadata=metadata, label_img=images)

    def write_embedding_no_labels(self, all_embeddings, images) -> None:
        self.writer.add_embedding(all_embeddings,label_img=images)