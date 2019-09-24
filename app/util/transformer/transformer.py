from torchvision import transforms
import torch
from typing import List

class Transformer:
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(40),
        transforms.ToTensor()
    ])

    def transform_frames(self, frames: List[torch.Tensor]) -> torch.Tensor:
        frames = [self.transformer(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # Realign the tensor BCWH -> CBWH
        return frames.to(dtype=torch.float)