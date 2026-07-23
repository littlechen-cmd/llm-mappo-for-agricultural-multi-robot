"""Shared local Actor and centralized global CNN Critic."""

import torch
from torch import nn


class SharedActor(nn.Module):
    """One parameter-shared policy applied independently to every AGV."""

    def __init__(self, vector_dim: int, local_channels: int, action_dim: int):
        super().__init__()
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.local_encoder = nn.Sequential(
            nn.Conv2d(local_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, vectors, local_grids, action_masks=None):
        encoded = torch.cat(
            (self.vector_encoder(vectors), self.local_encoder(local_grids)), dim=-1
        )
        logits = self.policy_head(encoded)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks.bool(), -1.0e9)
        return logits


class GlobalCNNValue(nn.Module):
    """Centralized value estimator over global image-like warehouse features."""

    def __init__(self, global_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(global_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, global_maps):
        return self.value_head(self.encoder(global_maps)).squeeze(-1)
