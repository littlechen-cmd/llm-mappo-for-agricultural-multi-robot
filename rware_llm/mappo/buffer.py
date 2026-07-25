"""On-policy rollout storage for a shared-policy cooperative MAPPO update."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from rware_llm.state import MAPPOState


@dataclass
class RolloutBuffer:
    states: List[MAPPOState] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, state, actions, log_probs, value, reward, done):
        self.states.append(state)
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.rewards)

    def as_arrays(self):
        if not self.states:
            raise ValueError("cannot update MAPPO from an empty rollout")
        return {
            "actor_vectors": np.stack([state.actor_vectors for state in self.states]),
            "local_grids": np.stack([state.local_grids for state in self.states]),
            "global_maps": np.stack([state.global_map for state in self.states]),
            "action_masks": np.stack([state.action_masks for state in self.states]),
            "prior_action_probs": np.stack(
                [state.prior_action_probs for state in self.states]
            ),
            "actions": np.stack(self.actions),
            "log_probs": np.stack(self.log_probs),
            "values": np.asarray(self.values, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
        }
