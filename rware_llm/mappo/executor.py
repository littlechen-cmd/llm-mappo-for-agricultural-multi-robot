"""Parameter-shared MAPPO executor with a centralized CNN critic."""

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from rware_llm.mappo.buffer import RolloutBuffer
from rware_llm.mappo.networks import GlobalCNNValue, SharedActor


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_learning_rate: float = 3.0e-4
    critic_learning_rate: float = 3.0e-4
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    prior_coefficient: float = 0.1
    prior_mixing_coefficient: float = 0.8
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 128


@dataclass(frozen=True)
class ActionOutput:
    actions: np.ndarray
    log_probs: np.ndarray
    value: float
    entropy: float


class MAPPOExecutor:
    """CTDE executor: shared decentralized Actor plus centralized Critic."""

    def __init__(
        self,
        vector_dim: int,
        local_channels: int,
        global_channels: int,
        action_dim: int,
        config: MAPPOConfig,
        device: str = "cpu",
    ):
        self.config = config
        self.device = torch.device(device)
        self.actor = SharedActor(vector_dim, local_channels, action_dim).to(self.device)
        self.critic = GlobalCNNValue(global_channels).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": config.actor_learning_rate},
                {"params": self.critic.parameters(), "lr": config.critic_learning_rate},
            ]
        )
        self.prior_coefficient = config.prior_coefficient
        self.prior_mixing_coefficient = config.prior_mixing_coefficient

    def set_prior_strength(self, mixing_coefficient: float, coefficient: float) -> None:
        """Set current curriculum values used for sampling and PPO updates."""

        self.prior_mixing_coefficient = max(0.0, float(mixing_coefficient))
        self.prior_coefficient = max(0.0, float(coefficient))

    @torch.no_grad()
    def act(self, state, deterministic: bool = False) -> ActionOutput:
        vectors = self._tensor(state.actor_vectors, torch.float32)
        local_grids = self._tensor(state.local_grids, torch.float32)
        masks = self._tensor(state.action_masks, torch.bool)
        global_map = self._tensor(state.global_map[None, ...], torch.float32)
        actor_logits = self.actor(vectors, local_grids, masks)
        probs = self._behavior_probs(
            actor_logits, self._tensor(state.prior_action_probs, torch.float32)
        )
        distribution = Categorical(probs=probs)
        actions = torch.argmax(probs, dim=-1) if deterministic else distribution.sample()
        return ActionOutput(
            actions=actions.cpu().numpy(),
            log_probs=distribution.log_prob(actions).cpu().numpy(),
            value=float(self.critic(global_map).item()),
            entropy=float(distribution.entropy().mean().item()),
        )

    @torch.no_grad()
    def value(self, global_map) -> float:
        return float(self.critic(self._tensor(global_map[None, ...], torch.float32)).item())

    def update(self, rollout: RolloutBuffer, last_global_map, last_done: bool) -> Dict[str, float]:
        arrays = rollout.as_arrays()
        advantages, returns = self._gae(
            arrays["rewards"], arrays["values"], arrays["dones"], self.value(last_global_map), last_done
        )
        samples = self._flatten_samples(arrays, advantages, returns)
        sample_count = samples["actions"].shape[0]
        advantages_tensor = samples["advantages"]
        samples["advantages"] = (advantages_tensor - advantages_tensor.mean()) / (
            advantages_tensor.std(unbiased=False) + 1.0e-8
        )

        metrics = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "prior_loss": 0.0,
        }
        update_count = 0
        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                metric = self._update_minibatch(samples, indices)
                for key, value in metric.items():
                    metrics[key] += value
                update_count += 1
        return {key: value / max(update_count, 1) for key, value in metrics.items()}

    def _update_minibatch(self, samples, indices):
        actor_logits = self.actor(
            samples["actor_vectors"][indices],
            samples["local_grids"][indices],
            samples["action_masks"][indices],
        )
        prior_probs = samples["prior_action_probs"][indices]
        distribution = Categorical(probs=self._behavior_probs(actor_logits, prior_probs))
        log_probs = distribution.log_prob(samples["actions"][indices])
        ratio = torch.exp(log_probs - samples["old_log_probs"][indices])
        advantages = samples["advantages"][indices]
        surrogate_one = ratio * advantages
        surrogate_two = torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        ) * advantages
        actor_loss = -torch.min(surrogate_one, surrogate_two).mean()
        entropy = distribution.entropy().mean()
        values = self.critic(samples["global_maps"][indices])
        critic_loss = nn.functional.mse_loss(values, samples["returns"][indices])
        prior_loss = nn.functional.kl_div(
            torch.log_softmax(actor_logits, dim=-1),
            prior_probs,
            reduction="batchmean",
        )
        loss = (
            actor_loss
            + self.config.value_coefficient * critic_loss
            - self.config.entropy_coefficient * entropy
            + self.prior_coefficient * prior_loss
        )
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        return {
            "actor_loss": float(actor_loss.detach().item()),
            "critic_loss": float(critic_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "prior_loss": float(prior_loss.detach().item()),
        }

    def _gae(self, rewards, values, dones, last_value, last_done):
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = 0.0 if last_done else last_value
        for index in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[index]
            delta = rewards[index] + self.config.gamma * next_value * non_terminal - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * gae
            advantages[index] = gae
            next_value = values[index]
        return advantages, advantages + values

    def _flatten_samples(self, arrays, advantages, returns):
        steps, agents = arrays["actions"].shape
        def flatten_agent_axis(value):
            return value.reshape((steps * agents, *value.shape[2:]))

        global_maps = np.repeat(arrays["global_maps"], agents, axis=0)
        return {
            "actor_vectors": self._tensor(flatten_agent_axis(arrays["actor_vectors"]), torch.float32),
            "local_grids": self._tensor(flatten_agent_axis(arrays["local_grids"]), torch.float32),
            "global_maps": self._tensor(global_maps, torch.float32),
            "action_masks": self._tensor(flatten_agent_axis(arrays["action_masks"]), torch.bool),
            "prior_action_probs": self._tensor(
                flatten_agent_axis(arrays["prior_action_probs"]), torch.float32
            ),
            "actions": self._tensor(arrays["actions"].reshape(-1), torch.long),
            "old_log_probs": self._tensor(arrays["log_probs"].reshape(-1), torch.float32),
            "advantages": self._tensor(np.repeat(advantages, agents), torch.float32),
            "returns": self._tensor(np.repeat(returns, agents), torch.float32),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "prior_strength": {
                "mixing_coefficient": self.prior_mixing_coefficient,
                "coefficient": self.prior_coefficient,
            },
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        prior_strength = state_dict.get("prior_strength", {})
        self.set_prior_strength(
            prior_strength.get("mixing_coefficient", self.config.prior_mixing_coefficient),
            prior_strength.get("coefficient", self.config.prior_coefficient),
        )

    def _behavior_probs(self, actor_logits, prior_probs):
        """Return a convex actor/prior mixture for rollout and PPO ratios.

        Adding log prior probabilities to actor logits does not make a mixture:
        an arbitrary actor logit can still override a high-confidence rule.  A
        probability-space mixture gives the curriculum coefficient its intended
        meaning and ensures a strong prior really controls early rollouts.
        """

        actor_probs = torch.softmax(actor_logits, dim=-1)
        mixing = self.prior_mixing_coefficient
        if mixing <= 0.0:
            return actor_probs
        probs = (1.0 - mixing) * actor_probs + mixing * prior_probs
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _tensor(self, value, dtype):
        return torch.as_tensor(value, dtype=dtype, device=self.device)
