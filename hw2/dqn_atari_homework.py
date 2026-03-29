from __future__ import annotations

import argparse
import gc
import json
import os
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.dqn.policies import CnnPolicy

gym.register_envs(ale_py)


@dataclass(frozen=True)
class GameConfig:
    name: str
    env_id: str
    dqn_target_mean: float
    dqn_target_std: float
    total_timesteps: int
    buffer_size: int
    learning_starts: int
    eval_freq: int
    n_eval_episodes: int
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.10
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01
    frame_skip: int = 4
    frame_stack: int = 4
    screen_size: int = 84
    noop_max: int = 30
    action_repeat_probability: float = 0.0
    max_grad_norm: float = 10.0
    terminal_on_life_loss: bool = True
    optimizer_name: str = "adam"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    n_steps: int = 1
    optimize_memory_usage: bool = True
    double_dqn: bool = False
    dueling: bool = False
    prioritized_replay: bool = False
    prioritized_alpha: float = 0.6
    prioritized_beta_start: float = 0.4
    prioritized_beta_end: float = 1.0
    prioritized_eps: float = 1e-6

    @property
    def success_threshold(self) -> float:
        return self.dqn_target_mean - self.dqn_target_std


@dataclass(frozen=True)
class RunConfig:
    games: tuple[str, ...] = ("Pong", "Freeway")
    seed: int = 42
    device: str = "auto"
    smoke_test: bool = False
    total_timesteps_override: int | None = None
    eval_freq_override: int | None = None
    n_eval_episodes_override: int | None = None


BASE_GAME_CONFIGS: dict[str, GameConfig] = {
    "Pong": GameConfig(
        name="Pong",
        env_id="ALE/Pong-v5",
        dqn_target_mean=18.9,
        dqn_target_std=1.3,
        total_timesteps=2_000_000,
        buffer_size=80_000,
        learning_starts=20_000,
        eval_freq=50_000,
        n_eval_episodes=10,
    ),
    "Freeway": GameConfig(
        name="Freeway",
        env_id="ALE/Freeway-v5",
        dqn_target_mean=30.3,
        dqn_target_std=0.7,
        total_timesteps=3_000_000,
        buffer_size=120_000,
        learning_starts=50_000,
        eval_freq=50_000,
        n_eval_episodes=20,
        learning_rate=1e-4,
        target_update_interval=8_000,
        exploration_fraction=0.4,
        exploration_final_eps=0.02,
        terminal_on_life_loss=False,
        optimizer_kwargs={"eps": 1.5e-4},
        n_steps=3,
        optimize_memory_usage=False,
        double_dqn=True,
        dueling=True,
        prioritized_replay=True,
    ),
}


def running_in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    return get_ipython() is not None


class DuelingQNetwork(BasePolicy):
    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim

        action_dim = int(self.action_space.n)
        self.value_net = nn.Sequential(
            *create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        )
        self.advantage_net = nn.Sequential(
            *create_mlp(
                self.features_dim, action_dim, self.net_arch, self.activation_fn
            )
        )

    def forward(self, obs: PyTorchObs) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        values = self.value_net(features)
        advantages = self.advantage_net(features)
        return values + advantages - advantages.mean(dim=1, keepdim=True)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = True
    ) -> torch.Tensor:
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class DuelingCnnPolicy(CnnPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return DuelingQNetwork(**net_args).to(self.device)


class PrioritizedNStepReplayBuffer(NStepReplayBuffer):
    def __init__(
        self,
        *args,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise NotImplementedError(
                "PrioritizedNStepReplayBuffer currently supports only n_envs=1."
            )

        self.alpha = alpha
        self.default_beta = beta
        self.eps = eps
        self.priorities = np.zeros(self.buffer_size, dtype=np.float32)
        self.max_priority = 1.0
        self.last_sampled_indices: np.ndarray | None = None
        self.last_importance_weights: torch.Tensor | None = None

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        current_pos = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.priorities[current_pos] = self.max_priority

    def sample(
        self,
        batch_size: int,
        env=None,
        beta: float | None = None,
    ):
        valid_size = self.buffer_size if self.full else self.pos
        if valid_size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        beta = self.default_beta if beta is None else beta
        valid_priorities = np.maximum(self.priorities[:valid_size], self.eps)
        scaled_priorities = np.power(valid_priorities, self.alpha, dtype=np.float32)
        probability_sum = float(scaled_priorities.sum())

        if not np.isfinite(probability_sum) or probability_sum <= 0.0:
            probabilities = np.full(valid_size, 1.0 / valid_size, dtype=np.float32)
        else:
            probabilities = scaled_priorities / probability_sum

        batch_inds = np.random.choice(
            valid_size, size=batch_size, replace=True, p=probabilities
        )
        weights = np.power(valid_size * probabilities[batch_inds], -beta)
        weights /= weights.max()

        self.last_sampled_indices = batch_inds
        self.last_importance_weights = self.to_torch(
            weights.reshape(-1, 1).astype(np.float32)
        )
        return self._get_samples(batch_inds, env=env)

    def update_priorities(self, batch_inds: np.ndarray, priorities: np.ndarray) -> None:
        clipped_priorities = np.maximum(
            np.asarray(priorities, dtype=np.float32).reshape(-1), self.eps
        )
        self.priorities[np.asarray(batch_inds, dtype=np.int64)] = clipped_priorities
        if clipped_priorities.size > 0:
            self.max_priority = max(self.max_priority, float(clipped_priorities.max()))


class EnhancedDQN(DQN):
    def __init__(
        self,
        *args,
        double_dqn: bool = False,
        prioritized_replay: bool = False,
        prioritized_beta_start: float = 0.4,
        prioritized_beta_end: float = 1.0,
        prioritized_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay
        self.prioritized_beta_start = prioritized_beta_start
        self.prioritized_beta_end = prioritized_beta_end
        self.prioritized_eps = prioritized_eps
        super().__init__(*args, **kwargs)

    def _current_beta(self) -> float:
        training_progress = 1.0 - self._current_progress_remaining
        return self.prioritized_beta_start + training_progress * (
            self.prioritized_beta_end - self.prioritized_beta_start
        )

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        td_error_means: list[float] = []

        for _ in range(gradient_steps):
            importance_weights = None
            sampled_indices = None

            if self.prioritized_replay:
                assert isinstance(self.replay_buffer, PrioritizedNStepReplayBuffer)
                replay_data = self.replay_buffer.sample(
                    batch_size,
                    env=self._vec_normalize_env,
                    beta=self._current_beta(),
                )
                importance_weights = self.replay_buffer.last_importance_weights
                sampled_indices = self.replay_buffer.last_sampled_indices
            else:
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )

            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )

            with torch.no_grad():
                if self.double_dqn:
                    next_actions = self.q_net(replay_data.next_observations).argmax(
                        dim=1, keepdim=True
                    )
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    next_q_values = torch.gather(
                        next_q_values, dim=1, index=next_actions
                    )
                else:
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    next_q_values = next_q_values.max(dim=1, keepdim=True).values

                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            td_errors = target_q_values - current_q_values
            elementwise_loss = F.smooth_l1_loss(
                current_q_values, target_q_values, reduction="none"
            )
            if importance_weights is not None:
                elementwise_loss = elementwise_loss * importance_weights
            loss = elementwise_loss.mean()

            losses.append(loss.item())
            td_error_means.append(td_errors.abs().mean().item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.prioritized_replay:
                assert isinstance(self.replay_buffer, PrioritizedNStepReplayBuffer)
                assert sampled_indices is not None
                new_priorities = (
                    td_errors.detach().abs().cpu().numpy().reshape(-1)
                    + self.prioritized_eps
                )
                self.replay_buffer.update_priorities(sampled_indices, new_priorities)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/td_error_abs_mean", np.mean(td_error_means))
        if self.prioritized_replay:
            self.logger.record("train/prioritized_beta", self._current_beta())


def locate_hw2_dir() -> Path:
    candidates: list[Path] = []

    if "__file__" in globals():
        file_path = Path(__file__).resolve()
        candidates.extend([file_path.parent, file_path.parent.parent])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd / "hw2", cwd.parent / "hw2"])

    for candidate in candidates:
        if candidate.name == "hw2" and candidate.exists():
            return candidate
        child = candidate / "hw2"
        if child.exists():
            return child

    raise FileNotFoundError("Could not locate the hw2 directory.")


HW2_DIR = locate_hw2_dir()
ARTIFACTS_DIR = HW2_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"
EVAL_LOGS_DIR = ARTIFACTS_DIR / "eval_logs"
CONFIGS_DIR = ARTIFACTS_DIR / "configs"


def ensure_output_dirs() -> None:
    for path in (ARTIFACTS_DIR, PLOTS_DIR, MODELS_DIR, EVAL_LOGS_DIR, CONFIGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_run_config(run_config: RunConfig) -> RunConfig:
    unique_games = tuple(dict.fromkeys(run_config.games))
    unknown_games = [game for game in unique_games if game not in BASE_GAME_CONFIGS]
    if unknown_games:
        supported_games = ", ".join(sorted(BASE_GAME_CONFIGS))
        raise ValueError(
            f"Unknown games: {unknown_games}. Supported values: {supported_games}."
        )
    return replace(run_config, games=unique_games)


def prepare_game_config(base_config: GameConfig, run_config: RunConfig) -> GameConfig:
    game_config = base_config

    if run_config.total_timesteps_override is not None:
        game_config = replace(
            game_config, total_timesteps=run_config.total_timesteps_override
        )
        adaptive_learning_starts = max(512, game_config.total_timesteps // 10)
        adaptive_buffer_size = max(
            5_000, min(game_config.buffer_size, game_config.total_timesteps)
        )
        game_config = replace(
            game_config,
            learning_starts=min(game_config.learning_starts, adaptive_learning_starts),
            buffer_size=adaptive_buffer_size,
        )
    if run_config.eval_freq_override is not None:
        game_config = replace(game_config, eval_freq=run_config.eval_freq_override)
    if run_config.n_eval_episodes_override is not None:
        game_config = replace(
            game_config, n_eval_episodes=run_config.n_eval_episodes_override
        )

    if run_config.smoke_test:
        game_config = replace(
            game_config,
            total_timesteps=min(game_config.total_timesteps, 4_000),
            buffer_size=min(game_config.buffer_size, 5_000),
            learning_starts=min(game_config.learning_starts, 256),
            eval_freq=min(game_config.eval_freq, 1_000),
            n_eval_episodes=min(game_config.n_eval_episodes, 2),
        )

    return game_config


def make_env(game_config: GameConfig, seed: int, training: bool):
    env_kwargs = {
        "frameskip": 1,
        "repeat_action_probability": game_config.action_repeat_probability,
        "full_action_space": False,
    }
    wrapper_kwargs = {
        "noop_max": game_config.noop_max,
        "frame_skip": game_config.frame_skip,
        "screen_size": game_config.screen_size,
        "terminal_on_life_loss": game_config.terminal_on_life_loss
        if training
        else False,
        "clip_reward": training,
        "action_repeat_probability": game_config.action_repeat_probability,
    }

    env = make_atari_env(
        game_config.env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )
    env = VecFrameStack(env, n_stack=game_config.frame_stack)
    env = VecTransposeImage(env)
    return env


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


def build_policy_kwargs(game_config: GameConfig) -> dict[str, Any]:
    optimizer_name = game_config.optimizer_name.lower()
    optimizer_map = {
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
    }

    if optimizer_name not in optimizer_map:
        supported = ", ".join(sorted(optimizer_map))
        raise ValueError(
            f"Unsupported optimizer '{game_config.optimizer_name}'. Supported: {supported}."
        )

    policy_kwargs: dict[str, Any] = {
        "optimizer_class": optimizer_map[optimizer_name],
    }
    if game_config.optimizer_kwargs:
        policy_kwargs["optimizer_kwargs"] = dict(game_config.optimizer_kwargs)
    return policy_kwargs


def build_replay_buffer_settings(
    game_config: GameConfig,
) -> tuple[type[NStepReplayBuffer] | None, dict[str, Any]]:
    replay_buffer_kwargs: dict[str, Any] = {"handle_timeout_termination": False}

    if game_config.prioritized_replay:
        replay_buffer_kwargs.update(
            {
                "n_steps": game_config.n_steps,
                "gamma": game_config.gamma,
                "alpha": game_config.prioritized_alpha,
                "beta": game_config.prioritized_beta_start,
                "eps": game_config.prioritized_eps,
            }
        )
        return PrioritizedNStepReplayBuffer, replay_buffer_kwargs

    return None, replay_buffer_kwargs


def build_model(game_config: GameConfig, train_env, run_config: RunConfig, device: str):
    replay_buffer_class, replay_buffer_kwargs = build_replay_buffer_settings(
        game_config
    )

    return EnhancedDQN(
        policy=DuelingCnnPolicy if game_config.dueling else CnnPolicy,
        env=train_env,
        learning_rate=game_config.learning_rate,
        buffer_size=game_config.buffer_size,
        learning_starts=game_config.learning_starts,
        batch_size=game_config.batch_size,
        gamma=game_config.gamma,
        train_freq=(game_config.train_freq, "step"),
        gradient_steps=game_config.gradient_steps,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        optimize_memory_usage=game_config.optimize_memory_usage,
        n_steps=game_config.n_steps,
        target_update_interval=game_config.target_update_interval,
        exploration_fraction=game_config.exploration_fraction,
        exploration_initial_eps=game_config.exploration_initial_eps,
        exploration_final_eps=game_config.exploration_final_eps,
        max_grad_norm=game_config.max_grad_norm,
        stats_window_size=100,
        policy_kwargs=build_policy_kwargs(game_config),
        verbose=1,
        seed=run_config.seed,
        device=device,
        double_dqn=game_config.double_dqn,
        prioritized_replay=game_config.prioritized_replay,
        prioritized_beta_start=game_config.prioritized_beta_start,
        prioritized_beta_end=game_config.prioritized_beta_end,
        prioritized_eps=game_config.prioritized_eps,
    )


def get_action_index(env, action_name: str) -> int | None:
    action_meanings = list(env.envs[0].unwrapped.get_action_meanings())
    try:
        return action_meanings.index(action_name)
    except ValueError:
        return None


def evaluate_constant_action(
    env, action: int, n_eval_episodes: int
) -> tuple[float, float]:
    rewards: list[float] = []
    episode_reward = 0.0
    env.reset()

    while len(rewards) < n_eval_episodes:
        _, reward, done, _ = env.step([action])
        episode_reward += float(reward[0])
        if done[0]:
            rewards.append(episode_reward)
            episode_reward = 0.0

    reward_array = np.asarray(rewards, dtype=np.float64)
    return float(reward_array.mean()), float(reward_array.std())


def plot_game_history(
    game_config: GameConfig, history: pd.DataFrame, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    if history.empty:
        ax.text(0.5, 0.5, "No evaluation points available.", ha="center", va="center")
    else:
        ax.plot(
            history["timesteps"],
            history["mean_reward"],
            color="#1f77b4",
            linewidth=2,
            label="Mean eval reward",
        )
        ax.fill_between(
            history["timesteps"],
            history["mean_reward"] - history["std_reward"],
            history["mean_reward"] + history["std_reward"],
            color="#1f77b4",
            alpha=0.15,
            label="±1 std over eval episodes",
        )

    ax.axhline(
        game_config.dqn_target_mean,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Lecture DQN mean",
    )
    ax.axhline(
        game_config.success_threshold,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.5,
        label="Mean - std success threshold",
    )
    ax.set_title(f"{game_config.name}: evaluation reward over training")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_history(results: list[dict[str, Any]], output_path: Path) -> None:
    if not results:
        return

    fig, axes = plt.subplots(
        1, len(results), figsize=(9 * len(results), 5), squeeze=False
    )

    for ax, result in zip(axes[0], results):
        game_config: GameConfig = result["game_config"]
        history: pd.DataFrame = result["history"]

        if history.empty:
            ax.text(
                0.5, 0.5, "No evaluation points available.", ha="center", va="center"
            )
        else:
            ax.plot(
                history["timesteps"],
                history["mean_reward"],
                color="#1f77b4",
                linewidth=2,
            )
            ax.fill_between(
                history["timesteps"],
                history["mean_reward"] - history["std_reward"],
                history["mean_reward"] + history["std_reward"],
                color="#1f77b4",
                alpha=0.15,
            )

        ax.axhline(
            game_config.dqn_target_mean, color="#d62728", linestyle="--", linewidth=1.5
        )
        ax.axhline(
            game_config.success_threshold, color="#ff7f0e", linestyle=":", linewidth=1.5
        )
        ax.set_title(game_config.name)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode reward")
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=2, label="Mean eval reward"),
        plt.Line2D(
            [0],
            [0],
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label="Lecture DQN mean",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#ff7f0e",
            linestyle=":",
            linewidth=1.5,
            label="Mean - std threshold",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if running_in_notebook():
        plt.show()
    plt.close(fig)


def train_single_game(
    game_config: GameConfig, run_config: RunConfig, device: str
) -> dict[str, Any]:
    slug = game_config.name.lower()
    train_env = make_env(game_config, seed=run_config.seed, training=True)
    eval_env = make_env(game_config, seed=run_config.seed + 1, training=False)
    baseline_mean_reward: float | None = None
    baseline_std_reward: float | None = None

    if game_config.name == "Freeway":
        always_up_action = get_action_index(eval_env, "UP")
        if always_up_action is not None:
            baseline_mean_reward, baseline_std_reward = evaluate_constant_action(
                eval_env, always_up_action, game_config.n_eval_episodes
            )
            eval_env.reset()

    best_model_dir = MODELS_DIR / f"{slug}_best"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir = EVAL_LOGS_DIR / slug
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=game_config.success_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        callback_on_new_best=stop_callback,
        n_eval_episodes=game_config.n_eval_episodes,
        eval_freq=game_config.eval_freq,
        log_path=str(eval_log_dir),
        best_model_save_path=str(best_model_dir),
        deterministic=True,
        verbose=1,
    )

    model = build_model(game_config, train_env, run_config, device)

    save_json(
        CONFIGS_DIR / f"{slug}_run_config.json",
        {
            "game_config": asdict(game_config),
            "run_config": asdict(run_config),
            "resolved_device": device,
        },
    )

    model.learn(
        total_timesteps=game_config.total_timesteps,
        callback=eval_callback,
        log_interval=10,
    )

    final_mean_reward, final_std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=game_config.n_eval_episodes,
        deterministic=True,
        return_episode_rewards=False,
    )

    model.save(MODELS_DIR / f"{slug}_last_model")

    eval_timesteps = np.array(eval_callback.evaluations_timesteps, dtype=np.int64)
    eval_results = np.array(eval_callback.evaluations_results, dtype=np.float64)

    if eval_results.size > 0:
        mean_rewards = eval_results.mean(axis=1)
        std_rewards = eval_results.std(axis=1)
    else:
        mean_rewards = np.array([], dtype=np.float64)
        std_rewards = np.array([], dtype=np.float64)

    if eval_timesteps.size == 0 or eval_timesteps[-1] != model.num_timesteps:
        eval_timesteps = np.append(eval_timesteps, model.num_timesteps)
        mean_rewards = np.append(mean_rewards, final_mean_reward)
        std_rewards = np.append(std_rewards, final_std_reward)

    history = pd.DataFrame(
        {
            "timesteps": eval_timesteps,
            "mean_reward": mean_rewards,
            "std_reward": std_rewards,
        }
    )
    history_path = EVAL_LOGS_DIR / f"{slug}_eval_history.csv"
    history.to_csv(history_path, index=False)

    game_plot_path = PLOTS_DIR / f"{slug}_learning_curve.png"
    plot_game_history(game_config, history, game_plot_path)

    best_mean_reward = (
        float(history["mean_reward"].max()) if not history.empty else float("nan")
    )
    solved = (
        bool(best_mean_reward >= game_config.success_threshold)
        if not history.empty
        else False
    )

    result = {
        "game": game_config.name,
        "env_id": game_config.env_id,
        "resolved_device": device,
        "steps_completed": int(model.num_timesteps),
        "target_mean": game_config.dqn_target_mean,
        "target_std": game_config.dqn_target_std,
        "success_threshold": game_config.success_threshold,
        "best_mean_reward": best_mean_reward,
        "final_mean_reward": float(final_mean_reward),
        "final_std_reward": float(final_std_reward),
        "baseline_up_mean_reward": baseline_mean_reward,
        "baseline_up_std_reward": baseline_std_reward,
        "n_steps": game_config.n_steps,
        "double_dqn": game_config.double_dqn,
        "dueling": game_config.dueling,
        "prioritized_replay": game_config.prioritized_replay,
        "solved": solved,
        "history_path": str(history_path.relative_to(HW2_DIR)),
        "plot_path": str(game_plot_path.relative_to(HW2_DIR)),
        "history": history,
        "game_config": game_config,
    }

    train_env.close()
    eval_env.close()
    del model, train_env, eval_env
    gc.collect()

    return result


def run_experiments(run_config: RunConfig) -> pd.DataFrame:
    ensure_output_dirs()
    set_global_seed(run_config.seed)
    device = resolve_device(run_config.device)

    results: list[dict[str, Any]] = []
    for game_name in run_config.games:
        game_config = prepare_game_config(BASE_GAME_CONFIGS[game_name], run_config)
        print(
            f"\n=== Training {game_config.name} ({game_config.env_id}) | "
            f"timesteps={game_config.total_timesteps:,} | device={device} ==="
        )
        result = train_single_game(game_config, run_config, device)
        results.append(result)

    combined_plot_path = PLOTS_DIR / "dqn_learning_curves.png"
    plot_combined_history(results, combined_plot_path)

    summary_rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"history", "game_config"}
        }
        for result in results
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ARTIFACTS_DIR / "dqn_summary.csv", index=False)
    save_json(
        ARTIFACTS_DIR / "dqn_summary.json",
        {
            "results": summary_rows,
            "combined_plot_path": str(combined_plot_path.relative_to(HW2_DIR)),
        },
    )

    print("\nSummary:")
    print(summary_df.to_string(index=False))
    return summary_df


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train DQN on Atari Pong and Freeway.")
    parser.add_argument(
        "--games",
        nargs="+",
        default=list(RunConfig.games),
        help=f"Subset of games to train. Supported: {', '.join(sorted(BASE_GAME_CONFIGS))}.",
    )
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument(
        "--device", type=str, default=RunConfig.device, help="auto, cpu or cuda"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Short integration run for quick verification.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps for each game.",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=None, help="Override evaluation frequency."
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episode count.",
    )
    args = parser.parse_args()

    return normalize_run_config(
        RunConfig(
            games=tuple(args.games),
            seed=args.seed,
            device=args.device,
            smoke_test=args.smoke_test,
            total_timesteps_override=args.total_timesteps,
            eval_freq_override=args.eval_freq,
            n_eval_episodes_override=args.n_eval_episodes,
        )
    )


if __name__ == "__main__":
    run_experiments(parse_args())
