import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "rl_mipt_xdg_cache"),
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "rl_mipt_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mdp_taxi_v2.transportation_env import TransportationEnv
    from mdp_taxi_v2.transportation_mdp import TransportationMDP
except ModuleNotFoundError:
    from hw1.mdp_taxi_v2.transportation_env import TransportationEnv
    from hw1.mdp_taxi_v2.transportation_mdp import TransportationMDP


DEFAULT_CONFIG = {
    "name": "base",
    "driver_start_point": (0, 0),
    "destinations": [("B", "b"), ("D", "d"), ("C", "c"), ("A", "a")],
    "passengers": {"A": (1, 1), "B": (2, 2), "C": (3, 3), "D": (4, 4)},
    "locations": {"a": (1, 2), "b": (2, 3), "c": (3, 4), "d": (5, 4)},
}

INFERENCE_CONFIGS = [
    {
        "name": "base",
        "driver_start_point": (0, 0),
        "destinations": [("B", "b"), ("D", "d"), ("C", "c"), ("A", "a")],
        "passengers": {"A": (1, 1), "B": (2, 2), "C": (3, 3), "D": (4, 4)},
        "locations": {"a": (1, 2), "b": (2, 3), "c": (3, 4), "d": (5, 4)},
    },
    {
        "name": "northern_hub",
        "driver_start_point": (0, 0),
        "destinations": [("B", "b"), ("D", "d"), ("C", "c"), ("A", "a")],
        "passengers": {"A": (0, 5), "B": (1, 1), "C": (5, 2), "D": (6, 6)},
        "locations": {"a": (2, 6), "b": (0, 2), "c": (6, 1), "d": (4, 6)},
    },
    {
        "name": "cross_city",
        "driver_start_point": (0, 0),
        "destinations": [("B", "b"), ("D", "d"), ("C", "c"), ("A", "a")],
        "passengers": {"A": (6, 0), "B": (0, 6), "C": (2, 3), "D": (5, 5)},
        "locations": {"a": (6, 2), "b": (2, 6), "c": (4, 1), "d": (1, 5)},
    },
]


@dataclass(frozen=True)
class IterationSnapshot:
    iteration: int
    start_value: float
    greedy_return: float
    max_delta: float


def state_to_tuple(state: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in state)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "driver_start_point": tuple(config["driver_start_point"]),
        "destinations": [tuple(pair) for pair in config["destinations"]],
        "passengers": {
            key: tuple(value) for key, value in config["passengers"].items()
        },
        "locations": {key: tuple(value) for key, value in config["locations"].items()},
    }


def get_action_value(
    mdp: TransportationMDP,
    values: dict[int, float],
    state: int,
    action: int,
    gamma: float,
) -> float:
    total = 0.0
    for next_state, probability in mdp.get_next_states(state, action).items():
        reward = mdp.get_reward(state, action, next_state)
        total += probability * (reward + gamma * values[next_state])
    return total


def select_best_action(
    mdp: TransportationMDP, values: dict[int, float], state: int, gamma: float
) -> tuple[int, float]:
    action_values: list[tuple[float, int]] = []
    for action in mdp.get_possible_actions(state):
        action_values.append(
            (get_action_value(mdp, values, state, action, gamma), action)
        )

    best_value = max(value for value, _ in action_values)
    best_actions = [
        action
        for value, action in action_values
        if math.isclose(value, best_value, abs_tol=1e-12)
    ]
    return min(best_actions), best_value


def extract_policy(
    mdp: TransportationMDP, values: dict[int, float], gamma: float
) -> dict[int, int | None]:
    policy: dict[int, int | None] = {}
    for state in mdp.get_all_states():
        if mdp.is_terminal(state):
            policy[state] = None
            continue
        best_action, _ = select_best_action(mdp, values, state, gamma)
        policy[state] = best_action
    return policy


def evaluate_policy(
    mdp: TransportationMDP, policy: dict[int, int | None], start_state: int = 0
) -> float:
    total_reward = 0.0
    state = start_state

    while not mdp.is_terminal(state):
        action = policy[state]
        if action is None:
            raise ValueError(f"Missing action for non-terminal state {state}.")
        next_state = next(iter(mdp.get_next_states(state, action)))
        total_reward += mdp.get_reward(state, action, next_state)
        state = next_state

    return total_reward


def value_iteration(
    mdp: TransportationMDP,
    gamma: float = 1.0,
    theta: float = 1e-12,
    max_iterations: int = 100,
) -> tuple[dict[int, float], dict[int, int | None], list[IterationSnapshot]]:
    values = {state: 0.0 for state in mdp.get_all_states()}
    history: list[IterationSnapshot] = []

    for iteration in range(1, max_iterations + 1):
        next_values = values.copy()
        delta = 0.0

        for state in mdp.get_all_states():
            if mdp.is_terminal(state):
                next_values[state] = 0.0
                continue

            _, best_value = select_best_action(mdp, values, state, gamma)
            delta = max(delta, abs(best_value - values[state]))
            next_values[state] = best_value

        values = next_values
        policy = extract_policy(mdp, values, gamma)
        history.append(
            IterationSnapshot(
                iteration=iteration,
                start_value=values[0],
                greedy_return=evaluate_policy(mdp, policy),
                max_delta=delta,
            )
        )
        if delta < theta:
            break

    return values, extract_policy(mdp, values, gamma), history


def format_order(destination_pair: tuple[str, str]) -> str:
    passenger, destination = destination_pair
    return f"{passenger}->{destination.lower()}"


def to_int_point(point: tuple[Any, Any]) -> list[int]:
    return [int(point[0]), int(point[1])]


def rollout_policy(
    config: dict[str, Any], mdp: TransportationMDP, policy: dict[int, int | None]
) -> dict[str, Any]:
    env = TransportationEnv(config)
    state = state_to_tuple(env.reset())
    state_index = mdp.state_to_index(state)

    actions_taken: list[int] = []
    orders_taken: list[str] = []
    state_trace: list[list[int]] = [list(state)]
    reward_trace: list[float] = []

    while not mdp.is_terminal(state_index):
        action = policy[state_index]
        if action is None:
            raise ValueError(f"Missing action for non-terminal state {state_index}.")

        next_state, reward, done, _ = env.step(action)
        state = state_to_tuple(next_state)
        state_index = mdp.state_to_index(state)
        actions_taken.append(action)
        orders_taken.append(format_order(config["destinations"][action - 1]))
        reward_trace.append(float(reward))
        state_trace.append(list(state))

        if done:
            break

    return {
        "actions": actions_taken,
        "orders": orders_taken,
        "rewards": reward_trace,
        "total_reward": float(sum(reward_trace)),
        "state_trace": state_trace,
        "driver_path": [to_int_point(point) for point in env.driver_path],
        "rendered_grid": env.render(mode="ansi"),
    }


def plot_learning_curve(history: list[IterationSnapshot], output_path: Path) -> Path:
    iterations = [snapshot.iteration for snapshot in history]
    start_values = [snapshot.start_value for snapshot in history]
    greedy_returns = [snapshot.greedy_return for snapshot in history]
    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    axis.plot(
        iterations,
        start_values,
        marker="o",
        linewidth=2.5,
        label="V(s0)",
        color="#277CEE",
    )
    axis.plot(
        iterations,
        greedy_returns,
        marker="s",
        linewidth=2.5,
        label="Greedy Return",
        color="#F57A20",
    )
    axis.set_title("Value Iteration")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Reward")
    axis.set_xticks(iterations)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, format="jpeg", dpi=100)
    plt.close(figure)
    return output_path


def solve_config(config: dict[str, Any], gamma: float = 1.0) -> dict[str, Any]:
    normalized_config = normalize_config(config)
    mdp = TransportationMDP(normalized_config)
    values, policy, history = value_iteration(mdp, gamma=gamma)
    rollout = rollout_policy(normalized_config, mdp, policy)

    return {
        "name": config.get("name", "scenario"),
        "values": {str(state): value for state, value in values.items()},
        "policy": {
            str(state): action for state, action in policy.items() if action is not None
        },
        "history": [snapshot.__dict__ for snapshot in history],
        "optimal_total_reward": rollout["total_reward"],
        "optimal_order": rollout["orders"],
        "rollout": rollout,
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    figure_path = output_dir / "value_iteration_progress.jpeg"
    results_path = output_dir / "task_2_results.json"

    default_result = solve_config(DEFAULT_CONFIG, gamma=1.0)
    plot_learning_curve(
        [IterationSnapshot(**snapshot) for snapshot in default_result["history"]],
        figure_path,
    )

    scenario_results = [solve_config(config, gamma=1.0) for config in INFERENCE_CONFIGS]
    payload = {
        "algorithm": "value_iteration",
        "gamma": 1.0,
        "figure": figure_path.name,
        "state_count": len(
            TransportationMDP(normalize_config(DEFAULT_CONFIG)).all_states
        ),
        "base_result": default_result,
        "scenario_results": scenario_results,
    }
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Value iteration finished.")
    print(f"States: {payload['state_count']}")
    print(f"Iterations: {len(default_result['history'])}")
    print(f"Optimal order (base): {', '.join(default_result['optimal_order'])}")
    print(f"Optimal reward (base): {default_result['optimal_total_reward']:.0f}")
    print(f"Figure: {figure_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
