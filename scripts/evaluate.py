"""
Avaliação e comparação: baselines (aleatório, heurística) vs agentes treinados (PPO, DQN),
agregando resultados sobre múltiplas sementes de treino e múltiplos episódios de avaliação.
Gera: results/comparison_table.csv, results/per_episode.csv, plots/*.png
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from envs.perishable_env import PerishableInventoryEnv
from baselines import RandomAgent, OrderUpToHeuristic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_SEEDS = [0, 1, 2, 3, 4]
EVAL_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # sementes de AVALIAÇÃO, distintas das de treino


def run_episode(env, agent, seed):
    obs, info = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    total_demand = 0
    total_sold = 0
    total_waste = 0
    total_stockout = 0
    days = 0
    cash_trace = [env.cash]
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_demand += info["demand"]
        total_sold += info["sold"]
        total_waste += info["waste"]
        total_stockout += info["stockout"]
        cash_trace.append(info["cash"])
        days += 1
    service_level = total_sold / total_demand if total_demand > 0 else 1.0
    return {
        "total_reward": total_reward,
        "days_survived": days,
        "service_level": service_level,
        "waste_total": total_waste,
        "stockout_total": total_stockout,
        "final_cash": cash_trace[-1],
        "bankrupt": terminated,
    }


def evaluate_agent(agent_name, agent_factory, n_train_seeds=None):
    """agent_factory(train_seed) -> agent  (train_seed=None para baselines sem treino)"""
    rows = []
    seeds_to_use = n_train_seeds if n_train_seeds is not None else [None]
    for tseed in seeds_to_use:
        agent, env = agent_factory(tseed)
        for eseed in EVAL_SEEDS:
            res = run_episode(env, agent, eseed)
            res.update({"agent": agent_name, "train_seed": tseed, "eval_seed": eseed})
            rows.append(res)
    return rows


def make_random_agent(_train_seed):
    env = PerishableInventoryEnv(randomize_dynamics=True, partial_observability=True)
    agent = RandomAgent(env.action_space, seed=_train_seed or 0)
    return agent, env


def make_heuristic_agent(_train_seed):
    env = PerishableInventoryEnv(randomize_dynamics=True, partial_observability=True)
    agent = OrderUpToHeuristic(env)
    return agent, env


def make_sb3_agent(algo_cls, algo_name):
    def factory(train_seed):
        env = PerishableInventoryEnv(randomize_dynamics=True, partial_observability=True)
        model = algo_cls.load(os.path.join(MODELS_DIR, f"{algo_name}_seed{train_seed}"))
        class Wrapper:
            def predict(self, obs, deterministic=True):
                return model.predict(obs, deterministic=deterministic)
        return Wrapper(), env
    return factory


def main():
    all_rows = []
    print("Avaliando baseline aleatória...")
    all_rows += evaluate_agent("Random", make_random_agent, n_train_seeds=TRAIN_SEEDS[:1])
    print("Avaliando heurística (order-up-to)...")
    all_rows += evaluate_agent("Heuristic", make_heuristic_agent, n_train_seeds=TRAIN_SEEDS[:1])
    print("Avaliando PPO (5 sementes de treino)...")
    all_rows += evaluate_agent("PPO", make_sb3_agent(PPO, "PPO"), n_train_seeds=TRAIN_SEEDS)
    print("Avaliando DQN (5 sementes de treino)...")
    all_rows += evaluate_agent("DQN", make_sb3_agent(DQN, "DQN"), n_train_seeds=TRAIN_SEEDS)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "per_episode.csv"), index=False)

    summary = df.groupby("agent").agg(
        reward_mean=("total_reward", "mean"),
        reward_std=("total_reward", "std"),
        service_level_mean=("service_level", "mean"),
        waste_mean=("waste_total", "mean"),
        stockout_mean=("stockout_total", "mean"),
        final_cash_mean=("final_cash", "mean"),
        bankrupt_rate=("bankrupt", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
