"""
Treinamento dos agentes (PPO e DQN) com múltiplas sementes.
Uso: python3 train.py
Gera modelos em ../models/<algo>_seed<N>.zip e logs em ../results/<algo>_seed<N>_monitor.csv
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from envs.perishable_env import PerishableInventoryEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
TOTAL_TIMESTEPS = 60_000

ALGOS = {
    "PPO": lambda env, seed: PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        n_steps=1024, batch_size=256, gamma=0.99,
        learning_rate=3e-4, ent_coef=0.01,
    ),
    "DQN": lambda env, seed: DQN(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=1e-3, buffer_size=50_000,
        learning_starts=1000, batch_size=128,
        gamma=0.99, train_freq=4, target_update_interval=500,
        exploration_fraction=0.3, exploration_final_eps=0.05,
    ),
}


def make_env(seed, monitor_path):
    env = PerishableInventoryEnv(randomize_dynamics=True, partial_observability=True)
    env.reset(seed=seed)
    env = Monitor(env, filename=monitor_path)
    return env


def train_one(algo_name, seed):
    set_random_seed(seed)
    monitor_path = os.path.join(RESULTS_DIR, f"{algo_name}_seed{seed}_monitor")
    env = make_env(seed, monitor_path)
    model = ALGOS[algo_name](env, seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model.save(os.path.join(MODELS_DIR, f"{algo_name}_seed{seed}"))
    env.close()
    print(f"[OK] {algo_name} seed={seed} treinado e salvo.")


if __name__ == "__main__":
    for algo_name in ALGOS:
        for seed in SEEDS:
            train_one(algo_name, seed)
