"""Gera os gráficos: curvas de aprendizado, comparação entre agentes, e trajetória de um episódio."""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


def load_monitor(algo, seed):
    path = os.path.join(RESULTS_DIR, f"{algo}_seed{seed}_monitor.monitor.csv")
    df = pd.read_csv(path, skiprows=1)
    df["cum_steps"] = df["l"].cumsum()
    return df


def plot_learning_curves():
    plt.figure(figsize=(8, 5))
    for algo, color in [("PPO", "tab:blue"), ("DQN", "tab:orange")]:
        curves = []
        max_len = 0
        raws = []
        for seed in SEEDS:
            df = load_monitor(algo, seed)
            raws.append(df)
            max_len = max(max_len, len(df))
        # interpola todas as curvas em uma grade comum de episódios
        n_points = min(max_len, 200)
        grid = np.linspace(0, 1, n_points)
        interp_rewards = []
        for df in raws:
            x = np.linspace(0, 1, len(df))
            y = df["r"].rolling(10, min_periods=1).mean().values
            interp_rewards.append(np.interp(grid, x, y))
        interp_rewards = np.array(interp_rewards)
        mean_r = interp_rewards.mean(axis=0)
        std_r = interp_rewards.std(axis=0)
        x_axis = grid * 100
        plt.plot(x_axis, mean_r, label=f"{algo} (média 5 seeds)", color=color)
        plt.fill_between(x_axis, mean_r - std_r, mean_r + std_r, alpha=0.2, color=color)
    plt.xlabel("Progresso do treinamento (%)")
    plt.ylabel("Recompensa por episódio (média móvel 10 ep.)")
    plt.title("Curvas de aprendizado — PPO vs DQN (5 sementes cada)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), dpi=130)
    plt.close()


def plot_comparison_bars():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "per_episode.csv"))
    agents = ["Random", "Heuristic", "PPO", "DQN"]
    metrics = [
        ("total_reward", "Recompensa total (lucro acumulado)"),
        ("service_level", "Nível de serviço (demanda atendida)"),
        ("waste_total", "Desperdício total (unidades)"),
        ("stockout_total", "Falta total (unidades)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.flat, metrics):
        means = [df[df.agent == a][col].mean() for a in agents]
        stds = [df[df.agent == a][col].std() for a in agents]
        colors = ["tab:gray", "tab:green", "tab:blue", "tab:orange"]
        ax.bar(agents, means, yerr=stds, capsize=4, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Comparação: baselines vs agentes treinados (médias sobre seeds x episódios de avaliação)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparison_bars.png"), dpi=130)
    plt.close()


def plot_episode_trace(surprise_seed=None):
    """Roda um episódio do melhor agente (PPO seed 0, por padrão) numa semente 'surpresa' e plota a trajetória."""
    from stable_baselines3 import PPO
    from envs.perishable_env import PerishableInventoryEnv

    if surprise_seed is None:
        surprise_seed = 4242

    env = PerishableInventoryEnv(randomize_dynamics=True, partial_observability=True)
    model = PPO.load(os.path.join(ROOT, "models", "PPO_seed0"))

    obs, info = env.reset(seed=surprise_seed)
    done = False
    stock_trace, demand_trace, sold_trace, cash_trace, order_trace = [], [], [], [env.cash], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        stock_trace.append(env.stock.sum())
        demand_trace.append(info["demand"])
        sold_trace.append(info["sold"])
        cash_trace.append(info["cash"])
        order_trace.append(info["order_qty"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(demand_trace, label="Demanda realizada", color="tab:red", alpha=0.7)
    axes[0].plot(sold_trace, label="Vendido", color="tab:green")
    axes[0].plot(order_trace, label="Pedido feito", color="tab:blue", linestyle="--")
    axes[0].legend(); axes[0].set_title(f"Demanda x Vendas x Pedidos (semente surpresa = {surprise_seed})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(stock_trace, color="tab:purple")
    axes[1].set_title("Estoque total ao final do dia")
    axes[1].grid(alpha=0.3)

    axes[2].plot(cash_trace, color="black")
    axes[2].axhline(0, color="red", linestyle=":")
    axes[2].set_title("Caixa acumulado")
    axes[2].set_xlabel("Dia")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "episode_trace_surprise_seed.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    plot_learning_curves()
    plot_comparison_bars()
    plot_episode_trace()
    print("Gráficos salvos em", PLOTS_DIR)
