"""
Demonstração ao vivo (para a apresentação): roda a política treinada em uma
semente escolhida na hora pelo professor, mostrando o comportamento dia a dia.

Uso:
    python3 demo.py --algo PPO --train_seed 0 --seed 12345
    python3 demo.py --algo DQN --train_seed 2 --seed 777 --render
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN
from envs.perishable_env import PerishableInventoryEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

ALGO_CLASSES = {"PPO": PPO, "DQN": DQN}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "DQN"], default="PPO")
    parser.add_argument("--train_seed", type=int, default=0, help="qual dos 5 modelos treinados usar (0-4)")
    parser.add_argument("--seed", type=int, required=True, help="semente 'surpresa' escolhida na hora")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, f"{args.algo}_seed{args.train_seed}")
    model = ALGO_CLASSES[args.algo].load(model_path)

    env = PerishableInventoryEnv(randomize_dynamics=True, partial_observability=True, render_mode="ansi")
    obs, info = env.reset(seed=args.seed)
    print(f"=== Demonstração: agente {args.algo} (treinado com seed {args.train_seed}) "
          f"rodando na semente surpresa {args.seed} ===")
    print(f"Dinâmica sorteada para este episódio -> lambda_base={env.base_lambda:.2f}, "
          f"prob_choque={env.shock_prob:.3f}, prob_falha_entrega={env.delivery_fail_prob:.3f}")

    total_reward = 0.0
    total_demand = 0
    total_sold = 0
    total_waste = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_demand += info["demand"]
        total_sold += info["sold"]
        total_waste += info["waste"]
        if args.render:
            print(env.render(), f"| ação(pedido)={info['order_qty']:2d} | recompensa={reward:7.2f}"
                  f"{'  <-- CHOQUE DE DEMANDA' if info['shock'] else ''}")

    service_level = total_sold / total_demand if total_demand > 0 else 1.0
    print("\n=== RESULTADO FINAL DO EPISÓDIO ===")
    print(f"Dias sobrevividos:      {env.day}")
    print(f"Recompensa acumulada:   {total_reward:.2f}")
    print(f"Caixa final:            {env.cash:.2f}")
    print(f"Nível de serviço:       {service_level*100:.1f}%")
    print(f"Total desperdiçado:     {total_waste} unidades")
    print(f"Faliu (early stop)?     {'SIM' if terminated else 'NÃO'}")


if __name__ == "__main__":
    main()
