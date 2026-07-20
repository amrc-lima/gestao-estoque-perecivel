"""Baselines para comparação com os agentes treinados."""
import numpy as np


class RandomAgent:
    """Agente aleatório: pede uma quantidade uniforme entre 0 e max_order."""
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


class OrderUpToHeuristic:
    """
    Política manual razoável: regra (s, S) baseada na média móvel da demanda.
    - Estima a demanda média recente a partir do histórico observado na obs.
    - Define um nível-alvo de estoque S = (lead_time + horizonte_seguranca) * demanda_media_estimada.
    - Pede o suficiente para repor até S, descontando estoque atual + pedidos pendentes.
    Esta é a baseline "heurística de reposição" clássica de gestão de estoque.
    """
    def __init__(self, env, safety_days=1.5):
        self.env = env
        self.safety_days = safety_days

    def predict(self, obs, deterministic=True):
        e = self.env
        current_stock = e.stock.sum()
        pending = e.pending_orders.sum() if e.lead_time > 0 else 0
        recent_demand = e.demand_hist[e.demand_hist > 0]
        avg_demand = recent_demand.mean() if len(recent_demand) > 0 else e.base_lambda if hasattr(e, "base_lambda") else 8.0

        target_level = avg_demand * (e.lead_time + self.safety_days)
        position = current_stock + pending
        order = max(0, target_level - position)
        order = int(np.clip(round(order), 0, e.max_order))
        return order, None
