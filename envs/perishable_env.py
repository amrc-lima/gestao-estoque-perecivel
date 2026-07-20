"""
Ambiente Gymnasium: Gestão de estoque perecível com demanda estocástica.

Grupo 1 - Trabalho de Aprendizado por Reforço.

Elementos de complexidade presentes (>= 3 exigidos, aqui ~10 implementados):
 1. Múltiplos objetivos      -> reward decomposto em lucro, desperdício e nível de serviço
 2. Restrições de recurso    -> capacidade máxima de armazenagem e orçamento (caixa) limitado
 3. Incerteza/eventos estocásticos -> demanda Poisson variável + choques de demanda + falhas de entrega
 4. Penalidades por ações ruins    -> penalidade por falta (stockout) e por desperdício
 5. Recompensa atrasada      -> lead time de entrega e produtos que só "estragam" dias depois
 6. Planejamento multi-passos -> pedido feito hoje só chega depois (lead time), efeito de overstock só aparece dias depois (validade)
 7. Conflito curto x longo prazo -> comprar muito reduz falta no curto prazo mas aumenta desperdício e custo de estoque depois
 8. Estados parcialmente informativos -> agente não observa os parâmetros reais da demanda, só o histórico recente
 9. Dinâmica muda entre episódios -> sazonalidade e propensão a choques são sorteadas a cada reset()
10. Custo de ação            -> custo fixo de pedido + custo variável por unidade comprada
11. Risco de fracasso/término antecipado -> caixa negativa além de um limite encerra o episódio com penalidade
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PerishableInventoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 2}

    def __init__(
        self,
        shelf_life: int = 4,          # dias que o produto permanece vendável (idade 0..shelf_life-1)
        max_stock: int = 40,          # capacidade máxima de armazenagem (restrição de recurso)
        max_order: int = 20,          # quantidade máxima que pode ser pedida por dia
        lead_time: int = 1,           # dias entre pedido e chegada (recompensa atrasada / planejamento)
        horizon: int = 60,            # dias por episódio
        demand_history_len: int = 5,  # janela observada de demanda passada (obs. parcial)
        unit_price: float = 4.0,      # preço de venda por unidade
        unit_cost: float = 2.0,       # custo de compra por unidade
        holding_cost: float = 0.15,   # custo de manter 1 unidade em estoque por dia
        waste_penalty: float = 2.5,   # penalidade por unidade que estraga (perda total do custo + multa)
        stockout_penalty: float = 1.5,# penalidade extra por unidade de demanda não atendida
        order_fixed_cost: float = 1.0,# custo fixo de logística sempre que um pedido é feito (custo de ação)
        starting_cash: float = 50.0,
        bankruptcy_threshold: float = -80.0,  # limite de caixa negativa -> término antecipado
        randomize_dynamics: bool = True,       # dinâmica muda entre episódios
        partial_observability: bool = True,    # se False, também expõe os parâmetros reais da demanda (para o experimento de comparação)
        render_mode: str | None = None,
    ):
        super().__init__()
        self.shelf_life = shelf_life
        self.max_stock = max_stock
        self.max_order = max_order
        self.lead_time = lead_time
        self.horizon = horizon
        self.demand_history_len = demand_history_len
        self.unit_price = unit_price
        self.unit_cost = unit_cost
        self.holding_cost = holding_cost
        self.waste_penalty = waste_penalty
        self.stockout_penalty = stockout_penalty
        self.order_fixed_cost = order_fixed_cost
        self.starting_cash = starting_cash
        self.bankruptcy_threshold = bankruptcy_threshold
        self.randomize_dynamics = randomize_dynamics
        self.partial_observability = partial_observability
        self.render_mode = render_mode

        # Ação: quantidade discreta a pedir, de 0 a max_order (passo de 1 unidade)
        self.action_space = spaces.Discrete(self.max_order + 1)

        # Observação:
        #  - shelf_life valores: estoque por idade (normalizado por max_stock)
        #  - lead_time valores: pedidos pendentes ainda não chegaram (normalizado)
        #  - demand_history_len valores: demanda realizada nos últimos dias (normalizada)
        #  - 2 valores: dia da semana (sin/cos)
        #  - 1 valor: caixa normalizada
        # se partial_observability=False, adiciona 2 valores extras (lambda base e prob. de choque reais)
        obs_dim = self.shelf_life + self.lead_time + self.demand_history_len + 2 + 1
        if not self.partial_observability:
            obs_dim += 2
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------ #
    def _sample_dynamics(self):
        """Sorteia os parâmetros de demanda do episódio (dinâmica muda entre episódios)."""
        if self.randomize_dynamics:
            self.base_lambda = self._rng.uniform(6.0, 14.0)
            self.weekday_mult = self._rng.uniform(0.7, 1.4, size=7)
            self.shock_prob = self._rng.uniform(0.03, 0.12)
            self.shock_mult = self._rng.uniform(1.8, 3.0)
            self.delivery_fail_prob = self._rng.uniform(0.02, 0.10)
        else:
            self.base_lambda = 10.0
            self.weekday_mult = np.array([1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 0.8])
            self.shock_prob = 0.06
            self.shock_mult = 2.2
            self.delivery_fail_prob = 0.05

    def _demand_today(self):
        dow = self.day % 7
        lam = self.base_lambda * self.weekday_mult[dow]
        shock = self._rng.random() < self.shock_prob
        if shock:
            lam *= self.shock_mult
        d = self._rng.poisson(lam)
        return int(d), bool(shock)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._sample_dynamics()

        self.day = 0
        self.cash = self.starting_cash
        self.stock = np.zeros(self.shelf_life, dtype=np.int64)  # idade 0..shelf_life-1
        self.pending_orders = np.zeros(self.lead_time, dtype=np.int64) if self.lead_time > 0 else np.zeros(0, dtype=np.int64)
        self.demand_hist = np.zeros(self.demand_history_len, dtype=np.float64)

        # métricas acumuladas (para análise, não usadas no reward diretamente)
        self.total_sold = 0
        self.total_demand = 0
        self.total_waste = 0
        self.total_profit = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        stock_norm = self.stock / self.max_stock
        pending_norm = self.pending_orders / self.max_order if self.lead_time > 0 else np.zeros(0)
        hist_norm = self.demand_hist / max(self.base_lambda * 2.5, 1e-6)
        dow = self.day % 7
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        cash_norm = np.clip(self.cash / 100.0, -5, 5)

        parts = [stock_norm, pending_norm, hist_norm, [dow_sin, dow_cos], [cash_norm]]
        if not self.partial_observability:
            parts.append([self.base_lambda / 14.0, self.shock_prob])
        obs = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    def step(self, action):
        action = int(np.clip(action, 0, self.max_order))
        order_qty = action

        # 1) chegada do pedido feito `lead_time` dias atrás
        if self.lead_time > 0:
            arriving = self.pending_orders[0]
            if arriving > 0 and self._rng.random() < self.delivery_fail_prob:
                # falha parcial de entrega (incerteza na cadeia de suprimento)
                arriving = int(arriving * self._rng.uniform(0.4, 0.8))
            # desloca a fila de pedidos pendentes
            self.pending_orders = np.roll(self.pending_orders, -1)
            self.pending_orders[-1] = 0
        else:
            arriving = order_qty  # sem lead time, chega no mesmo dia

        # produto que chega entra como idade 0 (mas respeita capacidade máxima)
        current_total = self.stock.sum()
        space_left = max(self.max_stock - current_total, 0)
        received = min(arriving, space_left)
        self.stock[0] += received

        # 2) demanda do dia
        demand, shock = self._demand_today()
        self.demand_hist = np.roll(self.demand_hist, -1)
        self.demand_hist[-1] = demand

        # 3) vendas: FIFO, prioriza produtos mais velhos
        remaining_demand = demand
        sold = 0
        for age in range(self.shelf_life - 1, -1, -1):
            take = min(self.stock[age], remaining_demand)
            self.stock[age] -= take
            sold += take
            remaining_demand -= take
            if remaining_demand <= 0:
                break
        stockout = remaining_demand  # demanda não atendida

        # 4) vencimento: o que está na idade mais velha (shelf_life-1) e não foi vendido, estraga
        waste = int(self.stock[-1])
        self.stock[-1] = 0
        # envelhece o estoque (desloca idades)
        self.stock = np.roll(self.stock, 1)
        self.stock[0] = 0  # idade 0 será preenchida no próximo passo pelo que chegar

        # 5) custos do pedido feito HOJE (custo de ação, paga-se ao pedir, produto chega depois)
        if self.lead_time > 0:
            self.pending_orders[-1] += order_qty
        purchase_cost = order_qty * self.unit_cost
        fixed_cost = self.order_fixed_cost if order_qty > 0 else 0.0

        # 6) cálculo financeiro do dia
        revenue = sold * self.unit_price
        holding = self.stock.sum() * self.holding_cost
        waste_cost = waste * self.waste_penalty
        stockout_cost = stockout * self.stockout_penalty

        profit = revenue - purchase_cost - fixed_cost - holding - waste_cost - stockout_cost
        self.cash += profit

        # reward escalarizado (multiobjetivo: lucro, desperdício, nível de serviço)
        reward = profit

        self.total_sold += sold
        self.total_demand += demand
        self.total_waste += waste
        self.total_profit += profit

        self.day += 1
        terminated = False
        truncated = False

        if self.cash <= self.bankruptcy_threshold:
            terminated = True
            reward -= 20.0  # penalidade adicional por falência (risco de término antecipado)

        if self.day >= self.horizon:
            truncated = True

        obs = self._get_obs()
        info = {
            "sold": sold,
            "demand": demand,
            "stockout": stockout,
            "waste": waste,
            "profit": profit,
            "cash": self.cash,
            "shock": shock,
            "order_qty": order_qty,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def render(self):
        bars = " ".join(f"a{ i}:{v:2d}" for i, v in enumerate(self.stock))
        line = (
            f"Dia {self.day:3d} | Estoque[{bars}] | Caixa: {self.cash:7.2f} | "
            f"Pend: {list(self.pending_orders)}"
        )
        if self.render_mode == "human":
            print(line)
        return line
