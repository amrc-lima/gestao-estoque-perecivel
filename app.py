"""
Interface Streamlit — Gestão de Estoque Perecível (Grupo 1).

Uso (com o venv ativado):
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from baselines import OrderUpToHeuristic, RandomAgent  # noqa: E402
from envs.perishable_env import PerishableInventoryEnv  # noqa: E402

MODELS_DIR = os.path.join(ROOT, "models")
WEEKDAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
TEAM_MEMBERS = [
    "Carlos Erick Dos Santos",
    "Eduardo Américo Santos Lima",
    "Elias Reis",
    "Hugo Gabriel Alves",
]


def render_team_footer():
    names = " · ".join(TEAM_MEMBERS)
    st.html(
        "<div style='"
        "margin-top:2.2rem;padding:1.15rem 0 0.5rem 0;border-top:1px solid #2c3848;"
        "text-align:center;font-family:DM Sans,sans-serif;'>"
        "<p style='color:#e8a838;font-size:0.72rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.12em;margin:0 0 0.45rem 0;'>Grupo 1 · Inteligência Artificial 2026.1</p>"
        f"<p style='color:#e8eef4;font-size:0.95rem;margin:0;line-height:1.55;'>{names}</p>"
        "<p style='color:#8b9aab;font-size:0.8rem;margin:0.45rem 0 0 0;'>"
        "Gestão de estoque perecível com demanda estocástica</p>"
        "</div>"
    )

st.set_page_config(
    page_title="Estoque Perecível · Grupo 1",
    page_icon=":package:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

    :root {
      --bg: #0f1419;
      --panel: #1a222c;
      --panel-2: #232d3a;
      --text: #e8eef4;
      --muted: #8b9aab;
      --accent: #e8a838;
      --ok: #3dba7a;
      --warn: #e07040;
      --line: #2c3848;
    }

    .stApp {
      background:
        radial-gradient(ellipse 80% 50% at 10% -10%, rgba(232, 168, 56, 0.12), transparent 55%),
        radial-gradient(ellipse 60% 40% at 100% 0%, rgba(61, 186, 122, 0.08), transparent 50%),
        linear-gradient(180deg, #0f1419 0%, #121820 100%);
      color: var(--text);
      font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, .brand-title {
      font-family: 'Fraunces', Georgia, serif !important;
      letter-spacing: -0.02em;
    }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #141b24 0%, #10161e 100%);
      border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * { color: var(--text); }

    .brand {
      padding: 0.2rem 0 1.2rem 0;
      border-bottom: 1px solid var(--line);
      margin-bottom: 1rem;
    }
    .brand-kicker {
      color: var(--accent);
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin: 0;
    }
    .brand-title {
      font-size: 1.55rem;
      margin: 0.25rem 0 0.35rem 0;
      color: var(--text);
      line-height: 1.15;
    }
    .brand-sub {
      color: var(--muted);
      font-size: 0.92rem;
      margin: 0;
      line-height: 1.4;
    }

    .metric-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.75rem;
      margin: 0.5rem 0 1.25rem 0;
    }
    .metric {
      background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.9rem 1rem;
    }
    .metric .label {
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0;
    }
    .metric .value {
      font-family: 'Fraunces', Georgia, serif;
      font-size: 1.45rem;
      margin: 0.15rem 0 0 0;
      color: var(--text);
    }
    .metric .value.ok { color: var(--ok); }
    .metric .value.warn { color: var(--warn); }
    .metric .value.accent { color: var(--accent); }

    .status-pill {
      display: inline-block;
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
    }
    .status-run { background: rgba(232, 168, 56, 0.18); color: var(--accent); }
    .status-ok { background: rgba(61, 186, 122, 0.18); color: var(--ok); }
    .status-fail { background: rgba(224, 112, 64, 0.2); color: var(--warn); }

    .dyn-box {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 1rem;
    }
    .dyn-box strong { color: var(--text); }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    .age-panel {
      background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1rem 1.1rem 0.85rem 1.1rem;
    }
    .age-panel h4 {
      font-family: 'Fraunces', Georgia, serif;
      margin: 0 0 0.15rem 0;
      font-size: 1.05rem;
      color: var(--text);
    }
    .age-panel .age-hint {
      color: var(--muted);
      font-size: 0.8rem;
      margin: 0 0 0.9rem 0;
    }
    .age-row {
      display: grid;
      grid-template-columns: 5.2rem 1fr 2.4rem;
      gap: 0.55rem;
      align-items: center;
      margin-bottom: 0.55rem;
    }
    .age-label {
      font-size: 0.82rem;
      color: var(--muted);
      line-height: 1.2;
    }
    .age-label strong {
      display: block;
      color: var(--text);
      font-size: 0.88rem;
    }
    .age-track {
      height: 1.35rem;
      background: #121820;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      position: relative;
    }
    .age-fill {
      height: 100%;
      border-radius: 5px;
      min-width: 0;
      transition: width 0.25s ease;
    }
    .age-count {
      font-family: 'Fraunces', Georgia, serif;
      font-size: 1.05rem;
      text-align: right;
      color: var(--text);
    }
    .age-empty {
      margin-top: 0.35rem;
      padding: 0.7rem 0.8rem;
      border-radius: 8px;
      background: rgba(224, 112, 64, 0.12);
      border: 1px solid rgba(224, 112, 64, 0.35);
      color: #f0b090;
      font-size: 0.86rem;
    }
    .age-foot {
      margin-top: 0.75rem;
      padding-top: 0.65rem;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.82rem;
      display: flex;
      justify-content: space-between;
      gap: 0.5rem;
      flex-wrap: wrap;
    }
    .age-foot strong { color: var(--text); }

    .page-footer {
      margin-top: 2.2rem;
      padding: 1.1rem 0 0.4rem 0;
      border-top: 1px solid var(--line);
      text-align: center;
    }
    .page-footer .footer-kicker {
      color: var(--accent);
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      margin: 0 0 0.45rem 0;
    }
    .page-footer .footer-names {
      color: var(--text);
      font-size: 0.95rem;
      margin: 0;
      line-height: 1.55;
    }
    .page-footer .footer-meta {
      color: var(--muted);
      font-size: 0.8rem;
      margin: 0.45rem 0 0 0;
    }

    @media (max-width: 900px) {
      .metric-strip { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class EpisodeResult:
    days: list[int] = field(default_factory=list)
    demand: list[int] = field(default_factory=list)
    sold: list[int] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    waste: list[int] = field(default_factory=list)
    stockout: list[int] = field(default_factory=list)
    stock_total: list[int] = field(default_factory=list)
    stock_by_age: list[list[int]] = field(default_factory=list)
    cash: list[float] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    shock: list[bool] = field(default_factory=list)
    weekday: list[str] = field(default_factory=list)
    total_reward: float = 0.0
    total_demand: int = 0
    total_sold: int = 0
    total_waste: int = 0
    bankrupt: bool = False
    final_cash: float = 0.0
    days_survived: int = 0
    dynamics: dict = field(default_factory=dict)

    @property
    def service_level(self) -> float:
        if self.total_demand <= 0:
            return 1.0
        return self.total_sold / self.total_demand


@st.cache_resource
def load_sb3_model(algo: str, train_seed: int):
    from stable_baselines3 import DQN, PPO

    path = os.path.join(MODELS_DIR, f"{algo}_seed{train_seed}")
    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(f"Modelo não encontrado: {path}.zip")
    cls = PPO if algo == "PPO" else DQN
    return cls.load(path)


def make_agent(agent_name: str, train_seed: int, env: PerishableInventoryEnv):
    if agent_name == "Random":
        return RandomAgent(env.action_space, seed=train_seed)
    if agent_name == "Heuristic":
        return OrderUpToHeuristic(env)
    model = load_sb3_model(agent_name, train_seed)

    class Wrapper:
        def predict(self, obs, deterministic=True):
            return model.predict(obs, deterministic=deterministic)

    return Wrapper()


def run_episode(agent_name: str, train_seed: int, eval_seed: int) -> EpisodeResult:
    env = PerishableInventoryEnv(
        randomize_dynamics=True,
        partial_observability=True,
        render_mode="ansi",
    )
    agent = make_agent(agent_name, train_seed, env)
    obs, _ = env.reset(seed=eval_seed)

    result = EpisodeResult(
        cash=[float(env.cash)],
        dynamics={
            "lambda_base": float(env.base_lambda),
            "shock_prob": float(env.shock_prob),
            "delivery_fail_prob": float(env.delivery_fail_prob),
        },
    )

    done = False
    terminated = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        result.days.append(env.day)
        result.demand.append(int(info["demand"]))
        result.sold.append(int(info["sold"]))
        result.order.append(int(info["order_qty"]))
        result.waste.append(int(info["waste"]))
        result.stockout.append(int(info["stockout"]))
        result.stock_total.append(int(env.stock.sum()))
        result.stock_by_age.append(env.stock.astype(int).tolist())
        result.cash.append(float(info["cash"]))
        result.reward.append(float(reward))
        result.shock.append(bool(info["shock"]))
        result.weekday.append(WEEKDAYS[(env.day - 1) % 7])
        result.total_reward += float(reward)
        result.total_demand += int(info["demand"])
        result.total_sold += int(info["sold"])
        result.total_waste += int(info["waste"])

    result.bankrupt = bool(terminated)
    result.final_cash = float(env.cash)
    result.days_survived = int(env.day)
    return result


def style_axes(ax):
    ax.set_facecolor("#1a222c")
    ax.tick_params(colors="#8b9aab")
    ax.xaxis.label.set_color("#8b9aab")
    ax.yaxis.label.set_color("#8b9aab")
    ax.title.set_color("#e8eef4")
    for spine in ax.spines.values():
        spine.set_color("#2c3848")
    ax.grid(True, alpha=0.25, color="#2c3848")


AGE_META = [
    ("Idade 0", "Recém-chegado", "#3dba7a"),
    ("Idade 1", "Ainda fresco", "#7bc96f"),
    ("Idade 2", "Envelhecendo", "#e8a838"),
    ("Idade 3", "À beira do vencimento", "#e07040"),
]


def plot_episode(result: EpisodeResult):
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True, gridspec_kw={"hspace": 0.28})
    fig.patch.set_facecolor("#121820")

    days = np.asarray(result.days)
    demand = np.asarray(result.demand)
    sold = np.asarray(result.sold)
    order = np.asarray(result.order)
    cash = np.asarray(result.cash)
    stock_total = np.asarray(result.stock_total)
    shock_mask = np.asarray(result.shock)
    shock_days = days[shock_mask]

    # --- 1) Demanda / vendas / pedidos
    ax = axes[0]
    for d in shock_days:
        ax.axvline(d, color="#e8a838", alpha=0.15, lw=1, zorder=1)
    # realça a falta: área entre demanda e vendido só onde a demanda não foi atendida
    gap_mask = demand > sold
    ax.fill_between(
        days, sold, demand,
        where=gap_mask,
        color="#e07040", alpha=0.28, interpolate=True, zorder=2,
        label="Demanda não atendida",
    )
    ax.plot(days, demand, color="#e8956a", label="Demanda", lw=1.6, alpha=0.9, zorder=3)
    ax.plot(days, sold, color="#3dba7a", label="Vendido", lw=2.1, zorder=4)
    ax.step(days, order, where="mid", color="#5b9bd5", label="Pedido", lw=1.4, linestyle="--", alpha=0.85, zorder=3)
    if len(shock_days):
        ax.scatter(
            shock_days, demand[shock_mask],
            color="#e8a838", s=40, zorder=6, edgecolors="#121820", linewidths=0.7,
            label="_nolegend_",
        )
    ax.set_ylabel("Unidades")
    ax.set_title("Demanda × Vendas × Pedidos")
    # Espaço extra no topo para a legenda nunca cobrir os picos de demanda/choque
    peak = float(demand.max()) if len(demand) else 1.0
    ax.set_ylim(0, peak * 1.5)
    ax.legend(
        facecolor="#1a222c", edgecolor="#2c3848", labelcolor="#e8eef4", fontsize=7.8,
        ncol=4, loc="upper center", framealpha=0.92,
        borderaxespad=0.6, handlelength=1.6, columnspacing=1.1,
    )
    style_axes(ax)

    # --- 2) Estoque total (idades ficam no painel ao lado; aqui evita linhas sobrepostas)
    ax = axes[1]
    ax.fill_between(days, stock_total, color="#4aa8c9", alpha=0.28)
    ax.plot(days, stock_total, color="#6ec4e0", lw=2.2, label="Estoque total")
    empty_days = days[stock_total == 0]
    if len(empty_days):
        ax.scatter(
            empty_days,
            np.zeros_like(empty_days),
            color="#e07040",
            s=28,
            zorder=5,
            label="Sem estoque",
            marker="x",
            linewidths=1.2,
        )
    for d in shock_days:
        ax.axvline(d, color="#e8a838", alpha=0.15, lw=1)
    ax.set_ylabel("Unidades")
    ax.set_title("Estoque total ao longo do episódio")
    ax.set_ylim(bottom=0)
    ymax = max(float(stock_total.max()) * 1.15, 5.0)
    ax.set_ylim(0, ymax)
    ax.legend(facecolor="#1a222c", edgecolor="#2c3848", labelcolor="#e8eef4", fontsize=8, loc="upper left")
    style_axes(ax)

    # --- 3) Caixa
    ax = axes[2]
    cash_x = np.arange(len(cash))  # inclui caixa inicial (dia 0)
    ax.fill_between(cash_x, cash, where=(cash >= 0), color="#e8a838", alpha=0.18)
    ax.fill_between(cash_x, cash, where=(cash < 0), color="#e07040", alpha=0.25)
    ax.plot(cash_x, cash, color="#e8a838", lw=2.0, label="Caixa")
    ax.axhline(0, color="#e07040", ls=":", lw=1.2, label="Linha zero")
    for d in shock_days:
        ax.axvline(d, color="#e8a838", alpha=0.15, lw=1)
    ax.set_ylabel("Caixa")
    ax.set_xlabel("Dia")
    ax.set_title("Caixa acumulado")
    ax.legend(facecolor="#1a222c", edgecolor="#2c3848", labelcolor="#e8eef4", fontsize=8, loc="upper left")
    style_axes(ax)

    fig.tight_layout()
    return fig


def render_stock_ages(result: EpisodeResult, day_idx: int) -> str:
    """Barras horizontais HTML com escala estável (não colapsa quando o estoque é 0)."""
    ages = result.stock_by_age[day_idx]
    total = int(sum(ages))
    episode_peak = max(max(row) for row in result.stock_by_age) if result.stock_by_age else 0
    scale = max(episode_peak, 10)

    rows_html = []
    for i, qty in enumerate(ages):
        name, hint, color = AGE_META[i] if i < len(AGE_META) else (f"Idade {i}", "", "#8b9aab")
        pct = 100.0 * qty / scale
        rows_html.append(
            "<div style='display:grid;grid-template-columns:6.2rem 1fr 2.2rem;"
            "gap:0.55rem;align-items:center;margin:0 0 0.55rem 0;'>"
            "<div style='font-size:0.82rem;color:#8b9aab;line-height:1.25;'>"
            f"<strong style='display:block;color:#e8eef4;font-size:0.88rem;'>{name}</strong>"
            f"{hint}</div>"
            "<div style='height:1.35rem;background:#121820;border:1px solid #2c3848;"
            "border-radius:6px;overflow:hidden;'>"
            f"<div style='height:100%;width:{pct:.1f}%;background:{color};border-radius:5px;'></div>"
            "</div>"
            f"<div style=\"font-family:'Fraunces',Georgia,serif;font-size:1.05rem;"
            f"text-align:right;color:#e8eef4;\">{qty}</div>"
            "</div>"
        )

    empty_html = ""
    if total == 0:
        falta = result.stockout[day_idx]
        empty_html = (
            "<div style='margin-top:0.4rem;padding:0.7rem 0.8rem;border-radius:8px;"
            "background:rgba(224,112,64,0.12);border:1px solid rgba(224,112,64,0.35);"
            "color:#f0b090;font-size:0.86rem;'>"
            "Estoque zerado neste dia"
            f"{f' — demanda não atendida: {falta} un.' if falta else ''}."
            " O pedido feito hoje só chega no dia seguinte (lead time)."
            "</div>"
        )

    day = result.days[day_idx]
    wd = result.weekday[day_idx]
    return (
        "<div style='background:linear-gradient(160deg,#1a222c 0%,#232d3a 100%);"
        "border:1px solid #2c3848;border-radius:12px;padding:1rem 1.1rem 0.9rem 1.1rem;"
        "font-family:DM Sans,sans-serif;color:#e8eef4;'>"
        f"<h4 style=\"font-family:'Fraunces',Georgia,serif;margin:0 0 0.2rem 0;"
        f"font-size:1.05rem;color:#e8eef4;\">Estoque por idade — dia {day} ({wd})</h4>"
        "<p style='color:#8b9aab;font-size:0.8rem;margin:0 0 0.9rem 0;'>"
        f"Escala relativa ao pico do episódio ({scale} un.). "
        "Verde = fresco · âmbar/vermelho = perto de vencer.</p>"
        f"{''.join(rows_html)}"
        f"{empty_html}"
        "<div style='margin-top:0.75rem;padding-top:0.65rem;border-top:1px solid #2c3848;"
        "color:#8b9aab;font-size:0.82rem;display:flex;justify-content:space-between;"
        "gap:0.5rem;flex-wrap:wrap;'>"
        f"<span>Total em prateleira: <strong style='color:#e8eef4;'>{total} un.</strong></span>"
        f"<span>Pedido: <strong style='color:#e8eef4;'>{result.order[day_idx]}</strong> · "
        f"Demanda: <strong style='color:#e8eef4;'>{result.demand[day_idx]}</strong> · "
        f"Vendido: <strong style='color:#e8eef4;'>{result.sold[day_idx]}</strong></span>"
        "</div></div>"
    )

def default_inspect_day(result: EpisodeResult) -> int:
    """Escolhe um dia com estoque (se existir), em vez de começar no dia 1 vazio."""
    for day, ages in zip(result.days, result.stock_by_age):
        if sum(ages) > 0:
            return int(day)
    return int(result.days[min(len(result.days) - 1, 29)])


# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <p class="brand-kicker">Grupo 1 · IA 2026.1</p>
          <h1 class="brand-title">Estoque Perecível</h1>
          <p class="brand-sub">Demonstração interativa do agente de reposição com demanda estocástica.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    agent_name = st.selectbox(
        "Agente",
        ["DQN", "PPO", "Heuristic", "Random"],
        index=0,
        help="DQN e PPO usam modelos já treinados; Heuristic e Random são baselines.",
    )
    train_seed = st.selectbox(
        "Semente de treino do modelo",
        [0, 1, 2, 3, 4],
        index=0,
        disabled=agent_name in ("Heuristic", "Random"),
    )
    eval_seed = st.number_input(
        "Semente surpresa (episódio)",
        min_value=0,
        max_value=10_000_000,
        value=42,
        step=1,
        help="Semente escolhida na hora pelo professor — testa generalização.",
    )

    run_btn = st.button("Rodar episódio", type="primary", use_container_width=True)
    st.caption("Roda 60 dias (ou até falência) com a política selecionada.")

# ------------------------------------------------------------------ Main
st.markdown("### Simulação do episódio")

if run_btn or "last_result" not in st.session_state:
    if run_btn:
        with st.spinner("Simulando episódio..."):
            try:
                st.session_state.last_result = run_episode(agent_name, int(train_seed), int(eval_seed))
                st.session_state.last_cfg = {
                    "agent": agent_name,
                    "train_seed": int(train_seed),
                    "eval_seed": int(eval_seed),
                }
            except FileNotFoundError as exc:
                st.error(str(exc))
                render_team_footer()
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Falha ao carregar/rodar o agente: {exc}")
                render_team_footer()
                st.stop()

if "last_result" not in st.session_state:
    st.info("Configure o agente e a semente na barra lateral e clique em **Rodar episódio**.")
    render_team_footer()
    st.stop()

result: EpisodeResult = st.session_state.last_result
cfg = st.session_state.get("last_cfg", {})

if result.bankrupt:
    status_html = '<span class="status-pill status-fail">Episódio encerrado por falência</span>'
else:
    status_html = '<span class="status-pill status-ok">Episódio completo (horizonte atingido)</span>'

st.markdown(status_html, unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="dyn-box">
      <strong>{cfg.get('agent', '?')}</strong>
      · treino seed <strong>{cfg.get('train_seed', '—')}</strong>
      · episódio seed <strong>{cfg.get('eval_seed', '—')}</strong>
      &nbsp;|&nbsp; λ base = <strong>{result.dynamics['lambda_base']:.2f}</strong>
      · P(choque) = <strong>{result.dynamics['shock_prob']:.3f}</strong>
      · P(falha entrega) = <strong>{result.dynamics['delivery_fail_prob']:.3f}</strong>
    </div>
    """,
    unsafe_allow_html=True,
)

svc = result.service_level * 100
reward_cls = "ok" if result.total_reward >= 700 else ("warn" if result.total_reward < 400 else "accent")
st.markdown(
    f"""
    <div class="metric-strip">
      <div class="metric">
        <p class="label">Recompensa</p>
        <p class="value {reward_cls}">{result.total_reward:.1f}</p>
      </div>
      <div class="metric">
        <p class="label">Caixa final</p>
        <p class="value">{result.final_cash:.1f}</p>
      </div>
      <div class="metric">
        <p class="label">Nível de serviço</p>
        <p class="value ok">{svc:.1f}%</p>
      </div>
      <div class="metric">
        <p class="label">Desperdício</p>
        <p class="value warn">{result.total_waste} un.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_chart, col_age = st.columns([1.55, 1.0])

with col_chart:
    st.pyplot(plot_episode(result), clear_figure=True)

with col_age:
    default_day = default_inspect_day(result)
    day_idx = st.slider(
        "Inspecionar estoque no dia",
        min_value=1,
        max_value=max(result.days),
        value=default_day,
        key=f"inspect_day_{cfg.get('agent')}_{cfg.get('train_seed')}_{cfg.get('eval_seed')}",
    )
    idx = day_idx - 1
    st.html(render_stock_ages(result, idx))
    detail = (
        f"Desperdício={result.waste[idx]} · Falta={result.stockout[idx]} · "
        f"Recompensa={result.reward[idx]:.2f}"
    )
    if result.shock[idx]:
        detail += " · CHOQUE DE DEMANDA"
    st.caption(detail)

st.markdown("#### Diário do episódio")
df = pd.DataFrame(
    {
        "Dia": result.days,
        "Semana": result.weekday,
        "Pedido": result.order,
        "Demanda": result.demand,
        "Vendido": result.sold,
        "Falta": result.stockout,
        "Desperdício": result.waste,
        "Estoque": result.stock_total,
        "Caixa": np.round(result.cash[1:], 2),
        "Recompensa": np.round(result.reward, 2),
        "Choque": result.shock,
    }
)
st.dataframe(df, use_container_width=True, hide_index=True, height=280)

cum = np.cumsum(result.reward)
st.caption(
    f"Dias sobrevivídos: **{result.days_survived}** · "
    f"Recompensa acumulada no último dia: **{cum[-1]:.1f}** · "
    f"Falência: **{'sim' if result.bankrupt else 'não'}**"
)

render_team_footer()
