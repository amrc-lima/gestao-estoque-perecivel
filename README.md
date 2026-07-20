# Gestão de Estoque Perecível com Aprendizado por Reforço (Grupo 1)

Este repositório contém a implementação de um ambiente de simulação e controle de estoque perecível baseado na API **Gymnasium**, além de scripts para treinamento, avaliação e plotagem de agentes de Aprendizado por Reforço (RL) utilizando as bibliotecas **Stable-Baselines3** (PPO e DQN) e baselines clássicas (Aleatório e Heurística de Reposição $(s, S)$).

---

## 📌 Visão Geral do Projeto

A gestão de estoques de produtos perecíveis (como alimentos, medicamentos e flores) é um desafio complexo na cadeia de suprimentos devido ao conflito constante entre evitar a falta de produto (falta de serviço) e evitar o desperdício por vencimento (shelf-life expirado). 

Este projeto modela esse problema como um **Processo de Decisão de Markov (MDP)** com as seguintes características e complexidades:
* **Múltiplos objetivos:** A recompensa é baseada no lucro líquido, penalizando o desperdício e a falta de estoque.
* **Restrições de recursos:** Capacidade máxima de armazenagem limitada (`max_stock`) e caixa/orçamento limitado (`starting_cash`).
* **Eventos estocásticos:** Demanda baseada em distribuição de Poisson com sazonalidade semanal, probabilidade de choques repentinos de demanda e possibilidade de falhas parciais na entrega logística.
* **Recompensa atrasada (Lead Time):** Um pedido feito hoje leva $N$ dias para chegar.
* **Observabilidade parcial:** O agente não observa os parâmetros exatos da demanda (como a taxa média $\lambda$), apenas o histórico de demanda recente.

---

## 📂 Estrutura do Diretório

```text
Grupo1_Estoque_Perecivel/
├── envs/
│   └── perishable_env.py      # Implementação do ambiente Gymnasium
├── models/
│   ├── DQN_seed0.zip ...      # Modelos treinados do DQN (seeds 0 a 4)
│   └── PPO_seed0.zip ...      # Modelos treinados do PPO (seeds 0 a 4)
├── plots/
│   ├── comparison_bars.png    # Gráfico de barras comparando os agentes
│   ├── learning_curves.png    # Curvas de aprendizado PPO vs DQN
│   └── episode_trace_surprise_seed.png # Gráfico com a trajetória de 1 episódio
├── report/
│   ├── Relatorio_Grupo1_Estoque_Perecivel.docx # Relatório final do grupo
│   └── build_report.js        # Script auxiliar para relatórios
├── results/
│   ├── comparison_table.csv   # Tabela agregada com os resultados finais
│   ├── per_episode.csv        # Logs detalhados por episódio de avaliação
│   └── *_monitor.monitor.csv  # Arquivos de monitoramento do treino
├── scripts/
│   ├── baselines.py           # Agentes de controle clássico (Random e Heuristic)
│   ├── demo.py                # Demonstração interativa e detalhada no terminal
│   ├── evaluate.py            # Avaliação comparativa de todos os agentes
│   ├── plots.py               # Script para geração dos gráficos
│   └── train.py               # Treinamento dos agentes PPO e DQN
├── requirements.txt           # Lista de dependências Python
└── README.md                  # Este arquivo de documentação
```

---

## ⚙️ Configuração do Ambiente

Siga as instruções abaixo para configurar o ambiente de execução e rodar os scripts:

### 1. Requisitos
* Python 3.10 ou superior (testado na versão 3.13)
* Windows, Linux ou macOS

### 2. Criação do Ambiente Virtual (venv) e Instalação
No terminal (PowerShell no Windows ou Bash no Linux/macOS), execute:

```bash
# Criar o ambiente virtual na pasta do projeto
python -m venv venv

# Ativar o ambiente virtual (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Ativar o ambiente virtual (Linux/macOS)
source venv/bin/activate

# Instalar as dependências do projeto
pip install -r requirements.txt
```

---

## 🚀 Como Executar o Projeto

Com o ambiente ativado, utilize os seguintes comandos para rodar o código do projeto:

### 🎮 1. Demonstração Interativa (Ver o agente em ação)
Veja o comportamento do agente treinado dia a dia no terminal. É sorteada uma semente "surpresa" de simulação que define parâmetros de taxa de demanda base, propensão a choques e falhas logísticas.
```bash
python scripts/demo.py --algo PPO --train_seed 0 --seed 42 --render
```
* `--algo`: escolha entre `PPO` ou `DQN`.
* `--train_seed`: escolhe qual semente de modelo treinado carregar (`0` a `4`).
* `--seed`: semente numérica aleatória para gerar a dinâmica do episódio.
* `--render`: habilita a visualização dia a dia das vendas, estoque por idade e caixa.

### 📊 2. Avaliação Comparativa (Gerar Resultados)
Roda simulações agregadas comparando a baseline aleatória, a heurística clássica de estoque (regra $(s,S)$ baseada em média móvel da demanda) e os agentes treinados de RL (PPO e DQN):
```bash
python scripts/evaluate.py
```

### 📈 3. Gráficos Comparativos
Gera os gráficos de análise e salva-os na pasta `/plots`:
```bash
python scripts/plots.py
```

---

## 📊 Resultados e Análise Comparativa

Os agentes foram avaliados sob **10 sementes de simulação distintas** (não vistas no treinamento). A tabela abaixo resume as métricas agregadas dos experimentos realizados:

| Agente | Recompensa Média | Desvio Padrão | Nível de Serviço | Desperdício Médio | Taxa de Falência |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Heurística (s, S)** | **990.90** | 236.46 | 86.1% | 2.00 | **0.0%** |
| **DQN** | 931.22 | **171.53** | 84.8% | **1.92** | **0.0%** |
| **PPO** | 789.76 | 339.95 | **87.0%** | 9.10 | 4.0% |
| **Random** | 563.56 | 304.05 | 75.6% | 2.70 | 10.0% |

### Análise Crítica:
1. **Força da Heurística Clássica:** A heurística baseada na regra $(s, S)$ obteve o maior lucro acumulado médio. Isso ilustra que políticas clássicas baseadas em estoque de segurança ajustadas pela média móvel são baselines extremamente robustas e difíceis de serem superadas por RL em cenários simples.
2. **DQN vs PPO:** O DQN superou o PPO em estabilidade e retorno financeiro com 0% de falência. O PPO, por ser um algoritmo *on-policy*, exige mais interações (timesteps) para convergir em problemas de observabilidade parcial e dinâmica oscilante. Nos 60.000 passos executados, o PPO se mostrou mais agressivo nas compras (maior nível de serviço, mas desperdício médio mais elevado e 4% de falência).

---

## ⚡ Nota sobre Treinamento CPU vs GPU (RTX / CUDA)

Se você tentar executar o script de treinamento com suporte a GPU/CUDA ativado no PyTorch, o `stable-baselines3` exibirá um aviso informativo (`UserWarning: You are trying to run PPO on the GPU...`).

* **O porquê:** Como a observação do estoque perecível é baseada em vetores numéricos curtos (`MlpPolicy`) e não em imagens (`CnnPolicy`), o tempo de transferência de dados entre a CPU (onde o Gymnasium simula os passos do dia) e a GPU (onde a rede neural faz previsões) é maior do que o tempo de processamento das contas na própria CPU.
* **Recomendação:** Para este projeto, o treinamento na **CPU** é mais eficiente e rápido. O código executa na GPU sem erros, mas com baixa utilização de hardware.

---

## 🧠 Dinâmica do Ambiente (`PerishableInventoryEnv`)

A modelagem de estoque perecível possui regras de transição específicas implementadas em [perishable_env.py](file:///C:/Users/eduardo-windows/Desktop/ia/Grupo1_Estoque_Perecivel/envs/perishable_env.py):

* **Envelhecimento FIFO:** O estoque total é mantido como um vetor de idades, onde a idade `0` representa produtos recém-chegados e a última posição representa produtos à beira de expirar. Ao vender, o sistema prioriza o produto mais antigo (**First-In, First-Out**).
* **Desperdício (Waste):** Produtos não vendidos na idade máxima expiram ao final do dia, gerando uma penalidade financeira de perda do custo do produto mais uma multa (`waste_penalty`).
* **Lead Time:** Pedidos demoram `lead_time` dias para chegar. Um pedido executado hoje sai do caixa na hora, mas só chega no armazém no dia seguinte.
* **Falência:** Se o caixa acumulado cair abaixo de `bankruptcy_threshold` (padrão: `-80.0`), o episódio é encerrado prematuramente com uma penalidade extra de `-20.0` pontos para punir comportamentos de alto risco.
