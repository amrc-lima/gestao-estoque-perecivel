const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, AlignmentType, BorderStyle, PageBreak
} = require("docx");

const PLOTS = "/home/claude/estoque_perecivel/plots";

function h1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } }); }
function h2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } }); }
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 150 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}
function code(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: "Courier New", size: 18 })],
    spacing: { after: 100 },
    shading: { type: ShadingType.CLEAR, fill: "F0F0F0" },
  });
}

function imageParagraph(path, width, height) {
  return new Paragraph({
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width, height } })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "DDEBF7" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: !!opts.header, size: 20 })] })],
  });
}

const compTableRows = [
  ["Agente", "Recompensa média", "Nível de serviço", "Desperdício médio", "Falta média", "Caixa final média", "Taxa de falência"],
  ["Aleatório", "563.6 ± 304.1", "75.7%", "2.70", "192.5", "615.6", "10%"],
  ["Heurística (s,S)", "990.9 ± 236.5", "86.2%", "2.00", "108.0", "1040.9", "0%"],
  ["PPO (5 seeds)", "789.8 ± 340.0", "87.1%", "9.10", "109.1", "840.6", "4%"],
  ["DQN (5 seeds)", "931.2 ± 171.5", "84.8%", "1.92", "130.5", "981.2", "0%"],
];

const compTable = new Table({
  width: { size: 9500, type: WidthType.DXA },
  columnWidths: [1700, 1500, 1300, 1400, 1200, 1500, 1200],
  rows: compTableRows.map((row, i) =>
    new TableRow({
      children: row.map((text, j) => cell(text, { header: i === 0, width: [1700, 1500, 1300, 1400, 1200, 1500, 1200][j] })),
    })
  ),
});

const complexityItems = [
  ["Múltiplos objetivos", "A recompensa combina lucro, penalidade por desperdício e penalidade por falta (nível de serviço), escalarizados em uma única função, mas rastreados separadamente para análise."],
  ["Restrições de recurso", "Capacidade máxima de armazenagem (max_stock) e orçamento/caixa limitado, que pode se esgotar."],
  ["Incerteza / eventos estocásticos", "Demanda diária segue distribuição de Poisson com média variável por dia da semana, choques aleatórios de demanda (picos) e falhas aleatórias de entrega (recebe menos do que pediu)."],
  ["Penalidades por ações ruins", "Penalidade explícita por unidade de demanda não atendida (stockout) e por unidade desperdiçada (vencida)."],
  ["Recompensa atrasada", "O pedido feito hoje só chega após o lead time (1 dia); o efeito de comprar demais só aparece quando o produto vence, dias depois."],
  ["Planejamento de vários passos", "O agente precisa antecipar a demanda futura ao decidir quanto pedir hoje, pois o produto não chega instantaneamente."],
  ["Conflito curto x longo prazo", "Comprar muito reduz o risco de falta imediata, mas aumenta o custo de estoque e o desperdício nos dias seguintes."],
  ["Estados parcialmente informativos", "O agente não observa os parâmetros reais da demanda (lambda-base, probabilidade de choque); só vê o histórico recente de demanda observada."],
  ["Dinâmica que muda entre episódios", "A cada reset(), a sazonalidade semanal, a demanda base, a probabilidade de choque e a probabilidade de falha de entrega são sorteadas novamente dentro de faixas plausíveis."],
  ["Custo de ação", "Custo variável (por unidade pedida) e custo fixo de logística toda vez que um pedido é feito, desincentivando pedidos triviais."],
  ["Risco de fracasso / término antecipado", "Se o caixa acumulado cai abaixo de um limite (falência), o episódio termina antecipadamente com penalidade adicional."],
];

const complexityTable = new Table({
  width: { size: 9500, type: WidthType.DXA },
  columnWidths: [2800, 6700],
  rows: [
    new TableRow({ children: [cell("Elemento", { header: true, width: 2800 }), cell("Como foi implementado", { header: true, width: 6700 })] }),
    ...complexityItems.map(([a, b]) => new TableRow({ children: [cell(a, { width: 2800 }), cell(b, { width: 6700 })] })),
  ],
});

const doc = new Document({
  sections: [{
    properties: {},
    children: [
      new Paragraph({ text: "Gestão de Estoque Perecível com Demanda Estocástica", heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, spacing: { after: 100 } }),
      new Paragraph({ text: "Grupo 1 — Trabalho de Aprendizado por Reforço", alignment: AlignmentType.CENTER, spacing: { after: 400 } }),

      h1("1. Contexto e Objetivo"),
      p("Um pequeno centro de distribuição vende produtos perecíveis e precisa decidir, todos os dias, quanto comprar para atender a uma demanda incerta. Comprar pouco gera perda de vendas (stockout); comprar demais gera desperdício de produtos vencidos. O objetivo deste trabalho é modelar esse problema como um processo de decisão de Markov (MDP), implementá-lo como ambiente Gymnasium, e treinar agentes de aprendizado por reforço (PPO e DQN, via Stable-Baselines3) que aprendam uma política de reposição melhor do que abordagens simples (aleatória e heurística de reposição)."),

      h1("2. Modelagem do Ambiente (MDP)"),
      h2("2.1 Estado (observação)"),
      p("O vetor de observação contém, todos normalizados:"),
      bullet("Estoque atual, separado por idade do produto (0 a shelf_life-1 dias), permitindo ao agente perceber o risco de vencimento iminente;"),
      bullet("Quantidade de produto já pedida mas ainda não entregue (pipeline / lead time);"),
      bullet("Histórico das últimas demandas realizadas (janela deslizante), já que a demanda real subjacente não é observável diretamente (observabilidade parcial);"),
      bullet("Dia da semana, codificado em seno/cosseno, capturando o padrão semanal de consumo;"),
      bullet("Caixa acumulada, normalizada."),
      p("Uma variante do ambiente (partial_observability=False) também expõe os parâmetros reais da demanda (lambda-base e probabilidade de choque), usada em um experimento de comparação (ver seção 5)."),

      h2("2.2 Ação"),
      p("Ação discreta: quantidade a pedir hoje, de 0 até um máximo (max_order = 20 unidades). O pedido chega após um lead time de 1 dia e está sujeito a uma probabilidade de falha parcial de entrega."),

      h2("2.3 Recompensa"),
      p("A recompensa diária é o lucro do dia:"),
      code("reward = receita_vendas - custo_compra - custo_fixo_pedido - custo_estoque - penalidade_desperdicio - penalidade_falta"),
      p("Se o caixa acumulado cai abaixo de um limite de falência, uma penalidade adicional de -20 é aplicada e o episódio termina antecipadamente."),

      h2("2.4 Episódio"),
      p("Cada episódio simula um horizonte de 60 dias (truncamento), podendo terminar antes em caso de falência (caixa negativa além do limite). A cada reset(), os parâmetros de demanda do episódio (sazonalidade, probabilidade de choques, probabilidade de falha de entrega) são sorteados novamente, forçando o agente a generalizar em vez de decorar uma única dinâmica."),

      h2("2.5 Visualização"),
      p("O ambiente possui um método render() textual que imprime, a cada dia, o estoque por idade, o caixa, os pedidos pendentes e (no script de demonstração) a demanda, a ação tomada e a recompensa recebida — suficiente para acompanhar o comportamento do agente dia a dia."),

      h1("3. Elementos de Complexidade"),
      p("O enunciado exige pelo menos três elementos de complexidade da lista fornecida. O ambiente implementado contém onze, listados a seguir:"),
      complexityTable,

      new Paragraph({ children: [new PageBreak()] }),
      h1("4. Baseline"),
      p("Duas baselines foram implementadas para comparação:"),
      bullet("Agente aleatório: escolhe a quantidade a pedir uniformemente entre 0 e o máximo permitido, a cada dia."),
      bullet("Heurística de reposição (order-up-to / política s,S): estima a demanda média recente a partir do histórico observado e pede o suficiente para repor o estoque (posição de estoque = estoque atual + pedidos pendentes) até um nível-alvo proporcional ao lead time mais uma margem de segurança. Esta é a regra clássica usada na prática de gestão de estoques."),

      h1("5. Configuração Experimental"),
      h2("5.1 Comparação obrigatória entre configurações"),
      p("Foram comparados dois algoritmos de RL sobre o mesmo ambiente e mesma função de recompensa: PPO (política on-policy, ator-crítico) e DQN (off-policy, valor de ação). Ambos usam MlpPolicy padrão do Stable-Baselines3, adaptados ao espaço de ação discreto do ambiente."),
      new Paragraph({ text: "Hiperparâmetros principais:", spacing: { after: 80 } }),
      bullet("PPO: n_steps=1024, batch_size=256, learning_rate=3e-4, ent_coef=0.01, gamma=0.99"),
      bullet("DQN: learning_rate=1e-3, buffer_size=50000, batch_size=128, exploration_fraction=0.3, gamma=0.99"),
      bullet("Ambos treinados por 60.000 timesteps por semente."),

      h2("5.2 Múltiplas sementes"),
      p("Cada algoritmo foi treinado com 5 sementes diferentes (0, 1, 2, 3, 4), gerando 10 modelos ao todo. A avaliação de cada modelo treinado foi feita em 10 sementes de avaliação fixas (100 a 109), distintas das sementes de treino, totalizando 50 episódios de avaliação por algoritmo. As baselines (aleatória e heurística) foram avaliadas nas mesmas 10 sementes de avaliação."),

      h1("6. Resultados"),
      h2("6.1 Tabela comparativa (médias sobre seeds x episódios de avaliação)"),
      compTable,
      new Paragraph({ text: "", spacing: { after: 200 } }),

      h2("6.2 Curvas de aprendizado"),
      imageParagraph(`${PLOTS}/learning_curves.png`, 550, 344),

      h2("6.3 Comparação entre agentes"),
      imageParagraph(`${PLOTS}/comparison_bars.png`, 550, 400),

      h2("6.4 Trajetória de um episódio (semente surpresa de exemplo)"),
      p("A seguir, a trajetória de um episódio do PPO (semente de treino 0) rodado sob uma semente de avaliação nunca vista durante o treinamento nem escolhida previamente, ilustrando o comportamento dia a dia: demanda, vendas, pedidos, estoque total e caixa acumulado."),
      imageParagraph(`${PLOTS}/episode_trace_surprise_seed.png`, 500, 400),

      new Paragraph({ children: [new PageBreak()] }),
      h1("7. Discussão"),
      p("Todos os agentes treinados (PPO e DQN) superam claramente o agente aleatório em recompensa acumulada, nível de serviço e taxa de falência, evidenciando que ambos aprenderam uma política não trivial de reposição — o requisito mínimo do trabalho."),
      p("Um resultado interessante é que a heurística de reposição (s,S), mesmo sendo uma regra simples e manualmente ajustada, obteve a maior recompensa média entre todas as políticas testadas, com o DQN chegando bem perto (931 vs. 991) e superando-a em desperdício médio (1.92 unidades vs. 2.00). O PPO, por sua vez, atingiu o maior nível de serviço (87.1%) mas com maior variância entre sementes e maior desperdício médio, sugerindo uma política mais agressiva de reposição (pede mais para evitar faltas, ao custo de vencimento)."),
      p("Isso é consistente com a literatura de gestão de estoques: heurísticas (s,S) bem calibradas são difíceis de superar em problemas de baixa dimensionalidade, e um orçamento de treinamento de 60 mil passos por semente é relativamente modesto para RL em ambientes estocásticos com dinâmica que muda a cada episódio. Isso não invalida os agentes treinados — eles aprenderam claramente uma política de reposição sensata e superam o acaso — mas indica que, com mais passos de treinamento, ajuste fino de hiperparâmetros e/ou reward shaping adicional (por exemplo, penalizar desperdício de forma mais suave e gradual em vez de abrupta), é provável que o desempenho de PPO/DQN se aproxime ainda mais ou ultrapasse a heurística de forma consistente."),
      p("O DQN, sendo off-policy, aproveitou melhor o buffer de replay dado o orçamento de passos, apresentando menor variância entre sementes (171.5 de desvio padrão) do que o PPO (340.0), o que sugere maior estabilidade de treinamento neste ambiente específico."),

      h1("8. Demonstração com Semente Surpresa"),
      p("O script scripts/demo.py foi criado especificamente para a demonstração ao vivo exigida no trabalho. Ele carrega um modelo já treinado (PPO ou DQN, escolhendo qual das 5 sementes de treino usar) e roda um episódio completo em uma semente escolhida na hora, imprimindo dia a dia o estoque, a demanda, a ação tomada e a recompensa, e ao final um resumo com nível de serviço, desperdício total e se houve falência."),
      code("python3 scripts/demo.py --algo PPO --train_seed 0 --seed <SEMENTE_DO_PROFESSOR> --render"),
      p("Como a dinâmica de demanda é resorteada a cada reset() (item 'dinâmica muda entre episódios'), rodar em uma semente nova de fato testa a generalização do agente, e não apenas a memorização de um caso fixo."),

      h1("9. Conclusão"),
      p("O ambiente implementado atende a todos os requisitos obrigatórios: é compatível com Gymnasium (observation_space, action_space, reset(), step(), critério de término, função de recompensa e render() textual), contém 11 dos elementos de complexidade exigidos (mínimo de 3), foi comparado contra duas baselines simples (aleatória e heurística), foi comparado em duas configurações de treinamento (PPO vs. DQN), e foi treinado com 5 sementes distintas por configuração. Os resultados mostram que os agentes de RL aprenderam políticas de reposição não triviais e significativamente melhores que o acaso, embora ainda não superem de forma consistente uma heurística clássica bem ajustada dentro do orçamento de treinamento utilizado — um achado honesto e esperado em problemas de controle de estoque de baixa dimensionalidade."),

      h1("Apêndice — Estrutura de Arquivos e Reprodução"),
      code("envs/perishable_env.py      -> ambiente Gymnasium"),
      code("scripts/baselines.py       -> agente aleatório e heurística (s,S)"),
      code("scripts/train.py           -> treina PPO e DQN com 5 sementes cada"),
      code("scripts/evaluate.py        -> avalia baselines e modelos treinados, gera tabelas"),
      code("scripts/plots.py           -> gera os gráficos (curvas de aprendizado, comparação, trajetória)"),
      code("scripts/demo.py            -> demonstração ao vivo com semente escolhida na hora"),
      code("models/*.zip               -> 10 modelos treinados (PPO/DQN x 5 seeds)"),
      code("results/*.csv              -> resultados brutos e tabela-resumo"),
      code("plots/*.png                -> gráficos gerados"),
      p("Para reproduzir do zero: python3 scripts/train.py && python3 scripts/evaluate.py && python3 scripts/plots.py"),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/claude/estoque_perecivel/report/Relatorio_Grupo1_Estoque_Perecivel.docx", buf);
  console.log("Relatório gerado.");
});
