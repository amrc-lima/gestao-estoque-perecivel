# T4 - Aprendizagem por Reforço

> **Prazo:** Vence hoje às 23:59 | **Encerramento:** Hoje às 23:59
> **Tentativas:** Vários envios permitidos

---

## Instruções
**Trabalho 4 — Aprendizagem por Reforço: Modelagem, Ambiente e Análise Experimental**

Bem-vindos(as) ao quarto e último trabalho prático da disciplina de Inteligência Artificial. Neste trabalho, o foco será **Aprendizagem por Reforço**. 

Diferente dos trabalhos anteriores, aqui o objetivo não é aprender a partir de exemplos rotulados, nem apenas recuperar informação de uma base documental. O desafio é **construir um agente que aprende por interação com um ambiente**, recebendo recompensas ou penalidades pelas consequências de suas ações.

A proposta é que cada grupo projete um problema, implemente um ambiente compatível com `Gymnasium`, treine um ou mais agentes de aprendizagem por reforço e analise criticamente os resultados obtidos.

**Aviso:** Não será suficiente apenas executar um exemplo pronto de `CartPole`, `FrozenLake`, `Taxi`, `LunarLander`, `MountainCar` ou algum tutorial comum disponível na internet. O trabalho deve envolver modelagem própria do ambiente, definição clara de estados, ações e recompensas, e uma análise experimental consistente. 

O exemplo do Mundo do Wumpus apresentado em sala serve como referência de complexidade mínima. O trabalho de cada grupo deve ir além dele, seja por maior riqueza do ambiente, múltiplos objetivos, incerteza, restrições operacionais, *reward design* mais elaborado ou comparação experimental mais cuidadosa.

---

## Objetivos de Aprendizagem
Ao final deste trabalho, espera-se que os grupos sejam capazes de:
- Modelar um problema como tarefa de Aprendizagem por Reforço;
- Definir estados, ações, recompensas, episódios e critérios de término;
- Implementar um ambiente compatível com `Gymnasium`;
- Treinar agentes usando `Stable-Baselines3` ou outra biblioteca apropriada;
- Comparar o desempenho do agente treinado com uma *baseline* simples;
- Executar experimentos com múltiplas sementes e configurações;
- Analisar curvas de aprendizado, estabilidade, falhas e comportamento aprendido;
- Discutir criticamente se o agente realmente aprendeu uma política útil.

---

## Organização dos Grupos
A turma será organizada nos mesmos 6 grupos de 4 pessoas. Cada grupo receberá um projeto diferente. Todos os projetos deverão ser implementados em **Python** e deverão usar, preferencialmente:
- `gymnasium` para implementação do ambiente;
- `stable-baselines3` para treinamento de agentes;
- `numpy`, `pandas`, `matplotlib` ou bibliotecas equivalentes para análise dos resultados.

*Outras bibliotecas de RL poderão ser usadas, desde que o grupo justifique a escolha.*

---

## Regras Gerais Obrigatórias

### 1. Ambiente Próprio
Cada grupo deverá implementar um ambiente próprio compatível com `Gymnasium`. O ambiente deverá conter, no mínimo:
- `observation_space`;
- `action_space`;
- Método `reset()`;
- Método `step()`;
- Critério de término do episódio;
- Função de recompensa;
- Algum tipo de visualização simples, como `render()` textual, gráfico, grid ou animação. (Não é obrigatório fazer uma visualização bonita, mas deve ser possível entender o comportamento do agente).

### 2. Complexidade Mínima
O ambiente não pode ser trivial. Deve conter **pelo menos três** dos seguintes elementos:
- Múltiplos objetivos;
- Restrições de recurso;
- Incerteza ou eventos estocásticos;
- Penalidades por ações ruins;
- Recompensa atrasada;
- Necessidade de planejamento de vários passos;
- Conflitos entre curto e longo prazo;
- Estados parcialmente informativos;
- Dinâmica que muda entre episódios;
- Custo de ação;
- Risco de fracasso ou término antecipado.

### 3. Baseline Obrigatória
Cada grupo deverá comparar o agente treinado com pelo menos uma *baseline* simples, por exemplo:
- Agente aleatório;
- Agente guloso simples;
- Regra heurística definida pelo grupo;
- Política manual razoável.

*O objetivo é mostrar se o agente treinado realmente aprendeu algo útil.*

### 4. Comparação Experimental Obrigatória
Além da *baseline*, cada grupo deve comparar **pelo menos duas configurações**. Exemplos:
- DQN vs PPO;
- PPO vs A2C;
- DQN com duas arquiteturas de rede;
- Mesma técnica com duas funções de recompensa;
- Mesmo agente com diferentes hiperparâmetros;
- Política treinada em ambiente simples vs ambiente mais difícil;
- Treinamento com e sem determinada informação no estado.

### 5. Múltiplas Sementes
Cada grupo deverá executar os experimentos com **pelo menos 5 sementes diferentes**. 
A documentação do `Stable-Baselines3` alerta que resultados totalmente reprodutíveis não são garantidos entre diferentes plataformas, versões de PyTorch ou CPU/GPU, mas recomenda controle de semente para melhorar a reprodutibilidade em uma mesma configuração.

### 6. Demonstração com Semente Surpresa
Na apresentação, o grupo deverá estar preparado para executar a política treinada em uma instância ou semente escolhida na hora pelo professor. Isso não é para "pegar o grupo de surpresa", mas para verificar se a solução não foi ajustada apenas para um caso fixo.
