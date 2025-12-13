# Previsão do Consumo de Eletricidade em Instituições Educacionais: Uma abordagem de Cooperative Ensemble Learning aplicada a modelos de Regressão e Classificação
![Status](https://img.shields.io/badge/status-concluído-green)
![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC_ND_4.0-lightgrey.svg)

Este repositório documenta os experimentos realizados para a dissertação de mestrado no âmbito do Programa de Pós-Graduação em Engenharia de Produção e Sistemas (PPGEPS) da Universidade Tecnológica Federal do Paraná (UTFPR) campus Pato Branco.

O foco da pesquisa é a aplicação e avaliação de modelos de aprendizado de máquina para a previsão do consumo de eletricidade em uma instituição educacional *multicampi*, utilizando dados históricos, climáticos e acadêmicos.

---

## 🎯 Contexto do Problema

A gestão eficiente de recursos energéticos é um desafio crescente para instituições de ensino, especialmente em um cenário de restrições orçamentárias e crescente demanda. Este trabalho aborda a complexidade da previsão de consumo de eletricidade, que é influenciada por fatores únicos como calendários acadêmicos, variáveis climáticas e múltiplos perfis de consumo em diferentes *campi*.

A literatura ainda carece de soluções que explorem essa previsão considerando os aspectos específicos tratados nesta dissertação, e este trabalho busca preencher essa lacuna.

## 🔬 Contribuições e Experimentos

Os experimentos neste repositório exploram seis contribuições principais para a área de pesquisa:

1.  **Aplicação em Ambientes Educacionais:** Uso de dados de uma instituição *multicampi* real (IFPR, localizado no Paraná), combinando consumo histórico, variáveis climáticas e dados acadêmicos.
2.  **Otimização de Modelos:** Emprego de diversas arquiteturas de ML (como Redes Neurais e Árvores de Decisão) com otimização de hiperparâmetros para identificar as abordagens de melhor desempenho.
3.  **Interpretabilidade (XAI):** Análise de importância de *features* (variáveis exógenas) utilizando **valores SHAP** para entender o impacto de cada variável (climática, acadêmica) nas previsões e reduzir a complexidade do modelo.
4.  **Modelo Cooperativo (WSB):** Avaliação de um novo modelo de *Cooperative Ensemble Learning* (o WSB) em um cenário *multicampi* complexo, expandindo sua aplicação original.
5.  **Regressão vs. Classificação:** Uma análise comparativa que converte o problema tradicional de regressão de série temporal em um problema de **classificação baseado em intervalos de previsão**.
6.  **Treinamento Local vs. Global:** Investigação sobre o impacto de agregar dados de múltiplos *campi* (treinamento **Global**) em comparação com o treinamento de modelos isolados para cada *campus* (treinamento **Local**).

## 📁 Estrutura do Repositório

Para navegar pelos experimentos, utilize a seguinte estrutura de pastas:
```
📁 /ppgeps_dissertacao
├── 📁 dados/ (Armazena os conjuntos de dados utilizados)
├── 📁 resultados/ (Contém os resultados exportados dos experimentos) 
├── 📓 01-Criação dos Datasets.ipynb 
├── 📓 02-Análise dos Dados.ipynb 
├── 📜 02-Teste de Hipoteses.R (Scripts para testes estatísticos em R)
├── 📓 03-Pre-Processamento.ipynb 
├── 📓 04-Otimização - Classificação.ipynb 
├── 📓 04-Otimização - Regressão.ipynb 
├── 📓 05-Resultados - Classificação.ipynb 
├── 📓 05-Resultados - Regressão.ipynb 
├── 📓 05-Resultados - SHAP.ipynb  
├── 🐍 COA.py (Implementação do algoritmo Coyote Optimization Algorithm) 
├── 🐍 anneal.py (Implementação do Simulated Annealing) 
├── 🐍 pyESN.py (Implementação de Echo State Networks) 
├── 🐍 wsb.py (Implementação do modelo WSB)  
└── 📄 README (Este arquivo) 
```

## 🚀 Tecnologias Utilizadas

* **Linguagem:** Python
* **Bibliotecas Principais:**
    * Pandas (Manipulação de dados)
    * Scikit-learn (Modelos de ML tradicionais)
    * SHAP (Interpretabilidade dos modelos)
    * Matplotlib / Seaborn (Visualização de dados)
    * Jupyter (Ambiente de experimentação)

## ⚡ Como Executar os Experimentos

1.  Clone este repositório:
    ```bash
    git clone https://github.com/eduardoalba0/ppgeps_dissertacao.git
    cd ppgeps_dissertacao
    ```

2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3.  Abra os notebooks na pasta principal utilizando o Jupyter:
    ```bash
    jupyter notebook
    ```

## 👨‍💻 Autor

* **Eduardo Luiz Alba**
* GitHub: [@eduardoalba0](https://github.com/eduardoalba0)
* Lattes: http://lattes.cnpq.br/8649588576930512
* Linkedin: https://www.linkedin.com/in/eduardo-luiz-alba-ab373a166/
* ResearchGate: https://www.researchgate.net/profile/Eduardo-Alba
* Orientador: Dr. Érick Oliveira Rodrigues
* Coorientador: Dr. Matheus Henrique Dal Molin Ribeiro

## 📄 Licença

Este trabalho e os materiais associados estão licenciados sob a [Creative Commons Atribuição–NãoComercial–SemDerivações 4.0 Internacional (CC BY-NC-ND 4.0)](https://https://creativecommons.org/licenses/by-nc-nd/4.0/).

![CC BY-NC-ND 4.0](https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png)