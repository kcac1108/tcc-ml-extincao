# TCC ML — Extinção

Classificação binária do risco de extinção de espécies de roedores (ordem Rodentia): a partir de dados de ocorrência do GBIF e de dados de avaliação da Lista Vermelha da IUCN, prevemos se uma espécie está **Ameaçada** (Vulnerable, Endangered, Critically Endangered) ou **Não Ameaçada** (Least Concern, Near Threatened).

**Autores:** João Avila Harduin (00332195), Kauã Cordeiro Cavalheiro (00331660), Luan Bonasorte Capella (00332061)
**Professora orientadora:** Edith Ranzini

## Estrutura

```
├── notebooks/
│   ├── 01_construcao_dataset.ipynb   Seções 1–5: aquisição, análise exploratória, construção do dataset, pré-processamento
│   ├── 02_modelagem.ipynb            Seções 6–10: Random Forest, SVM, XGBoost, SMOTE × SMOTENC, ajuste de hiperparâmetros
│   └── 03_fronteira_e_limpeza.ipynb  Seções 11–13: diagnóstico de fronteira, limpeza (ENN e Passo 1), Teste 4
├── src/tcc_extincao/                 código compartilhado pelos notebooks
│   ├── config.py                     caminhos, constantes (RANDOM_STATE etc.) e grades de hiperparâmetros
│   ├── dados.py                      download, leitura, rótulos, agregação do GBIF e junção das fontes
│   ├── preprocessamento.py           encoding, split treino/teste, escalonamento
│   ├── modelos.py                    treino/avaliação, GridSearchCV, trade-off de features
│   ├── avaliacao.py                  tabela de resultados, importância de features, diagnósticos
│   └── graficos.py                   gráficos usados em mais de um ponto
├── resultados/
│   ├── resultados_modelagem.csv      tabela consolidada do notebook 02 (lida pelo notebook 03)
│   └── buscas/                       melhores hiperparâmetros de cada GridSearchCV
└── requirements.txt
```

Os notebooks devem ser lidos em ordem, mas cada um roda sozinho: o 02 e o 03 reconstroem o dataset e o mesmo split treino/teste do 01 (`RANDOM_STATE = 42`). O 03 usa a tabela de resultados gerada pelo 02, que fica versionada em `resultados/`.

## Dados

Os arquivos de origem (`Rodentia_gbif.csv`, `assessments.csv`, `taxonomy.csv`) não ficam no repositório. Os notebooks baixam esses arquivos do Google Drive com `gdown` na primeira execução e os salvam em `data/`. Os datasets gerados são salvos em `data/final_data/`.

- **GBIF** (Global Biodiversity Information Facility): registros de ocorrência de roedores.
- **IUCN Red List**: categorias de ameaça (`assessments.csv`) e taxonomia (`taxonomy.csv`).

## Como rodar

### No Google Colab

Abra o notebook pelo link abaixo e execute as células em ordem. A primeira célula clona este repositório e instala o `matplotlib-venn`. As demais bibliotecas já vêm no Colab.

- [01 — Construção do dataset](https://colab.research.google.com/github/kcac1108/tcc-ml-extincao/blob/main/notebooks/01_construcao_dataset.ipynb)
- [02 — Modelagem](https://colab.research.google.com/github/kcac1108/tcc-ml-extincao/blob/main/notebooks/02_modelagem.ipynb)
- [03 — Fronteira e limpeza](https://colab.research.google.com/github/kcac1108/tcc-ml-extincao/blob/main/notebooks/03_fronteira_e_limpeza.ipynb)

### Localmente

```bash
git clone https://github.com/kcac1108/tcc-ml-extincao.git
cd tcc-ml-extincao
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

### Tempo de execução

Os `GridSearchCV` são a parte demorada. Como o balanceamento (SMOTE/SMOTENC/ENN) é refeito dentro de cada fold da validação cruzada, a busca completa leva cerca de 30 minutos numa máquina com 8 núcleos e de 5 a 6 horas no Colab gratuito (2 núcleos).

Por padrão, os notebooks **reaproveitam** os melhores hiperparâmetros salvos em `resultados/buscas/` e treinam só o modelo final, que é o mesmo modelo escolhido pela busca. Assim, cada notebook roda em poucos minutos. Para refazer as buscas, descomente `config.REFAZER_BUSCAS = True` na célula de imports do notebook (ou defina a variável de ambiente `TCC_REFAZER_BUSCAS=1`).

## Correções metodológicas (nov/2026)

Correções aplicadas depois da primeira versão dos experimentos (os resultados numéricos mudaram em relação à versão anterior):

1. **Imputação depois do split** (Seção 5.3): os valores ausentes eram preenchidos com a mediana do dataset inteiro, antes da divisão treino/teste. Agora a mediana é calculada só no treino.
2. **SMOTE sem `redlistCategory`** (Seção 6.2): as amostras sintéticas dos experimentos sem vazamento eram geradas com `redlistCategory` participando do cálculo dos vizinhos. Agora o SMOTE é refeito sem a coluna.
3. **Predições sobrescritas no trade-off** (Seção 6.7): o loop do trade-off sobrescrevia `y_pred_rf` e `y_pred_svm`, e a tabela consolidada registrava os modelos com apenas 5 features. Agora o loop usa variáveis próprias.
4. **Balanceamento dentro da validação cruzada** (Seções 7, 8.4, 9.2 e 13): o `GridSearchCV` rodava sobre dados já balanceados, então amostras sintéticas criadas a partir de um fold apareciam na validação de outro, inflando o F1 da validação cruzada. Agora SMOTE/SMOTENC, ENN e `StandardScaler` ficam dentro de um `Pipeline` do imblearn e são refeitos em cada fold.
5. **SVM padrão escalonado** (Seções 8.3, 12.1 e 12.2): os SVMs sem tuning com SMOTENC não eram escalonados, enquanto os otimizados eram. A comparação padrão × otimizado misturava o efeito do tuning com o do escalonamento.
6. **Diagnóstico de fronteira sem `redlistCategory`** (Seção 11): o Random Forest do Passo 1 usava `redlistCategory`, da qual o rótulo é derivado. A lista de espécies suspeitas (e o Teste 3) depende dessa correção.
7. **Rótulo do SVM otimizado com SMOTENC** (Seção 10): estava registrado como `SMOTENC (Escalado)`, o que quebrava os gráficos da Seção 13.4.
