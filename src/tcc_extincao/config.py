"""Caminhos, constantes e grades de hiperparâmetros compartilhados pelos notebooks."""
import os
from pathlib import Path

# ---------------------------------------------------------------- Caminhos
RAIZ           = Path(__file__).resolve().parents[2]
DATA_DIR       = RAIZ / 'data'
FINAL_DATA_DIR = DATA_DIR / 'final_data'
RESULTADOS_DIR = RAIZ / 'resultados'
BUSCAS_DIR     = RESULTADOS_DIR / 'buscas'

GBIF_PATH        = DATA_DIR / 'Rodentia_gbif.csv'
ASSESSMENTS_PATH = DATA_DIR / 'assessments.csv'
TAXONOMY_PATH    = DATA_DIR / 'taxonomy.csv'
OUTPUT_PATH      = FINAL_DATA_DIR / 'rodentia_dataset.csv'
ENCODED_PATH     = FINAL_DATA_DIR / 'rodentia_dataset_encoded.csv'

# Tabela de resultados do notebook 02, lida pelo notebook 03
RESULTADOS_MODELAGEM_PATH = RESULTADOS_DIR / 'resultados_modelagem.csv'

# IDs dos arquivos no Google Drive -> caminho local
ARQUIVOS_DRIVE = {
    '102t2404-T5dPg6CzJB1zZeS_vvCFwXHU': ASSESSMENTS_PATH,
    '1ak38MaCPL8h9OqMqI2o0TN9MUR8TqMBE': GBIF_PATH,
    '1cc6Rmx9HRB1Fh8CtvbiRXYsjPL9VVZPC': TAXONOMY_PATH,
}

# ---------------------------------------------------------------- Experimento
# Semente única usada em todos os splits, reamostragens e modelos
RANDOM_STATE = 42
TEST_SIZE    = 0.2   # 20% dos dados reservados para teste
CV_FOLDS     = 5     # folds da validação cruzada no GridSearchCV
# F1 prioriza o equilíbrio entre precisão e recall na identificação de
# espécies ameaçadas, e não apenas a acurácia geral
SCORING      = 'f1'

# Os GridSearchCV são a parte demorada (horas no Colab gratuito). Com False, os
# melhores hiperparâmetros salvos em resultados/buscas/ são reaproveitados e só o
# modelo final é treinado. Com True (ou TCC_REFAZER_BUSCAS=1), as buscas são refeitas.
REFAZER_BUSCAS = os.environ.get('TCC_REFAZER_BUSCAS') == '1'

# Grades de hiperparâmetros usadas em todos os GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf']
}

# learning_rate e n_estimators controlam o trade-off entre número de
# iterações de boosting e o tamanho do passo de cada uma; subsample e
# colsample_bytree introduzem aleatoriedade (semelhante ao bagging do RF)
# para reduzir overfitting, algo especialmente relevante dado o tamanho
# modesto do dataset
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# ---------------------------------------------------------------- Rótulos e encoding
INCLUDE_CATS = {
    'Least Concern'        : 'Not Threatened',
    'Near Threatened'      : 'Not Threatened',
    'Vulnerable'           : 'Threatened',
    'Endangered'           : 'Threatened',
    'Critically Endangered': 'Threatened',
}

# Escala ordinal da redlistCategory (hierarquia de risco da IUCN)
MAPEAMENTO_RISCO = {
    'Least Concern': 0,
    'Near Threatened': 1,
    'Vulnerable': 2,
    'Endangered': 3,
    'Critically Endangered': 4,
    'Extinct in the Wild': 5,
    'Extinct': 6
}

COLUNAS_ONEHOT = ['populationTrend', 'realm', 'familyName', 'genusName']
PREFIXOS_CATEGORICOS = ('populationTrend_', 'realm_', 'familyName_', 'genusName_')

# ---------------------------------------------------------------- Comparações
ORDEM_MODELO = ['Random Forest', 'SVM', 'XGBoost']
ORDEM_BALANCEAMENTO = ['SMOTENC', 'SMOTENC + ENN', 'SMOTENC + Passo1']
