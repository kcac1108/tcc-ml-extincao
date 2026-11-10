"""Encoding, divisão treino/teste e escalonamento."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


def codificar(dataset):
    """Transforma o dataset unificado em features numéricas.

    Retorna (df, colunas_systems): df com a coluna 'label' (0/1) e as colunas
    multi-label criadas a partir de 'systems'.
    """
    # remover colunas de index
    df = dataset.drop(columns=['scientificName'])

    # mapeando labels binárias
    df['possiblyExtinct'] = df['possiblyExtinct'].astype(str).str.lower().str.strip().map({'false': 0, 'true': 1})
    df['label'] = df['label'].map({'Not Threatened': 0, 'Threatened': 1})

    # mapeando labels ordinais (redlist category)
    df['redlistCategory'] = df['redlistCategory'].map(config.MAPEAMENTO_RISCO)

    # mapeamento multi-label (systems)
    sistemas_separados = df['systems'].str.get_dummies(sep='|')
    df = pd.concat([df, sistemas_separados], axis=1)
    df = df.drop(columns=['systems'])

    # one-hot encoding
    df = pd.get_dummies(df, columns=config.COLUNAS_ONEHOT, dtype=int)

    return df, list(sistemas_separados.columns)


def separar_X_y(df):
    return df.drop(columns=['label']), df['label']


def dividir_treino_teste(X, y):
    """Split estratificado; os vazios são preenchidos com a mediana calculada só no treino."""
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    medianas_treino = X_treino.median()
    return X_treino.fillna(medianas_treino), X_teste.fillna(medianas_treino), y_treino, y_teste


def sem_redlist(X):
    """Remove redlistCategory, da qual o rótulo é derivado (data leakage)."""
    return X.drop(columns=['redlistCategory'])


def colunas_categoricas(X, colunas_systems):
    """Colunas binárias/one-hot de X e suas posições (formato exigido pelo SMOTENC)."""
    colunas_dummies = [c for c in X.columns if c.startswith(config.PREFIXOS_CATEGORICOS)]
    colunas = colunas_dummies + [c for c in colunas_systems if c in X.columns] + ['possiblyExtinct']
    return colunas, [X.columns.get_loc(c) for c in colunas]


def escalonar(X_treino, X_teste):
    """StandardScaler ajustado no treino; devolve DataFrames com as mesmas colunas e índices."""
    scaler = StandardScaler()
    X_treino_scaled = pd.DataFrame(scaler.fit_transform(X_treino), columns=X_treino.columns, index=X_treino.index)
    X_teste_scaled = pd.DataFrame(scaler.transform(X_teste), columns=X_teste.columns, index=X_teste.index)
    return X_treino_scaled, X_teste_scaled
