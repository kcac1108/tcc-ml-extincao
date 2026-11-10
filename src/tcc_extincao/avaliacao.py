"""Métricas consolidadas, importância de features e diagnósticos."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from . import config


def registrar_resultado(lista, nome_modelo, balanceamento, tuning, y_true, y_pred, tempo_treino=None):
    """
    Calcula as métricas de classificação (classe positiva = 1, 'Ameaçada')
    e adiciona uma linha à lista de resultados, para consolidação posterior
    numa tabela comparativa entre modelos.
    """
    lista.append({
        'modelo': nome_modelo,
        'balanceamento': balanceamento,
        'tuning': 'Sim' if tuning else 'Não',
        'acuracia': accuracy_score(y_true, y_pred),
        'precisao': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'tempo_treino_s': tempo_treino,
    })


def tabela_resultados(resultados):
    """Lista de resultados -> DataFrame ordenado por F1 (maior primeiro)."""
    return pd.DataFrame(resultados).sort_values(by='f1', ascending=False).reset_index(drop=True)


def filtrar_cenarios(df_resultados, tuning):
    """Linhas dos 3 cenários SMOTENC / + ENN / + Passo1, ordenadas por modelo e cenário."""
    comparacao = df_resultados[
        (df_resultados['tuning'] == tuning) &
        (df_resultados['balanceamento'].isin(config.ORDEM_BALANCEAMENTO))
    ].copy()
    comparacao['modelo'] = pd.Categorical(comparacao['modelo'], categories=config.ORDEM_MODELO, ordered=True)
    comparacao['balanceamento'] = pd.Categorical(comparacao['balanceamento'], categories=config.ORDEM_BALANCEAMENTO, ordered=True)
    return comparacao.sort_values(['modelo', 'balanceamento']).reset_index(drop=True)


def importancia_gini(modelo, colunas):
    """feature_importances_ de um modelo de árvores, ordenado do maior para o menor."""
    return pd.DataFrame({
        'feature': colunas,
        'importance': modelo.feature_importances_,
    }).sort_values(by='importance', ascending=False)


def importancia_permutacao(modelo, X, y):
    """Permutation importance (queda média da acurácia ao embaralhar cada feature)."""
    resultado = permutation_importance(modelo, X, y, n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1)
    return pd.DataFrame({
        'feature': X.columns,
        'importance': resultado.importances_mean,
        'importance_std': resultado.importances_std,
    }).sort_values(by='importance', ascending=False)


def pares_alta_correlacao(df, colunas, limite=0.85):
    """Pares (linha, coluna, r) de colunas com |r| acima do limite."""
    corr_matrix = df[colunas].corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    return [
        (row, col, upper_tri.loc[row, col])
        for col in upper_tri.columns
        for row in upper_tri.index
        if abs(upper_tri.loc[row, col]) > limite
    ]


def diagnostico_fracionario(X_sinteticas, colunas_categoricas):
    """% de valores fora de {0, 1} gerados nas colunas categóricas das amostras sintéticas.

    Retorna (df_diagnostico, % médio por coluna, % de amostras com algum valor fracionário).
    """
    resultados = []
    for col in colunas_categoricas:
        valores = X_sinteticas[col]
        fracao_invalida = (~np.isclose(valores, 0) & ~np.isclose(valores, 1)).mean()
        resultados.append({'coluna': col, 'pct_valores_fracionarios': fracao_invalida * 100})

    df_diagnostico = pd.DataFrame(resultados).sort_values(by='pct_valores_fracionarios', ascending=False)
    linhas_com_algum_valor_invalido = (
        ~np.isclose(X_sinteticas[colunas_categoricas], 0) &
        ~np.isclose(X_sinteticas[colunas_categoricas], 1)
    ).any(axis=1).mean()
    return df_diagnostico, df_diagnostico['pct_valores_fracionarios'].mean(), linhas_com_algum_valor_invalido * 100


def rastrear_fronteira(X, y, nomes, n_repeticoes=20, n_splits=5):
    """Taxa de erro de cada espécie "Not Threatened" em validações cruzadas repetidas.

    Em cada fold: imputação pela mediana do treino do fold, undersampling só no
    treino e um Random Forest. X não deve conter redlistCategory.
    """
    rastreio = pd.DataFrame({'scientificName': nomes, 'label': y.values}, index=X.index)
    rastreio['vezes_no_teste'] = 0
    rastreio['vezes_errado'] = 0

    for seed in range(n_repeticoes):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (idx_treino, idx_teste) in enumerate(skf.split(X, y)):
            X_treino_fold, y_treino_fold = X.iloc[idx_treino], y.iloc[idx_treino]
            X_teste_fold,  y_teste_fold  = X.iloc[idx_teste],  y.iloc[idx_teste]

            # Vazios preenchidos com a mediana do treino do fold
            medianas_fold = X_treino_fold.median()
            X_treino_fold = X_treino_fold.fillna(medianas_fold)
            X_teste_fold = X_teste_fold.fillna(medianas_fold)

            # Undersampling só no treino: reduz "Not Threatened" (0) ao tamanho de "Threatened" (1)
            n_minoritaria = y_treino_fold.value_counts().min()
            idx_maj_amostrado = y_treino_fold[y_treino_fold == 0].sample(
                n=n_minoritaria, random_state=seed * 100 + fold_idx
            ).index
            idx_min = y_treino_fold[y_treino_fold == 1].index
            idx_treino_balanceado = idx_maj_amostrado.union(idx_min)

            rf = RandomForestClassifier(random_state=config.RANDOM_STATE)
            rf.fit(X_treino_fold.loc[idx_treino_balanceado], y_treino_fold.loc[idx_treino_balanceado])

            y_pred_fold = pd.Series(rf.predict(X_teste_fold), index=X_teste_fold.index)

            # Só nos interessa avaliar as espécies "Not Threatened" (0) que caíram no teste
            idx_maj_no_teste = y_teste_fold[y_teste_fold == 0].index
            rastreio.loc[idx_maj_no_teste, 'vezes_no_teste'] += 1

            erradas = idx_maj_no_teste[y_pred_fold.loc[idx_maj_no_teste] != 0]
            rastreio.loc[erradas, 'vezes_errado'] += 1

    rastreio['taxa_erro'] = rastreio['vezes_errado'] / rastreio['vezes_no_teste']
    return rastreio
