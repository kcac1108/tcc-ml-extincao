"""Treino, avaliação e busca de hiperparâmetros."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from . import config


def imprimir_avaliacao(y_teste, y_pred, nome):
    print(f'\nRelatório de Classificação {nome}:')
    print(classification_report(y_teste, y_pred))
    print(f'Matriz de Confusão {nome}:\n', confusion_matrix(y_teste, y_pred))
    print(f'Acurácia {nome}: {accuracy_score(y_teste, y_pred):.4f}')


def treinar_e_avaliar(modelo, X_treino, y_treino, X_teste, y_teste, nome):
    """Treina, mede o tempo, avalia no teste e imprime o relatório. Retorna (y_pred, tempo)."""
    print(f'### {nome} ###')
    inicio = time.time()
    modelo.fit(X_treino, y_treino)
    tempo = time.time() - inicio
    y_pred = modelo.predict(X_teste)

    print(f'\nTempo de treinamento: {tempo:.4f} segundos')
    imprimir_avaliacao(y_teste, y_pred, nome)
    return y_pred, tempo


def grade_pipeline(grade):
    """Prefixa os hiperparâmetros com 'modelo__' (nome do estimador final nos Pipelines)."""
    return {f'modelo__{k}': v for k, v in grade.items()}


def rodar_grid(nome_busca, estimador, grade, X_treino, y_treino, X_teste, y_teste, nome):
    """GridSearchCV (cv=CV_FOLDS, scoring=SCORING) seguido da avaliação no teste.

    O resultado da busca fica salvo em resultados/buscas/<nome_busca>.json. Se o
    arquivo já existe e config.REFAZER_BUSCAS é False, a busca não é refeita:
    o estimador é treinado com os melhores hiperparâmetros salvos, o que dá o
    mesmo modelo que o best_estimator_ da busca.

    Retorna (melhor_modelo, y_pred, tempo_busca).
    """
    print(f'### GridSearchCV — {nome} ###')
    caminho = config.BUSCAS_DIR / f'{nome_busca}.json'

    if caminho.exists() and not config.REFAZER_BUSCAS:
        salvo = json.loads(caminho.read_text(encoding='utf-8'))
        melhor = clone(estimador).set_params(**salvo['best_params'])
        melhor.fit(X_treino, y_treino)
        print(f'Busca reaproveitada de resultados/buscas/{caminho.name} '
              '(para refazer, defina config.REFAZER_BUSCAS = True)')
    else:
        grid = GridSearchCV(estimador, grade, cv=config.CV_FOLDS, scoring=config.SCORING, n_jobs=-1, verbose=1)
        inicio = time.time()
        grid.fit(X_treino, y_treino)
        salvo = {
            'best_params': grid.best_params_,
            'best_score': float(grid.best_score_),
            'tempo_busca_s': time.time() - inicio,
        }
        config.BUSCAS_DIR.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(salvo, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        melhor = grid.best_estimator_

    print(f"\nTempo total de busca: {salvo['tempo_busca_s']:.4f} segundos")
    print(f"Melhores hiperparâmetros encontrados: {salvo['best_params']}")
    print(f"Melhor score (F1, média da validação cruzada): {salvo['best_score']:.4f}")

    y_pred = melhor.predict(X_teste)
    imprimir_avaliacao(y_teste, y_pred, f'{nome} (Otimizado)')
    return melhor, y_pred, salvo['tempo_busca_s']


def avaliar_tradeoff(ranking, X_treino, y_treino, X_teste, y_teste, criar_modelo, contagens, n_repeticoes=1):
    """Treina com as N features mais importantes do ranking, para cada N em contagens.

    criar_modelo(seed) devolve um modelo novo; cada N é repetido n_repeticoes
    vezes (seeds RANDOM_STATE, RANDOM_STATE + 1, ...) e guardamos a média.
    """
    resultados = []
    for n_features in contagens:
        colunas = ranking[:n_features]
        tempos, acuracias = [], []
        for rep in range(n_repeticoes):
            modelo = criar_modelo(config.RANDOM_STATE + rep)
            inicio = time.time()
            modelo.fit(X_treino[colunas], y_treino)
            tempos.append(time.time() - inicio)
            acuracias.append(accuracy_score(y_teste, modelo.predict(X_teste[colunas])))
        resultados.append({
            'n_features': n_features,
            'training_time': np.mean(tempos),
            'accuracy': np.mean(acuracias),
            'accuracy_std': np.std(acuracias),
        })
    return pd.DataFrame(resultados).sort_values(by='n_features', ascending=False).reset_index(drop=True)
