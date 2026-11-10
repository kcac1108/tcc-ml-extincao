"""Gráficos reutilizados em mais de um ponto dos notebooks."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import config

# Paleta categórica validada (skill de dataviz): slots 1, 2 e 3 (azul, laranja, verde-água)
CORES_BALANCEAMENTO = {
    'SMOTENC': '#2a78d6',
    'SMOTENC + ENN': '#eb6834',
    'SMOTENC + Passo1': '#1baf7a',
}
TITULOS_METRICAS = {'recall': 'Recall (classe "Threatened")', 'f1': 'F1-score (classe "Threatened")'}


def plot_importancias(importancias, titulo, xlabel, palette='viridis', n=20):
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', hue='feature', data=importancias.head(n), palette=palette, legend=False)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_tradeoff(resultados, titulo, cor_tempo, cor_acuracia, eixo_categorico=False):
    """Tempo de treinamento (eixo esquerdo) e acurácia (eixo direito) por número de features.

    eixo_categorico=True usa a posição de cada ponto no eixo X (evita a distorção
    do espaçamento desigual) e mostra o desvio-padrão das repetições.
    """
    fig, ax1 = plt.subplots(figsize=(11, 6) if eixo_categorico else (10, 6))
    ax2 = ax1.twinx()  # segundo eixo Y

    if eixo_categorico:
        x = list(range(len(resultados)))
        ax1.set_xticks(x)
        ax1.set_xticklabels(resultados['n_features'], rotation=45)
        ax1.set_xlabel('Quantidade de Features (amostradas, não linear)')
    else:
        x = resultados['n_features']
        ax1.set_xlabel('Quantidade de Features (em ordem decrescente)')

    ax1.plot(x, resultados['training_time'], 'o-', color=cor_tempo, label='Tempo de Treinamento (s)')
    ax1.set_ylabel('Tempo de Treinamento (s)', color=cor_tempo)
    ax1.tick_params(axis='y', labelcolor=cor_tempo)
    ax1.set_title(titulo)
    ax1.grid(True, linestyle='--', alpha=0.7)

    if eixo_categorico:
        ax2.errorbar(x, resultados['accuracy'], yerr=resultados['accuracy_std'], fmt='x-', color=cor_acuracia,
                     label='Precisão (média ± desvio, 3 execuções)', capsize=3)
    else:
        ax2.plot(x, resultados['accuracy'], 'x-', color=cor_acuracia, label='Precisão')
    ax2.set_ylabel('Precisão', color=cor_acuracia)
    ax2.tick_params(axis='y', labelcolor=cor_acuracia)

    linhas1, rotulos1 = ax1.get_legend_handles_labels()
    linhas2, rotulos2 = ax2.get_legend_handles_labels()
    ax2.legend(linhas1 + linhas2, rotulos1 + rotulos2, loc='upper right')

    plt.tight_layout()
    plt.show()


def _anotar_barras(ax, barras, **kwargs):
    for barra in barras:
        altura = barra.get_height()
        ax.annotate(f'{altura:.2f}', xy=(barra.get_x() + barra.get_width() / 2, altura),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, **kwargs)


def _estilo_eixo(ax, titulo, **kwargs):
    ax.set_xticks(np.arange(len(config.ORDEM_MODELO)))
    ax.set_xticklabels(config.ORDEM_MODELO)
    ax.set_title(titulo, **kwargs)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _valores(comparacao, balanceamento, metrica):
    return [
        comparacao.loc[(comparacao['modelo'] == m) & (comparacao['balanceamento'] == balanceamento), metrica].values[0]
        for m in config.ORDEM_MODELO
    ]


def plot_comparacao_cenarios(comparacao, titulo, metricas=('recall', 'f1')):
    """Barras agrupadas por modelo, uma cor por cenário de balanceamento/limpeza."""
    fig, axes = plt.subplots(1, len(metricas), figsize=(12, 5.5), sharey=True)
    largura = 0.26
    posicoes = np.arange(len(config.ORDEM_MODELO))
    offsets = [-largura, 0, largura]

    for ax, metrica in zip(axes, metricas):
        for balanceamento, offset in zip(config.ORDEM_BALANCEAMENTO, offsets):
            barras = ax.bar(posicoes + offset, _valores(comparacao, balanceamento, metrica), largura,
                            label=balanceamento, color=CORES_BALANCEAMENTO[balanceamento])
            _anotar_barras(ax, barras)
        _estilo_eixo(ax, TITULOS_METRICAS[metrica])

    axes[0].set_ylabel('Score')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle(titulo, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_efeito_tuning(comparacao_default, comparacao_tuning):
    """F1 default × otimizado, um painel por cenário."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    largura = 0.35
    posicoes = np.arange(len(config.ORDEM_MODELO))

    for ax, bal in zip(axes, config.ORDEM_BALANCEAMENTO):
        barras_def = ax.bar(posicoes - largura / 2, _valores(comparacao_default, bal, 'f1'), largura,
                            label='Default (Sem Tuning)', color='#8faadc')
        barras_oti = ax.bar(posicoes + largura / 2, _valores(comparacao_tuning, bal, 'f1'), largura,
                            label='Otimizado (Com Tuning)', color='#2b5c8f')
        _anotar_barras(ax, barras_def)
        _anotar_barras(ax, barras_oti, fontweight='bold')
        _estilo_eixo(ax, bal, fontsize=11, fontweight='semibold')

    axes[0].set_ylabel('F1-score (classe "Threatened")')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle('Impacto do Ajuste de Hiperparâmetros (F1-score): Default vs. Otimizado por Cenário', fontsize=12)
    plt.tight_layout()
    plt.show()
