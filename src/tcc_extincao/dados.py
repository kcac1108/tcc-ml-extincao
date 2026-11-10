"""Aquisição das três fontes (GBIF, IUCN assessments, taxonomia) e construção do dataset."""
import gdown
import pandas as pd

from . import config


def baixar_dados(forcar=False):
    """Baixa os CSVs do Google Drive para data/ (pula os que já existem, a menos que forcar=True)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.FINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for file_id, output_path in config.ARQUIVOS_DRIVE.items():
        if output_path.exists() and not forcar:
            print(f'{output_path.name} já existe, download pulado.')
            continue
        print(f'Downloading {output_path.name}...')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', str(output_path), quiet=True)


def ler_fontes():
    """Retorna (gbif, assessments, taxonomy)."""
    gbif        = pd.read_csv(config.GBIF_PATH, sep='\t', low_memory=False)
    assessments = pd.read_csv(config.ASSESSMENTS_PATH)
    taxonomy    = pd.read_csv(config.TAXONOMY_PATH)
    return gbif, assessments, taxonomy


def criar_rotulos(assessments):
    """Mantém só as categorias da classificação binária e cria a coluna 'label'."""
    assess_labeled = assessments[
        assessments['redlistCategory'].isin(config.INCLUDE_CATS)
    ].copy()
    assess_labeled['label'] = assess_labeled['redlistCategory'].map(config.INCLUDE_CATS)
    return assess_labeled


def agregar_gbif(df):
    """Agrega as ocorrências do GBIF em uma linha por espécie, com features numéricas."""
    df = df.dropna(subset=['species'])
    feats = df.groupby('species').agg(
        n_occurrences          = ('gbifID',                        'count'),
        n_countries            = ('countryCode',                   'nunique'),
        lat_mean               = ('decimalLatitude',               'mean'),
        lat_std                = ('decimalLatitude',               'std'),
        lon_mean               = ('decimalLongitude',              'mean'),
        lon_std                = ('decimalLongitude',              'std'),
        lat_min                = ('decimalLatitude',               'min'),
        lat_max                = ('decimalLatitude',               'max'),
        lon_min                = ('decimalLongitude',              'min'),
        lon_max                = ('decimalLongitude',              'max'),
        year_min               = ('year',                          'min'),
        year_max               = ('year',                          'max'),
        n_months_active        = ('month',                         'nunique'),
        n_basis_types          = ('basisOfRecord',                 'nunique'),
        coord_uncertainty_mean = ('coordinateUncertaintyInMeters', 'mean'),
    ).reset_index()

    # Features derivadas
    feats['lat_range']  = feats['lat_max'] - feats['lat_min']
    feats['lon_range']  = feats['lon_max'] - feats['lon_min']
    feats['year_range'] = feats['year_max'] - feats['year_min']

    # Proporção por tipo de registro
    basis_counts = df.groupby(['species', 'basisOfRecord']).size().unstack(fill_value=0)
    totals = basis_counts.sum(axis=1)
    for col, alias in [('HUMAN_OBSERVATION', 'pct_human_obs'),
                       ('PRESERVED_SPECIMEN', 'pct_specimen')]:
        if col in basis_counts.columns:
            serie = (basis_counts[col] / totals).rename(alias).reset_index()
            feats = feats.merge(serie, on='species', how='left')
        else:
            feats[alias] = 0.0

    return feats


def juntar_fontes(assess_labeled, taxonomy, gbif_features):
    """assessments + taxonomy (left) + features do GBIF (inner), pela chave scientificName."""
    assess_cols = [
        'scientificName', 'redlistCategory', 'label',
        'populationTrend', 'yearPublished', 'systems',
        'realm', 'possiblyExtinct',
    ]
    tax_cols = ['scientificName', 'familyName', 'genusName']

    # Join 1: assessments + taxonomy (left join — taxonomy é subconjunto)
    redlist_base = assess_labeled[assess_cols].merge(taxonomy[tax_cols], on='scientificName', how='left')

    # Join 2: redlist_base + gbif_features (inner — mantém só espécies com dados em ambas)
    return redlist_base.merge(
        gbif_features.rename(columns={'species': 'scientificName'}),
        on='scientificName',
        how='inner'
    )


def exportar_dataset(X, y, nome_arquivo):
    """Salva X + coluna 'label' em data/final_data/."""
    df_export = X.copy()
    df_export['label'] = y.values if hasattr(y, 'values') else y
    caminho = config.FINAL_DATA_DIR / nome_arquivo
    df_export.to_csv(caminho, index=False)
    print(f'Exportado: {caminho}  (shape: {df_export.shape})')


def construir_dataset():
    """Baixa (se preciso) as três fontes e monta o dataset unificado, uma linha por espécie."""
    baixar_dados()
    gbif, assessments, taxonomy = ler_fontes()
    dataset = juntar_fontes(criar_rotulos(assessments), taxonomy, agregar_gbif(gbif))
    print(f'Dataset: {dataset.shape[0]} espécies x {dataset.shape[1]} colunas')
    return dataset
