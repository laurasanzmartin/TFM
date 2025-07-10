import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# Configuraciones generales para visualizaciones - COLORES MODIFICADOS
plt.style.use('seaborn-v0_8-whitegrid')

# Cambiar a una paleta de colores más diferenciable
# Opciones de paletas con colores más contrastantes:
# 'tab10', 'Set1', 'Dark2', 'Paired', 'Set3' son buenas opciones con colores bien diferenciados

# Definimos una paleta personalizada con colores altamente diferenciables
custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                 '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

# Aplicamos nuestra paleta personalizada
sns.set_palette(custom_palette)

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Cargar los datos
file_path = r"c:\Users\laura\Downloads\bbdd_eu50.xlsx"  
df = pd.read_excel(file_path)


print("="*80)
print("SECCIÓN 0: ANÁLISIS DE VALORES PERDIDOS")
print("="*80)

# Mostrar resumen de valores nulos por columna
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df) * 100).round(2)
missing_summary = pd.DataFrame({'missing_count': missing_counts, 'missing_pct': missing_percent})
missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values(by='missing_pct', ascending=False)

print("\nVariables con valores faltantes:")
print(missing_summary)

# Visualización patrón valores perdidos
plt.figure(figsize=(14,6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Mapa de calor de valores faltantes (NaNs)')
plt.show()



def simple_rf_imputation(df):
    df_imputed = df.copy()
    numeric_vars = df_imputed.select_dtypes(include=[np.number]).columns.tolist()

    # Imputación inicial rápida para predictores
    for col in numeric_vars:
        median_val = df_imputed[col].median()
        df_imputed[col].fillna(median_val, inplace=True)

    # Imputar con Random Forest solo los valores originalmente faltantes
    for var in numeric_vars:
        missing_mask = df[var].isnull()
        if missing_mask.sum() == 0:
            continue  # No imputar si no hay valores faltantes originales

        predictors = [col for col in numeric_vars if col != var]

        # Datos para entrenar
        X_train = df_imputed.loc[~missing_mask, predictors]
        y_train = df.loc[~missing_mask, var]

        # Datos para predecir
        X_pred = df_imputed.loc[missing_mask, predictors]

        # Entrenar y predecir
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)

        # Actualizar valores imputados
        df_imputed.loc[missing_mask, var] = y_pred
        print(f"Variable '{var}': imputados {missing_mask.sum()} valores")

    return df_imputed

# Definir variables numéricas (ajustar si hay columnas no deseadas)
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nEjecutando imputación iterativa con Random Forest para variables numéricas...")
df = simple_rf_imputation(df)

print(f"\nValores faltantes tras imputación: {df[numeric_vars].isnull().sum().sum()}")

df_original = pd.read_excel(file_path)
import math
# Visualización comparativa: distribución antes y después de imputar
variables_con_nan = [var for var in numeric_vars if df_original[var].isnull().any()]
batch_size = 6
num_batches = math.ceil(len(variables_con_nan) / batch_size)

for batch_num in range(num_batches):
    plt.figure(figsize=(18, 12))
    batch_vars = variables_con_nan[batch_num*batch_size : (batch_num+1)*batch_size]

    for i, var in enumerate(batch_vars, 1):
        plt.subplot(2, 3, i)
        original_data = df_original[var].dropna()
        imputed_data = df[var]

        plt.hist(original_data, bins=30, alpha=0.6, label='Original', density=True)
        plt.hist(imputed_data, bins=30, alpha=0.4, label='Imputado', density=True)
        plt.title(f'Distribución de {var}')
        plt.xlabel(var)
        plt.ylabel('Densidad')
        plt.legend()

    plt.tight_layout()
    plt.show()


# SECCIÓN 1: VISIÓN GENERAL Y LIMPIEZA DE DATOS
print("="*80)
print("SECCIÓN 1: VISIÓN GENERAL Y LIMPIEZA DE DATOS")
print("="*80)

# 1.1 Información básica del dataset
print("\n1.1 INFORMACIÓN BÁSICA DEL DATASET")
print(f"Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print("\nPrimeras 5 filas:")
print(df.head().to_string())

# 1.2 Verificar tipos de datos
print("\n1.2 TIPOS DE DATOS")
print(df.dtypes)

# 1.3 Verificar valores nulos
print("\n1.3 VALORES NULOS POR COLUMNA")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])
null_percentage = (df.isnull().sum() / len(df) * 100).round(2)
print("\nPorcentaje de valores nulos:")
print(null_percentage[null_percentage > 0])

# 1.4 Verificar valores duplicados
print(f"\n1.4 REGISTROS DUPLICADOS: {df.duplicated().sum()}")

# 1.5 Estadísticas descriptivas básicas
print("\n1.5 ESTADÍSTICAS DESCRIPTIVAS BÁSICAS")
desc_stats = df.describe(include='all').T
desc_stats['missing'] = df.isnull().sum()
desc_stats['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2)
print(desc_stats)

# SECCIÓN 2: ANÁLISIS DE LA COMPOSICIÓN DE LA MUESTRA
print("\n" + "="*80)
print("SECCIÓN 2: ANÁLISIS DE LA COMPOSICIÓN DE LA MUESTRA")
print("="*80)

# 2.1 Empresas en la muestra
print("\n2.1 EMPRESAS EN LA MUESTRA")
print(f"Número de empresas únicas: {df['identifier'].nunique()}")
print(f"Número de empresas por año:")
print(df.groupby('year')['identifier'].nunique())

# 2.2 Cobertura temporal
print("\n2.2 COBERTURA TEMPORAL")
print(f"Años disponibles: {sorted(df['year'].unique())}")
print(f"Número de años: {df['year'].nunique()}")
year_count = df['year'].value_counts().sort_index()
print("Observaciones por año:")
print(year_count)

# 2.3 Distribución sectorial
print("\n2.3 DISTRIBUCIÓN SECTORIAL")
sector_count = df['trbc_economic_sector_name'].value_counts()
print(f"Número de sectores: {sector_count.count()}")
print("Distribución por sector:")
print(sector_count)

# 2.4 Distribución geográfica
print("\n2.4 DISTRIBUCIÓN GEOGRÁFICA")
country_count = df['country_of_exchange'].value_counts()
print(f"Número de países: {country_count.count()}")
print("Distribución por país:")
print(country_count)

# 2.1 Empresas en la muestra
num_empresas = df['identifier'].nunique()
print(f"\n2.1 Número total de empresas únicas: {num_empresas}")

# Empresas por país (conteo de empresas únicas por país)
empresas_por_pais = df.groupby('country_of_exchange')['identifier'].nunique().sort_values(ascending=False)
print("\nNúmero de empresas por país:")
print(empresas_por_pais)

# Empresas por sector (conteo de empresas únicas por sector)
empresas_por_sector = df.groupby('trbc_economic_sector_name')['identifier'].nunique().sort_values(ascending=False)
print("\nNúmero de empresas por sector económico:")
print(empresas_por_sector)

# 2.2 Cobertura temporal
print("\n2.2 Cobertura temporal")
print(f"Años disponibles: {sorted(df['year'].unique())}")
print(f"Número de años: {df['year'].nunique()}")
year_count = df['year'].value_counts().sort_index()
print("Observaciones por año:")
print(year_count)


# SECCIÓN 3: ANÁLISIS DETALLADO DE VARIABLES ESG
print("\n" + "="*80)
print("SECCIÓN 3: ANÁLISIS DETALLADO DE VARIABLES ESG")
print("="*80)

# 3.1 Estadísticas descriptivas de variables ESG
print("\n3.1 ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES ESG")
esg_variables = ['esg', 'controversies', 'social', 'governance', 'environmental', 
                'resource_use', 'emissions', 'env_inn', 'workforce', 'human_rights', 
                'community', 'product_resp', 'management', 'shareholders', 'csr_strategy']
print(df[esg_variables].describe().T)

# 3.2 Evolución temporal de las variables ESG principales
print("\n3.2 EVOLUCIÓN TEMPORAL DE VARIABLES ESG PRINCIPALES")
esg_time_agg = df.groupby('year')[['esg', 'environmental', 'social', 'governance']].agg(['mean', 'median', 'std', 'min', 'max'])
print(esg_time_agg)

# 3.3 Comparación de puntuaciones ESG por sector
print("\n3.3 COMPARACIÓN DE PUNTUACIONES ESG POR SECTOR")
esg_sector_agg = df.groupby('trbc_economic_sector_name')[['esg', 'environmental', 'social', 'governance']].agg(['mean', 'median', 'std'])
print(esg_sector_agg)

# 3.4 Análisis de componentes ESG por año
esg_components_by_year = df.groupby('year')[esg_variables].mean().reset_index()
print("\n3.4 EVOLUCIÓN DE COMPONENTES ESG POR AÑO")
print(esg_components_by_year.to_string())

# SECCIÓN 4: ANÁLISIS DE VARIABLES FINANCIERAS
print("\n" + "="*80)
print("SECCIÓN 4: ANÁLISIS DE VARIABLES FINANCIERAS")
print("="*80)

# 4.1 Estadísticas descriptivas de variables financieras
print("\n4.1 ESTADÍSTICAS DESCRIPTIVAS DE VARIABLES FINANCIERAS")
financial_vars = ['roa', 'roe', 'price_bv', 'rev_business_activities', 'cash_flow', 
                  'total_liabilities', 'total_assets', 'interest_coverage_ratio', 
                  'debt', 'wacc_cost_of_debt']
print(df[financial_vars].describe().T)

# 4.2 Evolución temporal de ROA y ROE
print("\n4.2 EVOLUCIÓN TEMPORAL DE ROA Y ROE")
fin_time_agg = df.groupby('year')[['roa', 'roe']].agg(['mean', 'median', 'std', 'min', 'max'])
print(fin_time_agg)

# 4.3 Comparación de ROA y ROE por sector
print("\n4.3 COMPARACIÓN DE ROA Y ROE POR SECTOR")
fin_sector_agg = df.groupby('trbc_economic_sector_name')[['roa', 'roe']].agg(['mean', 'median', 'std'])
print(fin_sector_agg)

# 4.4 Análisis de ratio de endeudamiento y estructura financiera
print("\n4.4 ANÁLISIS DE ESTRUCTURA FINANCIERA")
# Calculando algunas ratios adicionales
df['debt_to_assets'] = df['total_liabilities'] / df['total_assets']
df['current_ratio'] = df['current_assets'] / df['total_liabilities']

structure_vars = ['total_assets', 'total_liabilities', 'debt_to_assets', 'current_ratio', 'interest_coverage_ratio']
print(df[structure_vars].describe().T)

# SECCIÓN 5: RELACIÓN ENTRE VARIABLES ESG Y FINANCIERAS
print("\n" + "="*80)
print("SECCIÓN 5: RELACIÓN ENTRE VARIABLES ESG Y FINANCIERAS")
print("="*80)

# 5.1 Correlación entre variables ESG y financieras
print("\n5.1 CORRELACIÓN ENTRE VARIABLES ESG Y FINANCIERAS")
correlation_vars = ['esg', 'environmental', 'social', 'governance', 'controversies',
                    'roa', 'roe', 'price_bv', 'debt_to_assets', 
                    'interest_coverage_ratio', 'cash_flow']

# Calcular matriz de correlación
corr_matrix = df[correlation_vars].corr().round(3)

# Función para calcular p-valores de correlaciones
def corr_pvals(df_vars):
    pvals = pd.DataFrame(np.ones((len(df_vars.columns), len(df_vars.columns))), columns=df_vars.columns, index=df_vars.columns)
    for i in range(len(df_vars.columns)):
        for j in range(i+1, len(df_vars.columns)):
            col1 = df_vars.columns[i]
            col2 = df_vars.columns[j]
            corr_test = pearsonr(df_vars[col1].dropna(), df_vars[col2].dropna())
            pvals.loc[col1, col2] = corr_test[1]
            pvals.loc[col2, col1] = corr_test[1]
    return pvals

pvals_matrix = corr_pvals(df[correlation_vars])

# Visualización matriz correlación con mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', center=0)
plt.title('Matriz de correlación entre variables ESG y financieras')
plt.show()

# Mostrar correlaciones significativas al 5%
alpha = 0.05
print("\nCorrelaciones estadísticamente significativas (p < 0.05):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        var1 = corr_matrix.columns[i]
        var2 = corr_matrix.columns[j]
        corr_value = corr_matrix.iloc[i, j]
        p_value = pvals_matrix.iloc[i, j]
        if p_value < alpha:
            print(f"{var1} - {var2}: correlación = {corr_value}, p-valor = {p_value:.4e}")

# 5.2 Evolución conjunta de ESG y rendimiento financiero
print("\n5.2 EVOLUCIÓN CONJUNTA DE ESG Y RENDIMIENTO FINANCIERO")
joint_evolution = df.groupby('year')[['esg', 'roa', 'roe']].mean()
print(joint_evolution)

# 5.3 ESG y rendimiento financiero por sector
print("\n5.3 ESG Y RENDIMIENTO FINANCIERO POR SECTOR")
sector_performance = df.groupby('trbc_economic_sector_name')[['esg', 'roa', 'roe']].mean().sort_values('esg', ascending=False)
print(sector_performance)

# SECCIÓN 6: VISUALIZACIONES

# 6.1 Distribución de puntuaciones ESG
plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=plt.gcf())

# Histograma de ESG global
ax1 = plt.subplot(gs[0, 0])
sns.histplot(df['esg'], kde=True, ax=ax1, color=custom_palette[0])
ax1.set_title('Distribución de Puntuaciones ESG')
ax1.set_xlabel('Puntuación ESG')

# Histogramas de componentes ESG
ax2 = plt.subplot(gs[0, 1])
components = ['environmental', 'social', 'governance']
# Usar colores específicos para cada componente para mejor diferenciación
colors = {'environmental': custom_palette[1], 'social': custom_palette[2], 'governance': custom_palette[3]}
for comp in components:
    sns.kdeplot(df[comp], label=comp.capitalize(), ax=ax2, color=colors[comp], linewidth=2.5)
ax2.set_title('Distribución de Componentes ESG')
ax2.set_xlabel('Puntuación')
ax2.legend()

# Boxplot de ESG por año
ax3 = plt.subplot(gs[0, 2])
sns.boxplot(x='year', y='esg', data=df, ax=ax3, palette=custom_palette)
ax3.set_title('Evolución de Puntuaciones ESG por Año')
ax3.set_xlabel('Año')
ax3.set_ylabel('Puntuación ESG')

# Boxplot de ESG por sector
ax4 = plt.subplot(gs[1, :])
sns.boxplot(x='trbc_economic_sector_name', y='esg', data=df, ax=ax4, palette=custom_palette)
ax4.set_title('Distribución de Puntuaciones ESG por Sector')
ax4.set_xlabel('Sector Económico')
ax4.set_ylabel('Puntuación ESG')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 6.2 Evolución temporal de variables clave
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=plt.gcf())

# Evolución de componentes ESG - colores específicos para cada línea
ax1 = plt.subplot(gs[0, 0])
for i, col in enumerate(['esg', 'environmental', 'social', 'governance']):
    esg_components_by_year.plot(x='year', y=col, marker='o', ax=ax1, 
                               color=custom_palette[i], 
                               linewidth=2.5,
                               label=col.capitalize())
ax1.set_title('Evolución de Componentes ESG')
ax1.set_xlabel('Año')
ax1.set_ylabel('Puntuación')
ax1.grid(True)
ax1.legend()

# Evolución de ROA y ROE - colores específicos
ax2 = plt.subplot(gs[0, 1])
fin_by_year = df.groupby('year')[['roa', 'roe']].mean().reset_index()
fin_by_year.plot(x='year', y='roa', marker='o', ax=ax2, color=custom_palette[4], linewidth=2.5, label='ROA')
fin_by_year.plot(x='year', y='roe', marker='o', ax=ax2, color=custom_palette[5], linewidth=2.5, label='ROE')
ax2.set_title('Evolución de ROA y ROE')
ax2.set_xlabel('Año')
ax2.set_ylabel('Porcentaje (%)')
ax2.grid(True)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend()

# ESG vs. Variables financieras - colores diferenciados
ax3 = plt.subplot(gs[1, 0])
# Usamos colores específicos para cada línea
joint_evolution.reset_index().plot(x='year', y='roa', marker='o', ax=ax3, color=custom_palette[6], linewidth=2.5, label='ROA')
joint_evolution.reset_index().plot(x='year', y='roe', marker='o', ax=ax3, color=custom_palette[7], linewidth=2.5, label='ROE')
joint_evolution.reset_index().plot(x='year', y='esg', marker='o', ax=ax3, color=custom_palette[8], linewidth=2.5, 
                                  secondary_y=True, label='ESG')
ax3.set_title('Evolución de ESG, ROA y ROE')
ax3.set_xlabel('Año')
ax3.set_ylabel('ROA/ROE')
ax3.right_ax.set_ylabel('ESG')
ax3.grid(True)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
# Crear una leyenda combinada
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3.right_ax.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='best')


# Estructura financiera promedio por año
ax4 = plt.subplot(gs[1, 1])
df.groupby('year')['debt_to_assets'].mean().plot(kind='bar', ax=ax4, color=custom_palette[9])
ax4.set_title('Ratio Promedio de Deuda/Activos por Año')
ax4.set_xlabel('Año')
ax4.set_ylabel('Deuda/Activos')
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax4.grid(True, axis='y')

plt.tight_layout()
plt.show()

# 6.3 Análisis sectorial - usar colores más distintos
plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=plt.gcf())

# Puntuación ESG promedio por sector
ax1 = plt.subplot(gs[0, 0])
sector_performance.sort_values('esg')['esg'].plot(kind='barh', ax=ax1, color=custom_palette[10])
ax1.set_title('Puntuación ESG Promedio por Sector')
ax1.set_xlabel('Puntuación ESG')
ax1.set_ylabel('Sector Económico')
ax1.grid(True, axis='x')

# ROA promedio por sector
ax2 = plt.subplot(gs[0, 1])
sector_performance.sort_values('roa')['roa'].plot(kind='barh', ax=ax2, color=custom_palette[11])
ax2.set_title('ROA Promedio por Sector')
ax2.set_xlabel('ROA')
ax2.set_ylabel('Sector Económico')
ax2.grid(True, axis='x')
ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# ROE promedio por sector
ax3 = plt.subplot(gs[1, 0])
sector_performance.sort_values('roe')['roe'].plot(kind='barh', ax=ax3, color=custom_palette[12])
ax3.set_title('ROE Promedio por Sector')
ax3.set_xlabel('ROE')
ax3.set_ylabel('Sector Económico')
ax3.grid(True, axis='x')
ax3.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Composición sectorial de la muestra - usar colormap para máxima diferenciación
ax4 = plt.subplot(gs[1, 1])
# Para el gráfico de pie, usaremos una paleta cíclica con colores claramente diferenciados
sector_count.plot(kind='pie', autopct='%1.1f%%', ax=ax4, colors=custom_palette)
ax4.set_title('Composición Sectorial de la Muestra')
ax4.set_ylabel('')

plt.tight_layout()
plt.show()

# 6.4 Correlación entre variables - usar un mapa de calor más contrastado
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# Cambiar a una paleta de color más contrastada para el mapa de calor
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", 
            mask=mask, vmin=-1, vmax=1, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación entre Variables ESG y Financieras')
plt.tight_layout()
plt.show()

# 6.5 Scatter plots de relaciones clave - mejorando los colores
plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=plt.gcf())

# ESG vs ROA - usando paleta de colores altamente diferenciables
ax1 = plt.subplot(gs[0, 0])
sns.scatterplot(x='esg', y='roa', hue='trbc_economic_sector_name', data=df, ax=ax1, palette=custom_palette)
ax1.set_title('Relación entre ESG y ROA')
ax1.set_xlabel('Puntuación ESG')
ax1.set_ylabel('ROA')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# ESG vs ROE
ax2 = plt.subplot(gs[0, 1])
sns.scatterplot(x='esg', y='roe', hue='trbc_economic_sector_name', data=df, ax=ax2, palette=custom_palette)
ax2.set_title('Relación entre ESG y ROE')
ax2.set_xlabel('Puntuación ESG')
ax2.set_ylabel('ROE')
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# ESG vs Price-to-Book
ax3 = plt.subplot(gs[1, 0])
sns.scatterplot(x='esg', y='price_bv', hue='trbc_economic_sector_name', data=df, ax=ax3, palette=custom_palette)
ax3.set_title('Relación entre ESG y Price-to-Book Value')
ax3.set_xlabel('Puntuación ESG')
ax3.set_ylabel('Price-to-Book Value')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Governance vs Interest Coverage Ratio
ax4 = plt.subplot(gs[1, 1])
sns.scatterplot(x='governance', y='interest_coverage_ratio', hue='trbc_economic_sector_name', data=df, ax=ax4, palette=custom_palette)
ax4.set_title('Relación entre Governance y Interest Coverage Ratio')
ax4.set_xlabel('Puntuación Governance')
ax4.set_ylabel('Interest Coverage Ratio')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# SECCIÓN 7: ANÁLISIS ADICIONAL - SUBCOMPONENTES ESG

# 7.1 Ranking de empresas por ESG
print("\n" + "="*80)
print("SECCIÓN 7: ANÁLISIS ADICIONAL - RANKINGS Y SUBCOMPONENTES")
print("="*80)

# Top 10 empresas por ESG (promedio de todos los años)
print("\n7.1 TOP 10 EMPRESAS POR PUNTUACIÓN ESG")
top_esg_companies = df.groupby('company_name')['esg'].mean().sort_values(ascending=False).head(10)
print(top_esg_companies)

# Bottom 10 empresas por ESG
print("\n7.2 BOTTOM 10 EMPRESAS POR PUNTUACIÓN ESG")
bottom_esg_companies = df.groupby('company_name')['esg'].mean().sort_values().head(10)
print(bottom_esg_companies)

# 7.3 Análisis detallado de subcomponentes
print("\n7.3 ANÁLISIS DE SUBCOMPONENTES ESG")
env_components = ['resource_use', 'emissions', 'env_inn']
social_components = ['workforce', 'human_rights', 'community', 'product_resp']
gov_components = ['management', 'shareholders', 'csr_strategy']

# Promedio por subcomponente
print("\nSubcomponentes Environmental:")
print(df[env_components].mean().sort_values(ascending=False))

print("\nSubcomponentes Social:")
print(df[social_components].mean().sort_values(ascending=False))

print("\nSubcomponentes Governance:")
print(df[gov_components].mean().sort_values(ascending=False))

# Correlación entre subcomponentes y rendimiento financiero
print("\n7.4 CORRELACIÓN ENTRE SUBCOMPONENTES Y RENDIMIENTO FINANCIERO")
subcomp_fin_corr = df[env_components + social_components + gov_components + ['roa', 'roe']].corr().loc[['roa', 'roe']].T
print(subcomp_fin_corr)

# SECCIÓN 8: RESUMEN GENERAL Y CONCLUSIONES
print("\n" + "="*80)
print("SECCIÓN 8: RESUMEN GENERAL Y CONCLUSIONES")
print("="*80)

# 8.1 Estadísticas clave
print("\n8.1 ESTADÍSTICAS CLAVE")
print(f"Número total de observaciones: {len(df)}")
print(f"Número de empresas: {df['identifier'].nunique()}")
print(f"Período temporal: {df['year'].min()} - {df['year'].max()}")
print(f"Promedio ESG global: {df['esg'].mean():.2f}")
print(f"Promedio ROA global: {df['roa'].mean():.4f} ({df['roa'].mean()*100:.2f}%)")
print(f"Promedio ROE global: {df['roe'].mean():.4f} ({df['roe'].mean()*100:.2f}%)")

# 8.2 Tendencias y patrones identificados
print("\n8.2 TENDENCIAS Y PATRONES IDENTIFICADOS")
print("- Evolución ESG:")
esg_trend = df.groupby('year')['esg'].mean().pct_change().mean() * 100
print(f"  * Variación promedio anual: {esg_trend:.2f}%")
print(f"  * Diferencia entre 2017 y 2023: {df[df['year']==2023]['esg'].mean() - df[df['year']==2017]['esg'].mean():.2f} puntos")

print("\n- Evolución Financiera:")
roa_trend = df.groupby('year')['roa'].mean().pct_change().mean() * 100
roe_trend = df.groupby('year')['roe'].mean().pct_change().mean() * 100
print(f"  * Variación ROA promedio anual: {roa_trend:.2f}%")
print(f"  * Variación ROE promedio anual: {roe_trend:.2f}%")

# 8.3 Diferencias sectoriales
print("\n8.3 DIFERENCIAS SECTORIALES SIGNIFICATIVAS")
sector_esg_max = sector_performance['esg'].idxmax()
sector_esg_min = sector_performance['esg'].idxmin()
print(f"- Sector con mayor puntuación ESG: {sector_esg_max} ({sector_performance.loc[sector_esg_max, 'esg']:.2f})")
print(f"- Sector con menor puntuación ESG: {sector_esg_min} ({sector_performance.loc[sector_esg_min, 'esg']:.2f})")

sector_roa_max = sector_performance['roa'].idxmax()
sector_roa_min = sector_performance['roa'].idxmin()
print(f"- Sector con mayor ROA: {sector_roa_max} ({sector_performance.loc[sector_roa_max, 'roa']*100:.2f}%)")
print(f"- Sector con menor ROA: {sector_roa_min} ({sector_performance.loc[sector_roa_min, 'roa']*100:.2f}%)")


# Definir categorías ESG según percentiles
esg_percentiles = df['esg'].quantile([0.25, 0.5, 0.75]).values
def categoria_esg(valor):
    if valor <= esg_percentiles[0]:
        return 'Bajo'
    elif valor <= esg_percentiles[1]:
        return 'Medio-bajo'
    elif valor <= esg_percentiles[2]:
        return 'Medio-alto'
    else:
        return 'Alto'

df['categoria_esg'] = df['esg'].apply(categoria_esg)

# Mostrar conteo de empresas por categoría ESG
print("\nNúmero de observaciones por categoría ESG:")
print(df['categoria_esg'].value_counts())

# Evolución del ROA promedio por año y categoría ESG
roa_por_categoria = df.groupby(['year', 'categoria_esg'])['roa'].mean().reset_index()

# Gráfico evolución ROA por categoría ESG
plt.figure(figsize=(12, 8))
sns.lineplot(data=roa_por_categoria, x='year', y='roa', hue='categoria_esg', marker='o',palette=['#2ca02c','#d62728','#1f77b4', '#ff7f0e'])
plt.title('Evolución del ROA por categoría ESG')
plt.xlabel('Año')
plt.ylabel('ROA promedio')
plt.grid(True)
plt.legend(title='Categoría ESG')
plt.show()

# Evolución del ROA promedio por año y categoría ESG
roe_por_categoria = df.groupby(['year', 'categoria_esg'])['roe'].mean().reset_index()

# Gráfico evolución ROA por categoría ESG
plt.figure(figsize=(12, 8))
sns.lineplot(data=roe_por_categoria, x='year', y='roe', hue='categoria_esg', marker='o',palette=['#2ca02c','#d62728','#1f77b4', '#ff7f0e'])
plt.title('Evolución del ROE por categoría ESG')
plt.xlabel('Año')
plt.ylabel('ROA promedio')
plt.grid(True)
plt.legend(title='Categoría ESG')
plt.show()

# Filtrar los datos para el año 2023
df_2023 = df[df['year'] == 2023]

# Agrupar por categoría ESG y calcular el promedio de ROA y ROE
roa_por_categoria_2023 = df_2023.groupby('categoria_esg')['roa'].mean().reset_index()
roe_por_categoria_2023 = df_2023.groupby('categoria_esg')['roe'].mean().reset_index()

# Gráfico de barras para ROA por categoría ESG en 2023 con colores personalizados
plt.figure(figsize=(10, 6))
sns.barplot(data=roa_por_categoria_2023, x='categoria_esg', y='roa', palette=['#2ca02c','#d62728','#1f77b4', '#ff7f0e'])
plt.title('ROA promedio por categoría ESG en 2023')
plt.xlabel('Categoría ESG')
plt.ylabel('ROA promedio')
plt.grid(True, axis='y')
plt.show()

# Gráfico de barras para ROE por categoría ESG en 2023 con colores personalizados
plt.figure(figsize=(10, 6))
sns.barplot(data=roe_por_categoria_2023, x='categoria_esg', y='roe', palette=['#2ca02c','#d62728','#1f77b4', '#ff7f0e'])
plt.title('ROE promedio por categoría ESG en 2023')
plt.xlabel('Categoría ESG')
plt.ylabel('ROE promedio')
plt.grid(True, axis='y')
plt.show()




print("\nAnálisis completado.")


# Guardar datos imputados
df.to_excel(r'c:\Users\laura\Downloads\bbdd_eu50_imputado.xlsx', index=False)
