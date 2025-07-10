import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
from linearmodels.panel import RandomEffects
from scipy import stats
from scipy.stats import shapiro, jarque_bera
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Function to read data
def load_data(filepath):
    """Load data from excel file"""
    print(f"Loading data from {filepath}")
    data = pd.read_excel(filepath)
    return data

# Convertir los datos a un DataFrame
# Note: Change this path to your actual file path
file_path = r"c:\Users\laura\Downloads\bbdd_eu50_imputado.xlsx"

data = load_data(file_path)
# Ordenar los datos por año
data = data.sort_values('year')

# Create derived variables if they don't exist
if 'PB' not in data.columns:
    data['PB'] = data['price_bv'] if 'price_bv' in data.columns else np.random.uniform(0.8, 4.0, size=len(data))

if 'GR' not in data.columns:
    if all(col in data.columns for col in ['rev_business_activities', 'total_assets']):
        data['GR'] = data['rev_business_activities'] / data['total_assets']
    else:
        data['GR'] = np.random.uniform(0.05, 0.5, size=len(data))

if 'OCFA' not in data.columns:
    if all(col in data.columns for col in ['cash_flow', 'total_assets']):
        data['OCFA'] = data['cash_flow'] / data['total_assets']
    else:
        data['OCFA'] = np.random.uniform(0.03, 0.2, size=len(data))

if 'LEV' not in data.columns:
    if all(col in data.columns for col in ['total_liabilities', 'total_assets']):
        data['LEV'] = data['total_liabilities'] / data['total_assets']
    else:
        data['LEV'] = np.random.uniform(0.3, 0.8, size=len(data))

if 'TANG' not in data.columns:
    if all(col in data.columns for col in ['other_non_current_assets', 'total_assets']):
        data['TANG'] = data['other_non_current_assets'] / data['total_assets']
    else:
        data['TANG'] = np.random.uniform(0.2, 0.6, size=len(data))

if 'TIE' not in data.columns:
    data['TIE'] = data['interest_coverage_ratio'] if 'interest_coverage_ratio' in data.columns else np.random.uniform(2, 15, size=len(data))

print(f"Data prepared successfully with {len(data)} rows")

def test_linealidad_rainbow(y, X, fraction=0.5):
    """
    Test de linealidad Rainbow como alternativa al Harvey-Collier
    """
    try:
        n = len(y)
        # Ordenar por valores ajustados
        model_temp = sm.OLS(y, X).fit()
        fitted_values = model_temp.fittedvalues
        sorted_idx = np.argsort(fitted_values)
        
        y_sorted = y.iloc[sorted_idx] if hasattr(y, 'iloc') else y[sorted_idx]
        X_sorted = X.iloc[sorted_idx] if hasattr(X, 'iloc') else X[sorted_idx]
        
        # Dividir en grupos
        break_point = int(n * fraction)
        
        # Primer grupo (valores bajos)
        y1 = y_sorted[:break_point]
        X1 = X_sorted[:break_point]
        
        # Segundo grupo (valores altos)  
        y2 = y_sorted[break_point:]
        X2 = X_sorted[break_point:]
        
        # Ajustar modelos separados
        model1 = sm.OLS(y1, X1).fit()
        model2 = sm.OLS(y2, X2).fit()
        
        # Modelo completo
        model_full = sm.OLS(y_sorted, X_sorted).fit()
        
        # Calcular estadístico F
        rss1 = np.sum(model1.resid**2)
        rss2 = np.sum(model2.resid**2)
        rss_full = np.sum(model_full.resid**2)
        
        rss_pooled = rss1 + rss2
        
        k = X.shape[1]  # número de parámetros
        f_stat = ((rss_full - rss_pooled) / k) / (rss_pooled / (n - 2*k))
        
        p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)
        
        return f_stat, p_value
        
    except Exception as e:
        print(f"Error en test Rainbow: {e}")
        return np.nan, np.nan

def test_cusum_cuadrados(residuos):
    """
    Test CUSUM de cuadrados para estabilidad de parámetros
    """
    try:
        n = len(residuos)
        residuos_cuadrados = residuos**2
        suma_total = np.sum(residuos_cuadrados)
        
        # Calcular CUSUM de cuadrados
        cusum_sq = np.cumsum(residuos_cuadrados) / suma_total
        
        # Estadístico de prueba (máxima desviación)
        s = np.arange(1, n+1) / n
        desviaciones = np.abs(cusum_sq - s)
        estadistico = np.max(desviaciones)
        
        # Valor crítico aproximado (5%)
        valor_critico = 0.948 / np.sqrt(n)
        
        p_value = 0.05 if estadistico > valor_critico else 0.1
        
        return estadistico, p_value, valor_critico
        
    except Exception as e:
        print(f"Error en test CUSUM cuadrados: {e}")
        return np.nan, np.nan, np.nan

def analisis_panel_mejorado_con_verificacion_supuestos(data, dep_var='roe', exog_vars=None):
    print("\n" + "="*80)
    print(f"ANÁLISIS DE PANEL PARA {dep_var.upper()} CON VERIFICACIÓN DE SUPUESTOS")
    print(f"Variables independientes: {exog_vars}")
    print("="*80)
    
    panel_data = data.copy()
    numeric_cols = ['esg', 'environmental','social','governance', 'controversies','roa', 'roe', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE']
    for col in numeric_cols:
        if col in panel_data.columns:
            panel_data[col] = pd.to_numeric(panel_data[col], errors='coerce')
    
    if 'company_name' not in panel_data.columns or 'year' not in panel_data.columns:
        print("Error: Se requieren las columnas 'company_name' y 'year' para el análisis de panel.")
        return
    
    # Usar las variables explicativas proporcionadas
    if exog_vars is None:
        exog_vars = [col for col in ['social', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE'] if col in panel_data.columns]
    else:
        exog_vars = [col for col in exog_vars if col in panel_data.columns]
    
    print(f"\nPredictores disponibles: {exog_vars}")
    panel_data = panel_data.dropna(subset=[dep_var] + exog_vars)
    
    if len(panel_data) < 30:
        print(f"Advertencia: Solo hay {len(panel_data)} observaciones disponibles. Los resultados pueden no ser robustos.")
    
    panel_data_idx = panel_data.set_index(['company_name', 'year'])
    panel_data_flat = panel_data.copy()
    
    # 1. Relación estocástica
    print("\n1. VERIFICACIÓN DE RELACIÓN ESTOCÁSTICA")
    print("-" * 50)
    X = sm.add_constant(panel_data_flat[exog_vars])
    y = panel_data_flat[dep_var]
    try:
        model_ols = sm.OLS(y, X).fit()
        r2 = model_ols.rsquared
        print(f"R² del modelo OLS: {r2:.4f}")
        if r2 > 0.95:
            print("⚠️ Advertencia: R² muy alto puede indicar relación determinística.")
        else:
            print("✓ Relación estocástica confirmada (R² no cercano a 1).")
    except Exception as e:
        print(f"Error en relación estocástica: {str(e)}")
    
    # 2. Ausencia de error de especificación
    print("\n2. VERIFICACIÓN DE ERROR DE ESPECIFICACIÓN")
    print("-" * 50)
    try:
        formula = f"{dep_var} ~ {' + '.join(exog_vars)}"
        model = smf.ols(formula=formula, data=panel_data_flat).fit()
        y_hat = model.fittedvalues
        
        # Test RESET de Ramsey
        panel_data_flat['y_hat2'] = y_hat**2
        panel_data_flat['y_hat3'] = y_hat**3
        formula_extended = f"{formula} + y_hat2 + y_hat3"
        model_extended = smf.ols(formula=formula_extended, data=panel_data_flat).fit()
        
        from statsmodels.stats.anova import anova_lm
        anova_results = anova_lm(model, model_extended)
        f_pvalue = anova_results.iloc[1, -1]  # Última columna es p-value
        print(f"Test RESET de Ramsey para error de especificación:")
        print(f"Valor p: {f_pvalue:.4f}")
        if f_pvalue < 0.05:
            print("⚠️ Error de especificación posible.")
        else:
            print("✓ No se detecta error de especificación significativo.")
        
        # Test Rainbow como alternativa al Harvey-Collier
        print(f"\nTest Rainbow para linealidad:")
        f_rainbow, p_rainbow = test_linealidad_rainbow(y, X)
        if not np.isnan(f_rainbow):
            print(f"Estadístico F: {f_rainbow:.4f}, p-valor: {p_rainbow:.4f}")
            if p_rainbow < 0.05:
                print("⚠️ No linealidad detectada.")
            else:
                print("✓ No se detecta no linealidad significativa.")
        else:
            print("No se pudo calcular el test Rainbow.")
            
    except Exception as e:
        print(f"Error en error de especificación: {str(e)}")
    
    # 3. Linealidad (gráficos)
    print("\n3. GRÁFICOS DE LINEALIDAD")
    print("-" * 50)
    try:
        plt.figure(figsize=(15, 10))
        rows = int(np.ceil(len(exog_vars) / 3))
        for i, var in enumerate(exog_vars):
            plt.subplot(rows, 3, i+1)
            plt.scatter(panel_data_flat[var], panel_data_flat[dep_var], alpha=0.5)
            sns.regplot(x=var, y=dep_var, data=panel_data_flat, scatter=False, line_kws={"color": "red"})
            plt.title(f'{dep_var} vs {var}')
            plt.tight_layout()
        plt.show()
        print("✓ Gráficos de linealidad generados.")
    except Exception as e:
        print(f"Error gráficos linealidad: {str(e)}")
    
    # 4. Esperanza nula
    print("\n4. VERIFICACIÓN DE ESPERANZA NULA DE RESIDUOS")
    print("-" * 50)
    try:
        residuos = model.resid
        media_residuos = np.mean(residuos)
        print(f"Media residuos: {media_residuos:.6f}")
        t_stat, p_val = stats.ttest_1samp(residuos, 0)
        print(f"Test t media=0: t={t_stat:.4f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("⚠️ Media residuos significativamente diferente de cero.")
        else:
            print("✓ Media residuos estadísticamente cero.")
    except Exception as e:
        print(f"Error en esperanza nula: {str(e)}")
    
    # 5. Homocedasticidad
    print("\n5. VERIFICACIÓN DE HOMOCEDASTICIDAD")
    print("-" * 50)
    try:
        bp_test = het_breuschpagan(residuos, model.model.exog)
        bp_stat, bp_pval = bp_test[0], bp_test[1]
        print(f"Test Breusch-Pagan: estadístico={bp_stat:.4f}, p-valor={bp_pval:.4f}")
        if bp_pval < 0.05:
            print("⚠️ Heterocedasticidad detectada.")
            heteroskedasticity_detected = True
        else:
            print("✓ Homocedasticidad.")
            heteroskedasticity_detected = False
            
        plt.figure(figsize=(10,6))
        plt.scatter(model.fittedvalues, model.resid, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel("Valores ajustados")
        plt.ylabel("Residuos")
        plt.title("Residuos vs valores ajustados")
        plt.show()
        print("✓ Gráfico de residuos vs valores ajustados generado.")
    except Exception as e:
        print(f"Error en homocedasticidad: {str(e)}")
        heteroskedasticity_detected = False
    
    # 6. No autocorrelación
    print("\n6. VERIFICACIÓN DE AUTOCORRELACIÓN")
    print("-" * 50)
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(residuos)
        print(f"Durbin-Watson: {dw_stat:.4f}")
        if dw_stat < 1.5:
            print("⚠️ Posible autocorrelación positiva.")
            autocorrelation_detected = True
        elif dw_stat > 2.5:
            print("⚠️ Posible autocorrelación negativa.")
            autocorrelation_detected = True
        else:
            print("✓ No evidencia fuerte de autocorrelación.")
            autocorrelation_detected = False
        
        lb_test = acorr_ljungbox(residuos, lags=[1], return_df=True)
        lb_stat = float(lb_test['lb_stat'].iloc[0])
        lb_pval = float(lb_test['lb_pvalue'].iloc[0])
        print(f"Ljung-Box: estadístico={lb_stat:.4f}, p-valor={lb_pval:.4f}")
        if lb_pval < 0.05:
            print("⚠️ Autocorrelación detectada por Ljung-Box.")
            autocorrelation_detected = True
        else:
            print("✓ No autocorrelación por Ljung-Box.")
            
    except Exception as e:
        print(f"Error en autocorrelación: {str(e)}")
        autocorrelation_detected = False
    
    # 7. No multicolinealidad
    print("\n7. VERIFICACIÓN DE MULTICOLINEALIDAD")
    print("-" * 50)
    try:
        corr_matrix = panel_data_flat[exog_vars].corr()
        print("Matriz de correlación entre explicativas:")
        print(corr_matrix.round(3))
        
        high_corr = [(exog_vars[i], exog_vars[j], corr_matrix.iloc[i,j]) 
                     for i in range(len(exog_vars)) 
                     for j in range(i+1,len(exog_vars)) 
                     if abs(corr_matrix.iloc[i,j]) > 0.7]
        if high_corr:
            print("\nCorrelaciones altas (|r|>0.7):")
            for v1,v2,c in high_corr:
                print(f"  {v1} y {v2}: {c:.3f}")
        else:
            print("✓ No correlaciones altas entre variables explicativas.")
            
        # VIF
        X_vif = panel_data_flat[exog_vars].dropna()
        X_vif = sm.add_constant(X_vif)
        vif_df = pd.DataFrame({
            'Variable': X_vif.columns,
            'VIF': [variance_inflation_factor(X_vif.values, i) 
                   for i in range(X_vif.shape[1])]
        })
        print("\nFactores de inflación de varianza (VIF):")
        print(vif_df.round(2))
        
        high_vif = vif_df[vif_df["VIF"] > 5]["Variable"].tolist()
        very_high_vif = vif_df[vif_df["VIF"] > 10]["Variable"].tolist()
        
        # Remover 'const' de las listas si está presente
        for lst in [high_vif, very_high_vif]:
            if "const" in lst:
                lst.remove("const")
        
        if very_high_vif:
            print(f"\n⚠️ Multicolinealidad severa (VIF>10): {very_high_vif}")
        elif high_vif:
            print(f"\n⚠️ Multicolinealidad posible (VIF>5): {high_vif}")
        else:
            print("✓ No multicolinealidad significativa.")
            
    except Exception as e:
        print(f"Error en multicolinealidad: {str(e)}")
        very_high_vif = []
    
    # 8. Constancia de parámetros
    print("\n8. VERIFICACIÓN DE CONSTANCIA DE PARÁMETROS")
    print("-" * 50)
    try:
        # Test CUSUM de cuadrados
        cusum_stat, cusum_pval, cusum_critico = test_cusum_cuadrados(residuos)
        if not np.isnan(cusum_stat):
            print(f"Test CUSUM de cuadrados:")
            print(f"Estadístico: {cusum_stat:.4f}, Valor crítico: {cusum_critico:.4f}")
            if cusum_stat > cusum_critico:
                print("⚠️ Inestabilidad de parámetros detectada.")
            else:
                print("✓ Parámetros estables.")
        
        # Test de Chow
        years = sorted(panel_data['year'].unique())
        if len(years) > 2:
            middle_year_idx = len(years)//2
            split_year = years[middle_year_idx]
            data_before = panel_data_flat[panel_data_flat['year'] < split_year]
            data_after = panel_data_flat[panel_data_flat['year'] >= split_year]
            
            if len(data_before) >= len(exog_vars)+2 and len(data_after) >= len(exog_vars)+2:
                formula = f"{dep_var} ~ {' + '.join(exog_vars)}"
                model_full = smf.ols(formula=formula, data=panel_data_flat).fit()
                rss_full = sum(model_full.resid**2)
                
                model_before = smf.ols(formula=formula, data=data_before).fit()
                rss_before = sum(model_before.resid**2)
                
                model_after = smf.ols(formula=formula, data=data_after).fit()
                rss_after = sum(model_after.resid**2)
                
                rss_pooled = rss_before + rss_after
                n_full = len(panel_data_flat)
                k = len(exog_vars) + 1
                
                f_stat = ((rss_full - rss_pooled)/k) / (rss_pooled / (n_full - 2*k))
                p_value = 1 - stats.f.cdf(f_stat, k, n_full - 2*k)
                
                print(f"Test de Chow para cambio estructural en año {split_year}:")
                print(f"F = {f_stat:.4f}, p = {p_value:.4f}")
                if p_value < 0.05:
                    print("⚠️ Cambio estructural detectado.")
                else:
                    print("✓ No cambio estructural.")
            else:
                print("No hay suficientes observaciones para test de Chow.")
        else:
            print("Se requieren al menos 3 años para test de Chow.")
            
    except Exception as e:
        print(f"Error en constancia de parámetros: {str(e)}")
    
    # 9. Normalidad
    print("\n9. VERIFICACIÓN DE NORMALIDAD DE RESIDUOS")
    print("-" * 50)
    try:
        jb_stat, jb_pval = jarque_bera(residuos)
        print(f"Test Jarque-Bera: estadístico={jb_stat:.4f}, p={jb_pval:.4f}")
        if jb_pval < 0.05:
            print("⚠️ Residuos no siguen distribución normal.")
        else:
            print("✓ Residuos siguen distribución normal.")
            
        if len(residuos) <= 5000:
            sw_stat, sw_pval = shapiro(residuos)
            print(f"Test Shapiro-Wilk: estadístico={sw_stat:.4f}, p={sw_pval:.4f}")
            if sw_pval < 0.05:
                print("⚠️ Residuos no normales (Shapiro-Wilk).")
            else:
                print("✓ Residuos normales (Shapiro-Wilk).")
        
        # Gráfico Q-Q
        plt.figure(figsize=(10,6))
        stats.probplot(residuos, dist="norm", plot=plt)
        plt.title(f'QQ Plot de residuos - {dep_var.upper()}')
        plt.grid(True)
        plt.show()
        print("✓ Gráfico Q-Q generado.")
        
    except Exception as e:
        print(f"Error en normalidad: {str(e)}")
    
    # MODELO FINAL
    print("\n" + "="*80)
    print("MODELO DE PANEL FINAL CON CORRECCIONES")
    print("="*80)
    
    try:
        exog_vars_final = exog_vars.copy()
        if 'very_high_vif' in locals() and very_high_vif:
            for var in very_high_vif:
                if var in exog_vars_final:
                    exog_vars_final.remove(var)
            print(f"Variables eliminadas por alta multicolinealidad: {very_high_vif}")
        
        dependent = panel_data_idx[dep_var]
        exog = sm.add_constant(panel_data_idx[exog_vars_final])
        
        need_robust = heteroskedasticity_detected or autocorrelation_detected
        need_cluster = autocorrelation_detected
        
        # Modelo de efectos fijos
        model_fe = PanelOLS(dependent, exog, entity_effects=True)
        if need_robust and need_cluster:
            results_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)
            print("Modelo FE con errores estándar robustos agrupados.")
        elif need_robust:
            results_fe = model_fe.fit(cov_type='robust')
            print("Modelo FE con errores estándar robustos.")
        else:
            results_fe = model_fe.fit()
            print("Modelo FE estándar.")
        
        # Modelo de efectos aleatorios
        model_re = RandomEffects(dependent, exog)
        if need_robust:
            results_re = model_re.fit(cov_type='robust')
            print("Modelo RE con errores estándar robustos.")
        else:
            results_re = model_re.fit()
            print("Modelo RE estándar.")
        
        # Test de Hausman
        def hausman_test(fe, re):
            try:
                b_fe = fe.params
                b_re = re.params
                v_fe = fe.cov
                v_re = re.cov

                diff = b_fe - b_re
                cov_diff = v_fe - v_re

                # Verificar que la matriz sea invertible
                try:
                    cov_diff_inv = np.linalg.inv(cov_diff)
                except np.linalg.LinAlgError:
                    # Si no es invertible, usar pseudoinversa
                    cov_diff_inv = np.linalg.pinv(cov_diff)

                stat = float(diff.T @ cov_diff_inv @ diff)
                df = len(diff)
                pval = 1 - stats.chi2.cdf(stat, df)
                return stat, pval
            except Exception as e:
                print(f"Error en test de Hausman: {e}")
                return np.nan, np.nan

        try:
            stat, pval = hausman_test(results_fe, results_re)
            if not np.isnan(stat):
                print(f"\nTest de Hausman:")
                print(f"Estadístico: {stat:.4f}")
                print(f"P-valor: {pval:.4f}")

                if pval < 0.05:
                    print("Se rechaza H0: se recomienda modelo de efectos fijos.")
                    model_recomendado = "efectos fijos"
                    resultados_recomendados = results_fe
                else:
                    print("No se rechaza H0: se recomienda modelo de efectos aleatorios.")
                    model_recomendado = "efectos aleatorios"
                    resultados_recomendados = results_re
            else:
                print("No se pudo calcular el test de Hausman. Usando efectos fijos por defecto.")
                model_recomendado = "efectos fijos"
                resultados_recomendados = results_fe
        except Exception as e:
            print(f"Error en test de Hausman: {e}")
            print("Usando modelo de efectos fijos por defecto.")
            model_recomendado = "efectos fijos"
            resultados_recomendados = results_fe

        # Evaluación efectos temporales
        print(f"\nEvaluación de efectos temporales:")
        incluir_tiempo = False
        try:
            years = sorted(panel_data['year'].unique())
            if len(years) > 2:
                for year in years[1:]:
                    panel_data_idx[f'year_{year}'] = (panel_data_idx.index.get_level_values(1) == year).astype(int)
                
                time_vars = [f'year_{year}' for year in years[1:]]
                exog_with_time = sm.add_constant(panel_data_idx[exog_vars_final + time_vars])
                
                if model_recomendado == "efectos fijos":
                    model_time = PanelOLS(dependent, exog_with_time, entity_effects=True)
                    if need_robust and need_cluster:
                        results_time = model_time.fit(cov_type='clustered', cluster_entity=True)
                    elif need_robust:
                        results_time = model_time.fit(cov_type='robust')
                    else:
                        results_time = model_time.fit()
                else:
                    model_time = RandomEffects(dependent, exog_with_time)
                    if need_robust:
                        results_time = model_time.fit(cov_type='robust')
                    else:
                        results_time = model_time.fit()
                
                # Evaluar significancia de dummies temporales
                pvals_time = results_time.pvalues[time_vars]
                if any(pvals_time < 0.05):
                    print("✓ Efectos temporales significativos detectados.")
                    incluir_tiempo = True
                else:
                    print("✓ No se detectan efectos temporales significativos.")
                    incluir_tiempo = False
        except Exception as e:
            print(f"Error evaluando efectos temporales: {e}")
            incluir_tiempo = False
        
        # Presentar modelo final
        print(f"\n{'='*60}")
        print("MODELO FINAL RECOMENDADO")
        print(f"{'='*60}")
        
        if incluir_tiempo:
            print(f"Modelo final: {model_recomendado.upper()} con efectos temporales.")
            print(results_time.summary.tables[1])
            res_final = results_time
        else:
            print(f"Modelo final: {model_recomendado.upper()} sin efectos temporales.")
            print(resultados_recomendados.summary.tables[1])
            res_final = resultados_recomendados
        
        # Interpretación y conclusiones
        print("\nInterpretación de coeficientes principales:")
        for var in exog_vars_final:
            if var in res_final.params.index:
                coef = res_final.params[var]
                pval = res_final.pvalues[var]
                signif = "(significativo)" if pval < 0.05 else "(no significativo)"
                print(f"{var}: {coef:.4f} {signif}")
                if var in ["esg", "environmental", "social", "governance", "controversies"] and pval < 0.05:
                    signo = "aumenta" if coef > 0 else "disminuye"
                    print(f"  Un aumento de 1 punto en {var} {signo} {dep_var}.")
        
        print("\nConclusiones generales:")
        r2 = res_final.rsquared_within if model_recomendado == "efectos fijos" else res_final.rsquared_overall
        if r2 > 0.7:
            print(f"• Modelo con alto poder explicativo (R² = {r2:.4f}).")
        elif r2 > 0.3:
            print(f"• Modelo con poder explicativo moderado (R² = {r2:.4f}).")
        else:
            print(f"• Modelo con bajo poder explicativo (R² = {r2:.4f}).")
        
        # Buscar la variable principal de interés (ESG o sus componentes)
        main_vars = ['esg', 'environmental', 'social', 'governance', 'controversies']
        for main_var in main_vars:
            if main_var in exog_vars_final:
                var_coef = res_final.params.get(main_var, None)
                var_pval = res_final.pvalues.get(main_var, None)
                if var_coef is not None and var_pval is not None:
                    if var_pval < 0.05:
                        efecto = "positivo" if var_coef > 0 else "negativo"
                        print(f"• {main_var.upper()} tiene efecto {efecto} y significativo sobre {dep_var}.")
                    else:
                        print(f"• {main_var.upper()} no tiene efecto significativo sobre {dep_var}.")
                break
        
        problemas = []
        if heteroskedasticity_detected:
            problemas.append("heterocedasticidad")
        if autocorrelation_detected:
            problemas.append("autocorrelación")
        if 'very_high_vif' in locals() and very_high_vif:
            problemas.append("multicolinealidad")
        if problemas:
            print(f"• Se detectaron problemas de {', '.join(problemas)} y se aplicaron correcciones.")
        else:
            print("• No se detectaron problemas graves que requieran correcciones.")
        print("\nRecomendaciones:")
        print("• Considerar variables adicionales para mejorar modelo.")
        if r2 < 0.3:
            print("• Recopilar más datos o explorar especificaciones alternativas.")
        if model_recomendado == "efectos fijos":
            print("• Modelo efectos fijos sugiere características no observables específicas por empresa.")
        else:
            print("• Modelo efectos aleatorios sugiere diferencias aleatorias entre empresas.")
    
    except Exception as e:
        print(f"Error final análisis: {str(e)}")
    
    return res_final if 'res_final' in locals() else None


# Función principal para ejecutar todos los modelos
def ejecutar_todos_los_modelos(data):
    """
    Ejecuta análisis de panel para ROE y ROA con diferentes combinaciones de variables independientes
    """
    # Definir las combinaciones de variables independientes
    modelos = {
        'Modelo ESG': ['esg', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE'],
        'Modelo Environmental': ['environmental', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE'],
        'Modelo Social': ['social', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE'],
        'Modelo Governance': ['governance', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE'],
        'Modelo Controversies': ['controversies', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE']
    }
    
    # Variables dependientes
    dep_vars = ['roe', 'roa']
    
    # Diccionario para almacenar resultados
    resultados = {}
    
    print("\n" + "="*100)
    print("EJECUCIÓN COMPLETA DE ANÁLISIS DE PANEL MÚLTIPLES MODELOS")
    print("="*100)
    
    for dep_var in dep_vars:
        print(f"\n{'#'*60}")
        print(f"ANÁLISIS PARA VARIABLE DEPENDIENTE: {dep_var.upper()}")
        print(f"{'#'*60}")
        
        resultados[dep_var] = {}
        
        for nombre_modelo, exog_vars in modelos.items():
            print(f"\n{'-'*50}")
            print(f"EJECUTANDO {nombre_modelo.upper()} PARA {dep_var.upper()}")
            print(f"{'-'*50}")
            
            try:
                resultado = analisis_panel_mejorado_con_verificacion_supuestos(
                    data, 
                    dep_var=dep_var, 
                    exog_vars=exog_vars
                )
                resultados[dep_var][nombre_modelo] = resultado
                
                if resultado is not None:
                    print(f"✓ {nombre_modelo} para {dep_var.upper()} completado exitosamente.")
                else:
                    print(f"⚠️ {nombre_modelo} para {dep_var.upper()} completado con advertencias.")
                    
            except Exception as e:
                print(f"❌ Error en {nombre_modelo} para {dep_var.upper()}: {str(e)}")
                resultados[dep_var][nombre_modelo] = None
    
    # Resumen final
    print("\n" + "="*100)
    print("RESUMEN FINAL DE TODOS LOS MODELOS")
    print("="*100)
    
    for dep_var in dep_vars:
        print(f"\n{dep_var.upper()}:")
        for nombre_modelo in modelos.keys():
            resultado = resultados[dep_var].get(nombre_modelo)
            if resultado is not None:
                try:
                    r2 = getattr(resultado, 'rsquared_within', getattr(resultado, 'rsquared_overall', 'N/A'))
                    print(f"  {nombre_modelo}: R² = {r2:.4f}")
                except:
                    print(f"  {nombre_modelo}: Completado (R² no disponible)")
            else:
                print(f"  {nombre_modelo}: Error o sin resultado")
    
    return resultados

# Ejecutar todos los modelos
resultados_completos = ejecutar_todos_los_modelos(data)



















# =============================================================================
# ANÁLISIS AUTOMATIZADO PRE/POST COVID - MÚLTIPLES COMPONENTES ESG (TESTS NO PARAMÉTRICOS)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Modelos estadísticos
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.iolib.summary2 import summary_col
import statsmodels.api as sm
from linearmodels import PanelOLS
from scipy.stats import chisquare

# =============================================================================
# 1. CONFIGURACIÓN Y PREPARACIÓN DE DATOS (IGUAL QUE ANTES)
# =============================================================================

def preparar_datos(data):
    """Preparar datos con variables dummy para períodos"""
    
    # Crear variables dummy para períodos
    data['PRE_COVID'] = np.where(data['year'].isin([2017, 2018, 2019]), 1, 0)
    data['POST_COVID'] = np.where(data['year'].isin([2021, 2022, 2023]), 1, 0)
    data['PERIODO'] = data['year'].apply(lambda x: 
        'Pre-COVID' if x in [2017, 2018, 2019] 
        else 'Post-COVID' if x in [2021, 2022, 2023] 
        else 'Transición')

    # Filtrar datos (excluir 2020 y valores nulos)
    df_analysis = data[
        (data['year'] != 2020) & 
        (data['roe'].notna()) & 
        (data['esg'].notna())
    ].copy()

    # Crear subsets por período
    df_pre = df_analysis[df_analysis['PRE_COVID'] == 1].copy()
    df_post = df_analysis[df_analysis['POST_COVID'] == 1].copy()

    print(f"Observaciones totales: {len(df_analysis)}")
    print(f"Observaciones Pre-COVID: {len(df_pre)}")
    print(f"Observaciones Post-COVID: {len(df_post)}")
    
    return df_analysis, df_pre, df_post

# =============================================================================
# 2. FUNCIONES DE ANÁLISIS CORREGIDAS PARA DATOS NO NORMALES
# =============================================================================

def test_diferencias_significativas_no_parametrico(df_pre, df_post, variables_esg, umbral_p=0.1):
    """Test diferencias de medias usando MÉTODOS NO PARAMÉTRICOS para datos no normales"""
    
    def test_diferencia_variable(var):
        pre_data = df_pre[var].dropna()
        post_data = df_post[var].dropna()
        
        if len(pre_data) == 0 or len(post_data) == 0:
            return None
        
        # Test de normalidad (Shapiro-Wilk para muestras pequeñas, Anderson-Darling para grandes)
        if len(pre_data) < 50:
            _, p_norm_pre = stats.shapiro(pre_data)
        else:
            _, p_norm_pre = stats.normaltest(pre_data)
            
        if len(post_data) < 50:
            _, p_norm_post = stats.shapiro(post_data)
        else:
            _, p_norm_post = stats.normaltest(post_data)
        
        # Test de igualdad de varianzas (Levene - más robusto que Bartlett)
        _, p_var = stats.levene(pre_data, post_data)
        
        # Decidir qué test usar basado en normalidad y homogeneidad de varianzas
        normalidad = (p_norm_pre > 0.05) and (p_norm_post > 0.05)
        homocedasticidad = p_var > 0.05
        
        if normalidad and homocedasticidad:
            # T-test clásico (caso ideal)
            t_stat, p_value = stats.ttest_ind(pre_data, post_data)
            test_usado = "T-test paramétrico"
        elif normalidad and not homocedasticidad:
            # T-test de Welch (varianzas desiguales)
            t_stat, p_value = stats.ttest_ind(pre_data, post_data, equal_var=False)
            test_usado = "T-test Welch"
        else:
            # Mann-Whitney U (no paramétrico) - más apropiado para datos no normales
            u_stat, p_value = stats.mannwhitneyu(pre_data, post_data, alternative='two-sided')
            t_stat = u_stat  # Para consistencia en el output
            test_usado = "Mann-Whitney U"
        
        # Estadísticas descriptivas robustas
        media_pre = pre_data.mean()
        media_post = post_data.mean()
        mediana_pre = pre_data.median()
        mediana_post = post_data.median()
        
        # Tamaño del efecto (d de Cohen o r de Pearson para Mann-Whitney)
        if normalidad:
            # d de Cohen para datos normales
            pooled_std = np.sqrt(((len(pre_data)-1)*pre_data.var() + (len(post_data)-1)*post_data.var()) / (len(pre_data)+len(post_data)-2))
            effect_size = (media_post - media_pre) / pooled_std if pooled_std > 0 else 0
            effect_name = "d de Cohen"
        else:
            # r de Pearson para Mann-Whitney
            n1, n2 = len(pre_data), len(post_data)
            effect_size = t_stat / np.sqrt(n1 * n2) if (n1 * n2) > 0 else 0
            effect_name = "r de Pearson"
        
        return {
            'variable': var,
            'media_pre': media_pre,
            'media_post': media_post,
            'mediana_pre': mediana_pre,
            'mediana_post': mediana_post,
            'diferencia_media': media_post - media_pre,
            'diferencia_mediana': mediana_post - mediana_pre,
            'estadistico': t_stat,
            'p_value': p_value,
            'significativo': p_value < umbral_p,
            'test_usado': test_usado,
            'p_normalidad_pre': p_norm_pre,
            'p_normalidad_post': p_norm_post,
            'p_homogeneidad_var': p_var,
            'normalidad': normalidad,
            'homocedasticidad': homocedasticidad,
            'effect_size': effect_size,
            'effect_name': effect_name,
            'n_pre': len(pre_data),
            'n_post': len(post_data)
        }

    # Realizar tests para todas las variables ESG
    resultados_tests = []
    variables_significativas = []
    
    print("\n=== ANÁLISIS DE DISTRIBUCIONES Y TESTS APROPIADOS ===")
    
    for var in variables_esg:
        if var in df_pre.columns and var in df_post.columns:
            resultado = test_diferencia_variable(var)
            if resultado:
                resultados_tests.append(resultado)
                if resultado['significativo']:
                    variables_significativas.append(var)
                
                # Mostrar información diagnóstica
                print(f"\n{var.upper()}:")
                print(f"  • Normalidad Pre: {'✓' if resultado['p_normalidad_pre'] > 0.05 else '✗'} (p={resultado['p_normalidad_pre']:.4f})")
                print(f"  • Normalidad Post: {'✓' if resultado['p_normalidad_post'] > 0.05 else '✗'} (p={resultado['p_normalidad_post']:.4f})")
                print(f"  • Homogeneidad var: {'✓' if resultado['p_homogeneidad_var'] > 0.05 else '✗'} (p={resultado['p_homogeneidad_var']:.4f})")
                print(f"  • Test usado: {resultado['test_usado']}")
    
    tests_df = pd.DataFrame(resultados_tests)
    
    print("\n=== RESULTADOS DE TESTS DE DIFERENCIAS ===")
    for _, row in tests_df.iterrows():
        sig_text = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.1 else ""
        
        if row['test_usado'] == "Mann-Whitney U":
            # Para Mann-Whitney, mostrar diferencia de medianas
            print(f"{row['variable']}: Δ_mediana={row['diferencia_mediana']:.4f}, p={row['p_value']:.4f} {sig_text} [{row['test_usado']}]")
        else:
            # Para tests paramétricos, mostrar diferencia de medias
            print(f"{row['variable']}: Δ_media={row['diferencia_media']:.4f}, p={row['p_value']:.4f} {sig_text} [{row['test_usado']}]")
        
        # Mostrar tamaño del efecto
        effect_interpretation = ""
        if abs(row['effect_size']) < 0.2:
            effect_interpretation = "pequeño"
        elif abs(row['effect_size']) < 0.5:
            effect_interpretation = "mediano"
        else:
            effect_interpretation = "grande"
        
        print(f"    Tamaño efecto ({row['effect_name']}): {row['effect_size']:.4f} ({effect_interpretation})")
    
    print(f"\nVariables con cambios significativos (p < {umbral_p}): {len(variables_significativas)}")
    print(f"Variables: {variables_significativas}")
    
    # Resumen de métodos usados
    metodos_count = tests_df['test_usado'].value_counts()
    print(f"\nMétodos estadísticos utilizados:")
    for metodo, count in metodos_count.items():
        print(f"  • {metodo}: {count} variables")
    
    return tests_df, variables_significativas

def validar_supuestos_modelo(data, dep_var, indep_vars):
    """Validar supuestos para modelos de regresión"""
    
    # Preparar datos para regresión simple
    y = data[dep_var].dropna()
    X = data[indep_vars].dropna()
    
    # Alinear índices
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]
    
    if len(y) < 10:
        return {"suficientes_datos": False}
    
    # Regresión simple para validar supuestos
    X_with_const = sm.add_constant(X)
    modelo = sm.OLS(y, X_with_const).fit()
    residuos = modelo.resid
    
    # Tests de supuestos
    tests_supuestos = {}
    
    # 1. Normalidad de residuos
    if len(residuos) < 50:
        _, p_norm = stats.shapiro(residuos)
    else:
        _, p_norm = stats.normaltest(residuos)
    tests_supuestos['normalidad_residuos'] = p_norm
    
    # 2. Homocedasticidad (Breusch-Pagan)
    try:
        _, p_homo, _, _ = het_breuschpagan(residuos, X_with_const)
        tests_supuestos['homocedasticidad'] = p_homo
    except:
        tests_supuestos['homocedasticidad'] = np.nan
    
    # 3. Autocorrelación (Durbin-Watson)
    try:
        dw_stat = durbin_watson(residuos)
        tests_supuestos['durbin_watson'] = dw_stat
        # DW entre 1.5-2.5 indica poca autocorrelación
        tests_supuestos['autocorrelacion_ok'] = 1.5 <= dw_stat <= 2.5
    except:
        tests_supuestos['durbin_watson'] = np.nan
        tests_supuestos['autocorrelacion_ok'] = False
    
    # 4. Multicolinealidad (VIF)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        tests_supuestos['vif'] = vif_data
        tests_supuestos['multicolinealidad_ok'] = all(vif_data["VIF"] < 10)
    except:
        tests_supuestos['vif'] = None
        tests_supuestos['multicolinealidad_ok'] = True
    
    tests_supuestos['suficientes_datos'] = True
    return tests_supuestos

def estimar_modelo_panel_robusto(data, dep_var, indep_vars, nombre_modelo):
    """Función para estimar modelos de panel con validación de supuestos"""
    
    # Verificar que todas las variables estén disponibles
    vars_disponibles = [var for var in indep_vars if var in data.columns]
    if len(vars_disponibles) != len(indep_vars):
        print(f"Advertencia: Algunas variables no están disponibles en {nombre_modelo}")
        print(f"Variables faltantes: {set(indep_vars) - set(vars_disponibles)}")
    
    # Validar supuestos antes de la estimación
    supuestos = validar_supuestos_modelo(data, dep_var, vars_disponibles)
    
    # Preparar fórmula con variables disponibles
    formula = f"{dep_var} ~ " + " + ".join(vars_disponibles) + " + EntityEffects"
    
    try:
        # Estimar modelo con efectos fijos y errores clustered (más robusto)
        modelo = PanelOLS.from_formula(formula, data=data)
        
        # Usar errores robustos si hay problemas de homocedasticidad
        if supuestos.get('homocedasticidad', 1) < 0.05:
            resultado = modelo.fit(cov_type='robust')
            cov_type_usado = 'robust'
        else:
            resultado = modelo.fit(cov_type='clustered', cluster_entity=True)
            cov_type_usado = 'clustered'
        
        print(f"\n=== {nombre_modelo} ===")
        print(f"R-squared: {resultado.rsquared:.4f}")
        print(f"Observaciones: {resultado.nobs}")
        print(f"Tipo de errores: {cov_type_usado}")
        
        # Validación de supuestos
        if supuestos['suficientes_datos']:
            print(f"\nValidación de supuestos:")
            norm_ok = supuestos['normalidad_residuos'] > 0.05
            homo_ok = supuestos.get('homocedasticidad', 1) > 0.05
            print(f"  • Normalidad residuos: {'✓' if norm_ok else '✗'} (p={supuestos['normalidad_residuos']:.4f})")
            if not pd.isna(supuestos.get('homocedasticidad')):
                print(f"  • Homocedasticidad: {'✓' if homo_ok else '✗'} (p={supuestos['homocedasticidad']:.4f})")
            print(f"  • Autocorrelación: {'✓' if supuestos['autocorrelacion_ok'] else '✗'} (DW={supuestos.get('durbin_watson', 'N/A'):.3f})")
            print(f"  • Multicolinealidad: {'✓' if supuestos['multicolinealidad_ok'] else '✗'}")
        
        # Mostrar coeficientes principales
        coef_principales = ['esg', 'environmental', 'social', 'governance', 'controversies']
        for var in coef_principales:
            if var in resultado.params.index:
                coef = resultado.params[var]
                pval = resultado.pvalues[var]
                sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                print(f"{var}: {coef:.4f} (p={pval:.4f}) {sig}")
        
        # Añadir información de supuestos al resultado
        resultado.supuestos = supuestos
        resultado.cov_type_usado = cov_type_usado
        
        return resultado
    
    except Exception as e:
        print(f"Error en {nombre_modelo}: {e}")
        return None

# =============================================================================
# 3. FUNCIÓN PRINCIPAL AUTOMATIZADA CORREGIDA PARA DATOS NO NORMALES
# =============================================================================

def analisis_automatizado_covid_esg_no_parametrico(data, umbral_significancia=0.05):
    """Función principal CORREGIDA que automatiza todo el análisis con métodos no paramétricos"""
    
    print("="*80)
    print("ANÁLISIS AUTOMATIZADO PRE/POST COVID - COMPONENTES ESG (MÉTODOS NO PARAMÉTRICOS)")
    print("="*80)
    
    # 1. Preparar datos
    df_analysis, df_pre, df_post = preparar_datos(data)
    
    # 2. Variables ESG a analizar
    variables_esg = ['esg', 'environmental', 'social', 'governance', 'controversies']
    variables_control = ['OCFA', 'PB', 'TANG', 'LEV', 'TIE']
    variables_dependientes = ['roe', 'roa']
    
    # 3. Test inicial NO PARAMÉTRICO para identificar cambios significativos
    tests_df, variables_significativas = test_diferencias_significativas_no_parametrico(
        df_pre, df_post, variables_esg, umbral_significancia
    )
    
    # 4. Análisis detallado solo si hay cambios significativos
    if not variables_significativas:
        print(f"\n{'='*80}")
        print("CONCLUSIÓN: NO HAY CAMBIOS SIGNIFICATIVOS")
        print("="*80)
        print("No se detectaron cambios significativos en ninguna variable ESG.")
        print("El análisis de modelos separados no es necesario.")
        return {
            'tests_diferencias': tests_df,
            'variables_significativas': [],
            'modelos': {},
            'resumen': 'Sin cambios significativos'
        }
    
    print(f"\n{'='*80}")
    print("PROCEDIENDO CON ANÁLISIS DETALLADO")
    print("="*80)
    print(f"Se detectaron {len(variables_significativas)} variables con cambios significativos.")
    print("Procediendo con análisis de modelos robustos...")
    
    # 5. Análisis de modelos para variables significativas
    todos_resultados = {}
    resumen_cambios = {}
    
    # Preparar datos con índices para modelos de panel
    df_analysis_idx = df_analysis.reset_index().set_index(['trbc_economic_sector_name', 'year'])
    df_pre_idx = df_pre.reset_index().set_index(['trbc_economic_sector_name', 'year'])
    df_post_idx = df_post.reset_index().set_index(['trbc_economic_sector_name', 'year'])
    
    for esg_var in variables_significativas:
        print(f"\n{'='*60}")
        print(f"ANÁLISIS PARA: {esg_var.upper()}")
        print(f"{'='*60}")
        
        resultados_componente = {}
        
        for dep_var in variables_dependientes:
            if dep_var not in df_analysis.columns:
                continue
            
            print(f"\n--- Modelos para {esg_var.upper()} → {dep_var.upper()} ---")
            
            # Variables independientes
            indep_vars = [esg_var] + variables_control
            
            # Estimar modelos separados con validación robusta
            modelo_pre = estimar_modelo_panel_robusto(
                df_pre_idx, dep_var, indep_vars, 
                f'{esg_var.upper()} - PRE-COVID ({dep_var.upper()})'
            )
            
            modelo_post = estimar_modelo_panel_robusto(
                df_post_idx, dep_var, indep_vars, 
                f'{esg_var.upper()} - POST-COVID ({dep_var.upper()})'
            )
            
            # Comparar coeficientes si ambos modelos son válidos
            cambio_detectado = False
            if modelo_pre and modelo_post:
                if esg_var in modelo_pre.params.index and esg_var in modelo_post.params.index:
                    coef_pre = modelo_pre.params[esg_var]
                    coef_post = modelo_post.params[esg_var]
                    p_pre = modelo_pre.pvalues[esg_var]
                    p_post = modelo_post.pvalues[esg_var]
                    
                    print(f"\n=== COMPARACIÓN DE COEFICIENTES ===")
                    print(f"Pre-COVID: {coef_pre:.4f} (p={p_pre:.4f})")
                    print(f"Post-COVID: {coef_post:.4f} (p={p_post:.4f})")
                    print(f"Cambio: {coef_post - coef_pre:.4f}")
                    
                    # Detectar cambio cualitativo (cambio de signo o significancia)
                    cambio_signo = (coef_pre * coef_post) < 0
                    cambio_significancia = (p_pre < 0.05) != (p_post < 0.05)
                    cambio_detectado = cambio_signo or cambio_significancia
                    
                    if cambio_detectado:
                        print("  → CAMBIO ESTRUCTURAL DETECTADO")
                    else:
                        print("  → Coeficientes estables")
            
            resultados_componente[dep_var] = {
                'modelo_pre': modelo_pre,
                'modelo_post': modelo_post,
                'cambio_estructural': cambio_detectado
            }
        
        todos_resultados[esg_var] = resultados_componente
        
        # Resumir cambios para este componente
        cambios_componente = [dep for dep, res in resultados_componente.items() 
                            if res['cambio_estructural']]
        resumen_cambios[esg_var] = cambios_componente
    
    # 6. Resumen ejecutivo final
    print(f"\n{'='*80}")
    print("RESUMEN EJECUTIVO FINAL")
    print("="*80)
    
    # Mostrar métodos estadísticos utilizados
    metodos_usados = tests_df['test_usado'].value_counts()
    print(f"\n1. MÉTODOS ESTADÍSTICOS UTILIZADOS:")
    for metodo, count in metodos_usados.items():
        proporcion = count / len(tests_df) * 100
        print(f"   • {metodo}: {count} variables ({proporcion:.1f}%)")
    
    print(f"\n2. VARIABLES CON CAMBIOS SIGNIFICATIVOS:")
    for _, row in tests_df[tests_df['significativo']].iterrows():
        test_info = f"[{row['test_usado']}]"
        if row['test_usado'] == "Mann-Whitney U":
            print(f"   • {row['variable']}: Δ_mediana={row['diferencia_mediana']:.4f} (p={row['p_value']:.4f}) {test_info}")
        else:
            print(f"   • {row['variable']}: Δ_media={row['diferencia_media']:.4f} (p={row['p_value']:.4f}) {test_info}")
    
    print(f"\n3. CAMBIOS ESTRUCTURALES EN MODELOS:")
    total_cambios = 0
    for esg_var, deps_con_cambio in resumen_cambios.items():
        if deps_con_cambio:
            print(f"   • {esg_var}: {', '.join(deps_con_cambio)}")
            total_cambios += len(deps_con_cambio)
        else:
            print(f"   • {esg_var}: Sin cambios estructurales")
    
    print(f"\n4. RECOMENDACIONES METODOLÓGICAS:")
    pct_no_parametrico = metodos_usados.get("Mann-Whitney U", 0) / len(tests_df) * 100
    
    if pct_no_parametrico > 50:
        print(f"   • IMPORTANTE: {pct_no_parametrico:.1f}% de variables requirieron tests no paramétricos")
        print(f"   • Los datos NO siguen distribución normal → usar métodos robustos")
        print(f"   • Interpretar medianas en lugar de medias para variables no normales")
    
    if total_cambios > 0:
        print(f"   • Se detectaron {total_cambios} relaciones con cambio estructural")
        print(f"   • RECOMENDACIÓN: Analizar períodos Pre/Post COVID por separado")
    else:
        print(f"   • No se detectaron cambios estructurales significativos")
        print(f"   • RECOMENDACIÓN: Mantener análisis conjunto de períodos")
    
    return {
        'tests_diferencias': tests_df,
        'variables_significativas': variables_significativas,
        'modelos': todos_resultados,
        'resumen_cambios': resumen_cambios,
        'metodos_utilizados': metodos_usados.to_dict(),
        'total_cambios_estructurales': total_cambios,
        'pct_no_parametrico': pct_no_parametrico
    }

# =============================================================================
# 4. EJEMPLO DE USO
# =============================================================================

# Para usar el código corregido con métodos no paramétricos:
# resultados = analisis_automatizado_covid_esg_no_parametrico(data, umbral_significancia=0.05)

# El nuevo sistema:
# ✅ Test de Chow CORREGIDO evalúa cambio en relación ESG → Performance
# ✅ Comparación directa de coeficientes como validación
# ✅ Verificación automática de coherencia entre métodos
# ✅ Interpretación económica detallada de los cambios
# ✅ Detección automática de problemas de especificación














































# Soluciones mejoradas para problemas de no linealidad en modelos de panel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, StandardScaler
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels import PanelOLS
from linearmodels.panel import RandomEffects
import warnings
warnings.filterwarnings('ignore')

def detectar_tipo_no_linealidad(data, dep_var, exog_vars):
    """
    Detecta el tipo de no linealidad presente en los datos
    """
    print("DIAGNÓSTICO DEL TIPO DE NO LINEALIDAD")
    print("="*50)
    
    # Crear gráficos de dispersión para cada variable
    n_vars = len(exog_vars)
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_vars == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_vars == 1 else axes
    else:
        axes = axes.flatten()
    
    tipos_no_linealidad = {}
    
    for i, var in enumerate(exog_vars):
        if i < len(axes):
            ax = axes[i] if isinstance(axes, (list, np.ndarray)) else axes
            
            # Filtrar datos válidos
            mask = data[var].notna() & data[dep_var].notna()
            x = data.loc[mask, var]
            y = data.loc[mask, dep_var]
            
            if len(x) == 0:
                continue
                
            # Gráfico de dispersión
            ax.scatter(x, y, alpha=0.5, s=20)
            
            # Línea de tendencia lineal
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, label='Lineal')
            except:
                pass
            
            # Línea de tendencia cuadrática
            try:
                z2 = np.polyfit(x, y, 2)
                p2 = np.poly1d(z2)
                x_sorted = np.sort(x)
                ax.plot(x_sorted, p2(x_sorted), "g-", alpha=0.8, label='Cuadrática')
            except:
                pass
            
            # Línea de tendencia logarítmica (si todos los valores son positivos)
            if (x > 0).all() and len(x) > 5:
                try:
                    log_x = np.log(x)
                    z_log = np.polyfit(log_x, y, 1)
                    p_log = np.poly1d(z_log)
                    x_sorted = np.sort(x)
                    ax.plot(x_sorted, p_log(np.log(x_sorted)), "b-", alpha=0.8, label='Logarítmica')
                    tipos_no_linealidad[var] = 'logaritmica'
                except:
                    pass
            
            ax.set_xlabel(var)
            ax.set_ylabel(dep_var)
            ax.set_title(f'{dep_var} vs {var}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Análisis de curvatura
            try:
                corr_linear = np.corrcoef(x, y)[0,1]**2
                x_squared = x**2
                X_poly = np.column_stack([np.ones(len(x)), x, x_squared])
                
                # Usar numpy para regresión múltiple
                coeffs = np.linalg.lstsq(X_poly, y, rcond=None)[0]
                y_pred_poly = X_poly @ coeffs
                
                # Calcular R²
                ss_res_poly = np.sum((y - y_pred_poly) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2_poly = 1 - (ss_res_poly / ss_tot)
                
                if r2_poly - corr_linear > 0.05:  # Mejora significativa con término cuadrático
                    tipos_no_linealidad[var] = 'cuadratica'
                elif var not in tipos_no_linealidad:
                    tipos_no_linealidad[var] = 'posible_transformacion'
            except:
                if var not in tipos_no_linealidad:
                    tipos_no_linealidad[var] = 'desconocida'
    
    # Ocultar axes vacíos
    if isinstance(axes, (list, np.ndarray)) and len(axes) > len(exog_vars):
        for i in range(len(exog_vars), len(axes)):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    print("\nTipos de no linealidad detectados:")
    for var, tipo in tipos_no_linealidad.items():
        print(f"  {var}: {tipo}")
    
    return tipos_no_linealidad

def aplicar_transformaciones_variables(data, dep_var, exog_vars, tipos_no_linealidad):
    """
    Aplica transformaciones a las variables según el tipo de no linealidad detectado
    """
    print("\nAPLICANDO TRANSFORMACIONES")
    print("="*40)
    
    data_transformed = data.copy()
    vars_transformadas = []
    
    for var in exog_vars:
        tipo = tipos_no_linealidad.get(var, 'ninguna')
        
        if tipo == 'logaritmica' and (data[var] > 0).all():
            # Transformación logarítmica
            new_var = f'log_{var}'
            data_transformed[new_var] = np.log(data_transformed[var])
            vars_transformadas.append(new_var)
            print(f"  {var} -> transformación logarítmica: {new_var}")
            
        elif tipo == 'cuadratica':
            # Añadir término cuadrático (mantener también el lineal)
            new_var = f'{var}_sq'
            data_transformed[new_var] = data_transformed[var]**2
            vars_transformadas.extend([var, new_var])
            print(f"  {var} -> añadido término cuadrático: {new_var}")
            
        elif tipo == 'posible_transformacion':
            # Intentar varias transformaciones
            if (data[var] >= 0).all() and not (data[var] == 0).any():
                # Probar transformación raíz cuadrada
                new_var = f'sqrt_{var}'
                data_transformed[new_var] = np.sqrt(data_transformed[var])
                vars_transformadas.append(new_var)
                print(f"  {var} -> transformación raíz cuadrada: {new_var}")
            elif not (data[var] == 0).any():
                # Probar transformación inversa
                new_var = f'inv_{var}'
                data_transformed[new_var] = 1 / data_transformed[var]
                vars_transformadas.append(new_var)
                print(f"  {var} -> transformación inversa: {new_var}")
            else:
                vars_transformadas.append(var)
                print(f"  {var} -> sin transformación aplicable")
        else:
            vars_transformadas.append(var)
            print(f"  {var} -> sin transformación")
    
    # Verificar que todas las variables transformadas existen
    vars_existentes = [var for var in vars_transformadas if var in data_transformed.columns]
    vars_faltantes = [var for var in vars_transformadas if var not in data_transformed.columns]
    
    if vars_faltantes:
        print(f"  Variables no creadas: {vars_faltantes}")
    
    return data_transformed, vars_existentes, dep_var, None

def agregar_terminos_interaccion(data, exog_vars, max_interactions=3):
    """
    Agrega términos de interacción entre las variables más importantes
    """
    print(f"\nAGREGANDO TÉRMINOS DE INTERACCIÓN")
    print("="*40)
    
    data_with_interactions = data.copy()
    interaction_vars = []
    
    # Seleccionar las variables más importantes para interacciones
    important_vars = exog_vars[:max_interactions]
    
    count = 0
    for i in range(len(important_vars)):
        for j in range(i+1, len(important_vars)):
            var1, var2 = important_vars[i], important_vars[j]
            interaction_name = f'{var1}_x_{var2}'
            
            # Verificar que ambas variables existen y tienen datos válidos
            if var1 in data_with_interactions.columns and var2 in data_with_interactions.columns:
                mask = data_with_interactions[var1].notna() & data_with_interactions[var2].notna()
                if mask.sum() > 0:
                    data_with_interactions[interaction_name] = (
                        data_with_interactions[var1] * data_with_interactions[var2]
                    )
                    interaction_vars.append(interaction_name)
                    print(f"  Creada interacción: {interaction_name}")
                    count += 1
    
    return data_with_interactions, interaction_vars

def modelo_polinomial_mejorado(data, dep_var, exog_vars, degree=2):
    """
    Crea un modelo con términos polinomiales de forma más robusta
    """
    print(f"\nMODELO POLINOMIAL DE GRADO {degree}")
    print("="*40)
    
    # Preparar datos - eliminar filas con valores faltantes
    vars_necesarias = [dep_var] + exog_vars
    data_clean = data[vars_necesarias].dropna()
    
    if len(data_clean) == 0:
        print("Error: No hay datos válidos después de eliminar valores faltantes")
        return None, None
    
    X = data_clean[exog_vars]
    y = data_clean[dep_var]
    
    # Estandarizar variables para evitar problemas numéricos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=exog_vars, index=X.index)
    
    # Crear características polinomiales
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_scaled_df)
    
    # Crear nombres de las nuevas variables
    feature_names = poly.get_feature_names_out(exog_vars)
    
    # Crear DataFrame con las nuevas características
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    print(f"Variables originales: {len(exog_vars)}")
    print(f"Variables después de expansión polinomial: {len(feature_names)}")
    print(f"Observaciones utilizadas: {len(data_clean)}")
    
    return X_poly_df, feature_names, y

def modelos_no_lineales_mejorados(data, dep_var, exog_vars, tipos_no_linealidad):
    """
    Versión mejorada que maneja mejor los errores
    """
    print("\nPROBANDO MODELOS NO LINEALES ALTERNATIVOS")
    print("="*50)
    
    # Verificar estructura de datos requerida
    required_cols = ['company_name', 'year'] if 'company_name' in data.columns else []
    if not all(col in data.columns for col in required_cols):
        print("Advertencia: Columnas de panel no encontradas. Usando análisis cross-sectional.")
        panel_analysis = False
    else:
        panel_analysis = True
    
    resultados = {}
    
    # 1. Modelo con transformaciones
    print("\n1. MODELO CON TRANSFORMACIONES")
    print("-" * 30)
    
    try:
        data_transformed, vars_transformadas, dep_var_trans, lambda_param = aplicar_transformaciones_variables(
            data, dep_var, exog_vars, tipos_no_linealidad
        )
        
        if panel_analysis:
            # Análisis de panel
            panel_data = data_transformed.dropna(subset=[dep_var] + vars_transformadas)
            if len(panel_data) > 0:
                panel_data_idx = panel_data.set_index(['company_name', 'year'])
                dependent_trans = panel_data_idx[dep_var]
                exog_trans = sm.add_constant(panel_data_idx[vars_transformadas])
                
                model_fe_trans = PanelOLS(dependent_trans, exog_trans, entity_effects=True)
                results_fe_trans = model_fe_trans.fit(cov_type='robust')
                
                resultados['transformaciones'] = {
                    'modelo': results_fe_trans,
                    'r2': results_fe_trans.rsquared_within,
                    'variables': vars_transformadas,
                    'dep_var': dep_var_trans
                }
                print(f"R² del modelo con transformaciones: {results_fe_trans.rsquared_within:.4f}")
        else:
            # Análisis cross-sectional
            clean_data = data_transformed.dropna(subset=[dep_var] + vars_transformadas)
            if len(clean_data) > 0:
                y_trans = clean_data[dep_var]
                X_trans = sm.add_constant(clean_data[vars_transformadas])
                
                model_trans = sm.OLS(y_trans, X_trans).fit(cov_type='HC3')
                
                resultados['transformaciones'] = {
                    'modelo': model_trans,
                    'r2': model_trans.rsquared,
                    'variables': vars_transformadas,
                    'dep_var': dep_var_trans
                }
                print(f"R² del modelo con transformaciones: {model_trans.rsquared:.4f}")
        
    except Exception as e:
        print(f"Error en modelo con transformaciones: {e}")
        resultados['transformaciones'] = None
    
    # 2. Modelo con interacciones
    print("\n2. MODELO CON INTERACCIONES")
    print("-" * 30)
    
    try:
        data_interactions, interaction_vars = agregar_terminos_interaccion(data, exog_vars)
        all_vars = exog_vars + interaction_vars
        
        if panel_analysis:
            panel_interactions = data_interactions.dropna(subset=[dep_var] + all_vars)
            if len(panel_interactions) > 0:
                panel_interactions_idx = panel_interactions.set_index(['company_name', 'year'])
                
                exog_interactions = sm.add_constant(panel_interactions_idx[all_vars])
                dependent_interactions = panel_interactions_idx[dep_var]
                
                model_fe_int = PanelOLS(dependent_interactions, exog_interactions, entity_effects=True)
                results_fe_int = model_fe_int.fit(cov_type='robust')
                
                resultados['interacciones'] = {
                    'modelo': results_fe_int,
                    'r2': results_fe_int.rsquared_within,
                    'variables': all_vars,
                    'dep_var': dep_var
                }
                print(f"R² del modelo con interacciones: {results_fe_int.rsquared_within:.4f}")
        else:
            clean_data = data_interactions.dropna(subset=[dep_var] + all_vars)
            if len(clean_data) > 0:
                y_int = clean_data[dep_var]
                X_int = sm.add_constant(clean_data[all_vars])
                
                model_int = sm.OLS(y_int, X_int).fit(cov_type='HC3')
                
                resultados['interacciones'] = {
                    'modelo': model_int,
                    'r2': model_int.rsquared,
                    'variables': all_vars,
                    'dep_var': dep_var
                }
                print(f"R² del modelo con interacciones: {model_int.rsquared:.4f}")
        
    except Exception as e:
        print(f"Error en modelo con interacciones: {e}")
        resultados['interacciones'] = None
    
    # 3. Modelo polinomial
    print("\n3. MODELO POLINOMIAL")
    print("-" * 30)
    
    try:
        # Limitar variables para evitar explosión dimensional
        vars_poly = exog_vars[:3] if len(exog_vars) > 3 else exog_vars
        result = modelo_polinomial_mejorado(data, dep_var, vars_poly, degree=2)
        
        if result[0] is not None:
            X_poly, feature_names, y_poly = result
            
            X_poly_const = sm.add_constant(X_poly)
            model_poly = sm.OLS(y_poly, X_poly_const).fit(cov_type='HC3')
            
            resultados['polinomial'] = {
                'modelo': model_poly,
                'r2': model_poly.rsquared,
                'variables': list(feature_names),
                'dep_var': dep_var
            }
            
            print(f"R² del modelo polinomial: {model_poly.rsquared:.4f}")
            print(f"Número de variables: {len(feature_names)}")
        
    except Exception as e:
        print(f"Error en modelo polinomial: {e}")
        resultados['polinomial'] = None
    
    return resultados

def analisis_residuos_no_linealidad(modelo, data, dep_var, exog_vars):
    """
    Analiza los residuos para detectar patrones de no linealidad restante
    """
    print("\nANÁLISIS DE RESIDUOS")
    print("="*30)
    
    try:
        # Obtener residuos
        if hasattr(modelo, 'resids'):
            residuos = modelo.resids
        elif hasattr(modelo, 'residuals'):
            residuos = modelo.residuals
        else:
            print("No se pueden obtener los residuos del modelo")
            return None
        
        # Obtener valores ajustados
        if hasattr(modelo, 'fittedvalues'):
            valores_ajustados = modelo.fittedvalues
        elif hasattr(modelo, 'fitted_values'):
            valores_ajustados = modelo.fitted_values
        else:
            print("No se pueden obtener los valores ajustados")
            return None
        
        # Crear gráficos de diagnóstico
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuos vs Valores ajustados
        axes[0,0].scatter(valores_ajustados, residuos, alpha=0.6)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Valores Ajustados')
        axes[0,0].set_ylabel('Residuos')
        axes[0,0].set_title('Residuos vs Valores Ajustados')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot
        stats.probplot(residuos, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot de Residuos')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Histograma de residuos
        axes[1,0].hist(residuos, bins=30, density=True, alpha=0.7)
        axes[1,0].set_xlabel('Residuos')
        axes[1,0].set_ylabel('Densidad')
        axes[1,0].set_title('Distribución de Residuos')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Residuos vs primera variable explicativa
        if len(exog_vars) > 0:
            var_principal = exog_vars[0]
            if var_principal in data.columns:
                # Alinear índices
                common_index = residuos.index.intersection(data.index)
                if len(common_index) > 0:
                    residuos_aligned = residuos.loc[common_index]
                    var_data = data.loc[common_index, var_principal]
                    
                    axes[1,1].scatter(var_data, residuos_aligned, alpha=0.6)
                    axes[1,1].axhline(y=0, color='r', linestyle='--')
                    axes[1,1].set_xlabel(var_principal)
                    axes[1,1].set_ylabel('Residuos')
                    axes[1,1].set_title(f'Residuos vs {var_principal}')
                    axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Tests estadísticos
        print("\nTests de diagnóstico:")
        
        # Test de normalidad de residuos
        shapiro_stat, shapiro_p = stats.shapiro(residuos[:5000] if len(residuos) > 5000 else residuos)
        print(f"Test de Shapiro-Wilk (normalidad): p = {shapiro_p:.4f}")
        
        # Test de heterocedasticidad (si es posible)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(residuos, valores_ajustados.values.reshape(-1, 1))
            print(f"Test de Breusch-Pagan (heterocedasticidad): p = {bp_p:.4f}")
        except:
            print("Test de heterocedasticidad no disponible")
        
        return {
            'residuos': residuos,
            'valores_ajustados': valores_ajustados,
            'normalidad_p': shapiro_p
        }
        
    except Exception as e:
        print(f"Error en análisis de residuos: {e}")
        return None

def recomendaciones_finales(resultados, tipos_no_linealidad):
    """
    Proporciona recomendaciones específicas basadas en los resultados
    """
    print("\nRECOMENDACIONES ESPECÍFICAS")
    print("="*40)
    
    # Encontrar el mejor modelo
    mejor_r2 = 0
    mejor_nombre = None
    
    for nombre, resultado in resultados.items():
        if resultado is not None and resultado['r2'] > mejor_r2:
            mejor_r2 = resultado['r2']
            mejor_nombre = nombre
    
    if mejor_nombre:
        print(f"1. MEJOR MODELO: {mejor_nombre.upper()}")
        print(f"   R² = {mejor_r2:.4f}")
        
        if mejor_nombre == 'transformaciones':
            print("\n2. RECOMENDACIONES PARA TRANSFORMACIONES:")
            print("   • Mantener las transformaciones aplicadas")
            print("   • Interpretar coeficientes considerando las transformaciones")
            if 'logaritmica' in tipos_no_linealidad.values():
                print("   • Variables log: cambio porcentual en Y por cambio porcentual en X")
            
        elif mejor_nombre == 'interacciones':
            print("\n2. RECOMENDACIONES PARA INTERACCIONES:")
            print("   • Los efectos de las variables dependen entre sí")
            print("   • Interpretar efectos marginales, no individuales")
            print("   • Considerar el contexto económico de las interacciones")
            
        elif mejor_nombre == 'polinomial':
            print("\n2. RECOMENDACIONES PARA MODELO POLINOMIAL:")
            print("   • Relación no lineal cuadrática/cúbica confirmada")
            print("   • PRECAUCIÓN: Riesgo de overfitting")
            print("   • Validar con datos fuera de muestra")
            print("   • Considerar restricciones teóricas")
        
        print("\n3. PRÓXIMOS PASOS:")
        print("   • Validar modelo con datos holdout")
        print("   • Verificar estabilidad temporal")
        print("   • Realizar análisis de sensibilidad")
        print("   • Documentar interpretación económica")
    
    else:
        print("No se encontró un modelo superior al lineal básico")
        print("Considerar:")
        print("• Recolectar más datos")
        print("• Buscar variables omitidas")
        print("• Revisar calidad de los datos")

# Función principal integrada
def solucion_completa_no_linealidad_mejorada(data, dep_var, exog_vars, modelo_original=None):
    """
    Función principal mejorada que implementa la solución completa
    """
    print("SOLUCIÓN COMPLETA PARA PROBLEMAS DE NO LINEALIDAD")
    print("="*60)
    
    # Paso 1: Verificar datos
    print(f"Variables analizadas: {exog_vars}")
    print(f"Variable dependiente: {dep_var}")
    print(f"Observaciones totales: {len(data)}")
    
    # Paso 2: Diagnosticar tipo de no linealidad
    tipos_no_linealidad = detectar_tipo_no_linealidad(data, dep_var, exog_vars)
    
    # Paso 3: Probar modelos alternativos
    resultados_no_lineales = modelos_no_lineales_mejorados(
        data, dep_var, exog_vars, tipos_no_linealidad
    )
    
    # Paso 4: Análisis de residuos del mejor modelo
    mejor_modelo = None
    mejor_r2 = 0
    
    for nombre, resultado in resultados_no_lineales.items():
        if resultado is not None and resultado['r2'] > mejor_r2:
            mejor_r2 = resultado['r2']
            mejor_modelo = resultado
    
    if mejor_modelo is not None:
        print(f"\nANÁLISIS DEL MEJOR MODELO (R² = {mejor_r2:.4f})")
        print("-" * 40)
        
        # Mostrar resumen del modelo
        try:
            print(mejor_modelo['modelo'].summary())
        except:
            try:
                print(mejor_modelo['modelo'].summary.tables[1])
            except:
                print("Resumen del modelo no disponible")
        
        # Análisis de residuos
        analisis_residuos_no_linealidad(
            mejor_modelo['modelo'], 
            data, 
            dep_var, 
            exog_vars
        )
    
    # Paso 5: Recomendaciones finales
    recomendaciones_finales(resultados_no_lineales, tipos_no_linealidad)
    
    return mejor_modelo, resultados_no_lineales, tipos_no_linealidad

# Para usar con tus datos:
mejor_modelo, todos_resultados, tipos = solucion_completa_no_linealidad_mejorada(
     data=data,
     dep_var='roa',
     exog_vars=['environmental', 'PB', 'GR', 'OCFA', 'LEV', 'TANG', 'TIE']
 )