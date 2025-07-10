# TFM
Código utilizado para la realización del Trabajo de Fin de Máster en Análisis Avanzado de Datos Multivariantes y Big Data "Estudio de la sostenibilidad del EuroStoxx 50: un enfoque basado en minería de texto y análisis multivariante"

El flujo de trabajo sigue los siguientes pasos:

## Estructura del repositorio

1. **`analisis_exploratorio.py`**  
   Realiza un análisis preliminar de los datos textuales extraídos de los informes. Se exploran frecuencias, distribución de términos y métricas básicas.

2. **`analisis_panel.py`**  
   Análisis econométrico con estructura de panel entre variables ESG y rendimientos financier.

3. **`pasar_pdf_txt.py`**  
   Script de extracción: convierte informes en formato PDF a texto plano para su posterior procesamiento.

4. **`limpieza_reportes.py`**  
   Limpieza y preprocesamiento de los textos: eliminación de caracteres no deseados, normalización, tokenización, etc.

5. **`results_bertopic.py`**  
   Aplicación del modelo BERTopic para extraer tópicos latentes de los informes y primera parte del análisis multivariante.

6. **`biplot_bertopic.R`**  
   Script en R para la representación gráfica tipo *biplot* de los tópicos generados por BERTopic, proporcionando una vista complementaria de los resultados.

## Requisitos

- Python 3.8+
- R (para `biplot_bertopic.R`)
- Paquetes:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `sklearn`, `scipy`, `nltk`, `bertopic`, etc.
- Para el script en R: `ggplot2`, `MultBiplotR`, `dyplyr`, etc.

## Uso

1. Extraer los textos desde PDF con `pasar_pdf_txt.py`
2. Limpiar los textos con `limpieza_reportes.py`
3. Realizar análisis preliminar con `analisis_exploratorio.py`
4. Realizar el análisis de panel entre métricas ESG y los rendimientos financieros con `analisis_panel.py`
5. Ejecutar `results_bertopic.py` para generar y visualizar los temas, la clusterización y las correlaciones
6. Visualizar los resultados del PCA, HJ-Biplots y las diferencias entre temas con `biplot_bertopic.R`

