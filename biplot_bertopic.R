library(MultBiplotR)
library(readxl)
library(dplyr) 
# Cargar datos
library(readr)
data <- read_csv("C:/Users/laura/Desktop/REPORTES/RESULTADOS/importance_matrix_relative_frequency_with_info.csv")
data <- data[,-c(1,12:31)]
data <- data[data$year == 2023, ]

# Cargar archivo de puntuaciones ESG
esg_data <- read_excel("C:/Users/laura/Downloads/esg_eu50.xlsx")

# Convertir también la columna de empresa del archivo ESG a minúsculas
esg_data <- esg_data[esg_data$year == 2023, ]
# Unir los datasets
merged_data <- merge(data, esg_data, by.x = "company", by.y = "company", all.x = TRUE)

# Extraer nombres de empresas y preparar matriz de datos
companies <- merged_data$company
X <- merged_data[, -c(1,12,13,14,15,16,17, ncol(merged_data))]  # Eliminar columnas no numéricas y la columna ESG

# 1. ajustar el pca escalando las variables para darles varianza unitaria
pca <- prcomp(X, center = TRUE, scale. = TRUE)

# 2. obtener resumen con varianza explicada por componente
summary(pca)

# 3. extraer autovalores (varianza de cada componente)
eigenvalues <- pca$sdev^2
proportion_var <- eigenvalues / sum(eigenvalues)
cumulative_var <- cumsum(proportion_var)

# 4. mostrar tablas de autovalores y varianza
results <- data.frame(
  componente = paste0("PC", 1:length(eigenvalues)),
  autovalor = round(eigenvalues, 3),
  proporción_var = round(proportion_var * 100, 2),
  varianza_acumulada = round(cumulative_var * 100, 2)
)
print(results)

# 5. inspeccionar cargas de las variables en los componentes
loadings <- round(pca$rotation, 3)
print(loadings)

# 6. recuperar puntuaciones (scores) de las observaciones
scores <- pca$x
head(scores)  # primeras filas de las coordenadas en el espacio PC

# Realizar el análisis biplot con MultBiplotR
biplot <- PCA.Biplot(X, alpha = 2, dimension = 2, Scaling = 5)

# Crear un nuevo dispositivo gráfico
dev.new()

# Obtener coordenadas de filas y columnas
row_coords <- biplot$RowCoordinates
col_coords <- biplot$ColCoordinates

# Calcular límites para el gráfico
xlim <- range(c(row_coords[,1], col_coords[,1])) * 1.1
ylim <- range(c(row_coords[,2], col_coords[,2])) * 1.1

# Crear el marco del gráfico vacío
plot(0, 0, type = "n", xlim = xlim, ylim = ylim,
     xlab = paste("PC1 (", round(biplot$Inertia[1], 1), "%)"),
     ylab = paste("PC2 (", round(biplot$Inertia[2], 1), "%)"),
     main = "PCA Biplot (Dim 1 vs Dim 2)")

# Añadir líneas de referencia
abline(h = 0, v = 0, lty = 2, col = "gray")

# Calcular distancias al origen para determinar empresas significativas
distances <- sqrt(row_coords[,1]^2 + row_coords[,2]^2)
threshold <- quantile(distances, 0)  # Ajustar según necesidad
significant_indices <- which(distances > threshold)

# CORRECCIÓN PRINCIPAL: Colorear puntos según la clasificación ESG
# Verificar que tenemos datos ESG válidos
print("Verificando datos ESG...")
print(paste("Número de empresas en merged_data:", nrow(merged_data)))
print(paste("Número de empresas con ESG válido:", sum(!is.na(merged_data$esg))))

# Filtrar solo empresas con datos ESG válidos
valid_esg_data <- merged_data[!is.na(merged_data$esg), ]

if(nrow(valid_esg_data) > 0) {
  # Ordenar las empresas por la puntuación ESG (menor valor = mejor ESG)
  esg_sorted <- valid_esg_data[order(valid_esg_data$esg), ]
  
  # Calcular cuántas empresas representan el 25%
  n_companies <- nrow(esg_sorted)
  top_25_count <- max(1, floor(0.25 * n_companies))
  bottom_25_count <- max(1, floor(0.25 * n_companies))
  
  # Identificar el 25% con mejor puntuación ESG (valores más bajos) y el 25% con peor puntuación ESG (valores más altos)
  top_25_companies <- esg_sorted[1:top_25_count, "company"]
  bottom_25_companies <- esg_sorted[(n_companies - bottom_25_count + 1):n_companies, "company"]
  
  # Mostrar información de debug
  print(paste("Top 25% ESG companies (", top_25_count, "):", paste(top_25_companies, collapse = ", ")))
  print(paste("Bottom 25% ESG companies (", bottom_25_count, "):", paste(bottom_25_companies, collapse = ", ")))
  
  # Crear vectores de colores usando los nombres exactos de companies
  colors <- rep("gray", length(companies))  # Colores predeterminados en gris
  
  # Asignar colores basándose en la comparación exacta de nombres
  for(i in 1:length(companies)) {
    if(companies[i] %in% top_25_companies) {
      colors[i] <- "green"  # Mejor ESG: verde
    } else if(companies[i] %in% bottom_25_companies) {
      colors[i] <- "red"    # Peor ESG: rojo
    }
  }
  
  # Verificar asignación de colores
  print(paste("Empresas verdes (mejor ESG):", sum(colors == "green")))
  print(paste("Empresas rojas (peor ESG):", sum(colors == "red")))
  print(paste("Empresas grises (medio):", sum(colors == "gray")))
  
} else {
  # Si no hay datos ESG válidos, usar gris para todas
  colors <- rep("gray", length(companies))
  print("No se encontraron datos ESG válidos, usando gris para todas las empresas")
}


# Añadir puntos para todas las empresas, coloreados según ESG
points(row_coords[,1], row_coords[,2], pch = 16, cex = 0.8, col = "black")

# Añadir flechas para las variables (tópicos)
arrows(0, 0, col_coords[,1], col_coords[,2], 
       length = 0.1, col = "black")

# Añadir nombres de las variables
topic_names <- colnames(X)
text(col_coords[,1] * 1.1, col_coords[,2] * 1.1,
     labels = topic_names, 
     cex = 0.7, col = "black")

# Añadir leyenda para ESG
legend("topright", legend = c("Top 25% ESG (mejor)", "Bottom 25% ESG (peor)", "Resto"), 
       col = c("green", "red", "gray"), pch = 16)

# Añadir nombres de empresas
text(row_coords[,1], row_coords[,2], 
     labels = companies, 
     cex = 0.6, pos = 4, col = "blue")


# Realizar el análisis biplot con MultBiplotR, utilizando PC1 y PC3
biplot_pc1_pc3 <- PCA.Biplot(X, alpha = 2, dimension = 3, Scaling = 5)

# Crear un nuevo dispositivo gráfico
dev.new()

# Obtener coordenadas de filas y columnas para PC1 y PC3
row_coords_pc1_pc3 <- biplot_pc1_pc3$RowCoordinates
col_coords_pc1_pc3 <- biplot_pc1_pc3$ColCoordinates

# Calcular límites para el gráfico con PC1 y PC3
xlim_pc1_pc3 <- range(c(row_coords_pc1_pc3[,1], col_coords_pc1_pc3[,1])) * 1.1
ylim_pc1_pc3 <- range(c(row_coords_pc1_pc3[,3], col_coords_pc1_pc3[,3])) * 1.1

# Crear el marco del gráfico vacío con las nuevas dimensiones
plot(0, 0, type = "n", xlim = xlim_pc1_pc3, ylim = ylim_pc1_pc3,
     xlab = paste("PC1 (", round(biplot_pc1_pc3$Inertia[1], 1), "%)"),
     ylab = paste("PC3 (", round(biplot_pc1_pc3$Inertia[3], 1), "%)"),  # CORRECCIÓN: cambié [5] por [3]
     main = "PCA Biplot (Dim 1 vs Dim 3)")

# Añadir líneas de referencia
abline(h = 0, v = 0, lty = 2, col = "gray")

# Calcular distancias al origen para determinar empresas significativas
distances_pc1_pc3 <- sqrt(row_coords_pc1_pc3[,1]^2 + row_coords_pc1_pc3[,3]^2)
threshold_pc1_pc3 <- quantile(distances_pc1_pc3, 0)  # Ajustar según necesidad
significant_indices_pc1_pc3 <- which(distances_pc1_pc3 > threshold_pc1_pc3)

# CORRECCIÓN PRINCIPAL: Colorear puntos según la clasificación ESG usando la lógica corregida
# Verificar que tenemos datos ESG válidos
print("Verificando datos ESG para PC1 vs PC3...")
print(paste("Número de empresas en merged_data:", nrow(merged_data)))
print(paste("Número de empresas con ESG válido:", sum(!is.na(merged_data$esg))))

# Filtrar solo empresas con datos ESG válidos
valid_esg_data <- merged_data[!is.na(merged_data$esg), ]

if(nrow(valid_esg_data) > 0) {
  # Ordenar las empresas por la puntuación ESG (menor valor = mejor ESG)
  esg_sorted <- valid_esg_data[order(valid_esg_data$esg), ]
  
  # Calcular cuántas empresas representan el 25%
  n_companies <- nrow(esg_sorted)
  top_25_count <- max(1, floor(0.25 * n_companies))
  bottom_25_count <- max(1, floor(0.25 * n_companies))
  
  # Identificar el 25% con mejor puntuación ESG (valores más bajos) y el 25% con peor puntuación ESG (valores más altos)
  top_25_companies <- esg_sorted[1:top_25_count, "company"]
  bottom_25_companies <- esg_sorted[(n_companies - bottom_25_count + 1):n_companies, "company"]
  
  # Mostrar información de debug
  print(paste("Top 25% ESG companies (", top_25_count, "):", paste(top_25_companies, collapse = ", ")))
  print(paste("Bottom 25% ESG companies (", bottom_25_count, "):", paste(bottom_25_companies, collapse = ", ")))
  
  # Crear vectores de colores usando los nombres exactos de companies
  colors_pc1_pc3 <- rep("gray", length(companies))  # Colores predeterminados en gris
  
  # Asignar colores basándose en la comparación exacta de nombres
  for(i in 1:length(companies)) {
    if(companies[i] %in% top_25_companies) {
      colors_pc1_pc3[i] <- "green"  # Mejor ESG: verde
    } else if(companies[i] %in% bottom_25_companies) {
      colors_pc1_pc3[i] <- "red"    # Peor ESG: rojo
    }
  }
  
  # Verificar asignación de colores
  print(paste("Empresas verdes (mejor ESG):", sum(colors_pc1_pc3 == "green")))
  print(paste("Empresas rojas (peor ESG):", sum(colors_pc1_pc3 == "red")))
  print(paste("Empresas grises (medio):", sum(colors_pc1_pc3 == "gray")))
  
} else {
  # Si no hay datos ESG válidos, usar gris para todas
  colors_pc1_pc3 <- rep("gray", length(companies))
  print("No se encontraron datos ESG válidos, usando gris para todas las empresas")
}

# Añadir puntos para todas las empresas, coloreados según ESG
points(row_coords_pc1_pc3[,1], row_coords_pc1_pc3[,3], pch = 16, cex = 0.8, col = "black")

# Añadir flechas para las variables (tópicos)
arrows(0, 0, col_coords_pc1_pc3[,1], col_coords_pc1_pc3[,3], 
       length = 0.1, col = "black")

# Añadir nombres de las variables
text(col_coords_pc1_pc3[,1] * 1.1, col_coords_pc1_pc3[,3] * 1.1,
     labels = topic_names, 
     cex = 0.7, col = "black")

# Añadir nombres de empresas
text(row_coords_pc1_pc3[,1], row_coords_pc1_pc3[,3],
     labels = companies, 
     cex = 0.6, pos = 4, col = "blue")

# Añadir leyenda para ESG
legend("topright", legend = c("Top 25% ESG (mejor)", "Bottom 25% ESG (peor)", "Resto"), 
       col = c("green", "red", "gray"), pch = 16)


# Análisis de diferencias en proporciones de tópicos entre empresas con ESG alto y bajo (2023)
# Usando la matriz de importancia relativa de tópicos

library(readr)
library(dplyr)
library(ggplot2)
library(stringr)
library(tidyr)
library(readxl)

# Configurar directorio de trabajo
output_dir <- "C:/Users/laura/Desktop/REPORTES/RESULTADOS"
setwd(output_dir)

# 1. Cargar datos
print("Cargando datos...")

# Cargar matriz de importancia de tópicos
data <- read_csv("C:/Users/laura/Desktop/REPORTES/RESULTADOS/importance_matrix_relative_frequency_with_info.csv")
print(paste("Matriz de importancia cargada:", nrow(data), "filas,", ncol(data), "columnas"))

# Cargar datos ESG
esg_data <- readxl::read_excel("C:/Users/laura/Downloads/esg_eu50.xlsx")
print(paste("Datos ESG cargados:", nrow(esg_data), "filas,", ncol(esg_data), "columnas"))

data <- data[data$year == 2023, ]
esg_data <- esg_data[esg_data$year == 2023, ]
# 2. Preprocesamiento de datos
print("Preprocesando datos...")

# Normalizar nombres de empresas (eliminar espacios y convertir a minúsculas)
normalize_company_names <- function(names) {
  return(str_to_lower(str_replace_all(names, "\\s+", "")))
}

# Normalizar en ambos datasets
data$company_normalized <- normalize_company_names(data$company)
esg_data$company_normalized <- normalize_company_names(esg_data$company)

# Normalizar años
data$year <- as.character(data$year)
esg_data$year <- as.character(esg_data$year)

# Mostrar empresas únicas para verificar
print("Empresas únicas en matriz de importancia:")
print(unique(data$company_normalized))
print("Empresas únicas en datos ESG:")
print(unique(esg_data$company_normalized))

# 3. Filtrar solo datos de 2023
print("Filtrando datos para 2023...")
data_2023 <- data %>% filter(year == "2023")
esg_2023 <- esg_data %>% filter(year == "2023")

print(paste("Registros de 2023 en matriz de importancia:", nrow(data_2023)))
print(paste("Registros de 2023 en datos ESG:", nrow(esg_2023)))

# 4. Fusionar datasets
print("Fusionando datasets...")

# Primera fusión por empresa y año
analysis_df <- data_2023 %>%
  inner_join(esg_2023 %>% select(company_normalized, year, sector, esg), 
             by = c("company_normalized", "year"))

print(paste("Registros tras fusión por empresa y año:", nrow(analysis_df)))

# Si no hay registros, intentar fusionar solo por empresa
if(nrow(analysis_df) == 0) {
  print("No hay registros comunes por empresa y año. Intentando fusión solo por empresa...")
  analysis_df <- data_2023 %>%
    inner_join(esg_2023 %>% select(company_normalized, sector, esg), 
               by = "company_normalized")
  print(paste("Registros tras fusión solo por empresa:", nrow(analysis_df)))
}

if(nrow(analysis_df) == 0) {
  stop("ERROR: No hay registros comunes tras la fusión. Verificar nombres de empresas.")
}

print("Sectores disponibles tras fusión:")
print(unique(analysis_df$sector))

# 5. Identificar columnas de tópicos (formato: solo números como "0", "1", "2", etc.)
print("Identificando columnas de tópicos...")

# Buscar columnas que sean solo números
topic_columns <- names(analysis_df)[str_detect(names(analysis_df), "^\\d+$")]
print(paste("Columnas de tópicos encontradas (formato numérico):", length(topic_columns)))

if(length(topic_columns) > 0) {
  # Ordenar las columnas numéricamente
  topic_columns <- topic_columns[order(as.numeric(topic_columns))]
  print("Primeras 10 columnas de tópicos:")
  print(head(topic_columns, 10))
  print("Últimas 5 columnas de tópicos:")
  print(tail(topic_columns, 5))
} else {
  # Si no encuentra columnas numéricas, mostrar todas las columnas para debug
  print("No se encontraron columnas de tópicos. Todas las columnas disponibles:")
  print(names(analysis_df))
  stop("ERROR: No se pudieron identificar las columnas de tópicos")
}

# 6. Corrección por sector (normalización)
print("Corrigiendo proporciones de tópicos por sector...")

# Calcular ESG corregido por sector
analysis_df <- analysis_df %>%
  group_by(sector) %>%
  mutate(
    sector_avg_esg = mean(esg, na.rm = TRUE),
    esg_corrected = ifelse(sector_avg_esg != 0, 
                           (esg - sector_avg_esg) / sector_avg_esg, 
                           0)
  ) %>%
  ungroup()

# Calcular proporciones de tópicos corregidas por sector
for(topic_col in topic_columns) {
  corrected_col <- paste0(topic_col, "_corrected")
  
  analysis_df <- analysis_df %>%
    group_by(sector) %>%
    mutate(
      sector_avg_topic = mean(!!sym(topic_col), na.rm = TRUE),
      !!corrected_col := ifelse(sector_avg_topic != 0,
                                (!!sym(topic_col) - sector_avg_topic) / sector_avg_topic,
                                0)
    ) %>%
    ungroup() %>%
    select(-sector_avg_topic)
}

# 7. Crear cuartiles de ESG
print("Creando cuartiles de ESG...")

analysis_df <- analysis_df %>%
  arrange(esg_corrected) %>%
  mutate(
    row_number = row_number(),
    total_rows = n(),
    esg_quartile = case_when(
      row_number <= total_rows * 0.25 ~ 0,  # Cuartil más bajo
      row_number <= total_rows * 0.50 ~ 1,
      row_number <= total_rows * 0.75 ~ 2,
      TRUE ~ 3  # Cuartil más alto
    )
  )

# Verificar distribución de cuartiles
quartile_counts <- table(analysis_df$esg_quartile)
print("Distribución de cuartiles:")
print(quartile_counts)

# 8. Análisis de diferencias entre cuartiles extremos
print("Analizando diferencias entre cuartiles extremos...")

# Identificar columnas de tópicos corregidas
corrected_topic_columns <- paste0(topic_columns, "_corrected")

# Filtrar empresas con ESG alto (cuartil 3) y bajo (cuartil 0)
top_performers <- analysis_df %>% filter(esg_quartile == 3)
bottom_performers <- analysis_df %>% filter(esg_quartile == 0)

print(paste("Empresas en cuartil superior (ESG alto):", nrow(top_performers)))
print(paste("Empresas en cuartil inferior (ESG bajo):", nrow(bottom_performers)))

if(nrow(top_performers) > 0 && nrow(bottom_performers) > 0) {
  
  # Calcular promedios por grupo
  top_avg <- top_performers %>%
    select(all_of(corrected_topic_columns)) %>%
    summarise_all(mean, na.rm = TRUE)
  
  bottom_avg <- bottom_performers %>%
    select(all_of(corrected_topic_columns)) %>%
    summarise_all(mean, na.rm = TRUE)
  
  # Calcular diferencias
  differences <- as.numeric(top_avg - bottom_avg)
  
  # Crear DataFrame de resultados
  diff_df <- data.frame(
    Topic = topic_columns,  # Ya son números como caracteres
    Topic_Label = paste("Topic", topic_columns),
    Top_Quartile_Avg = as.numeric(top_avg),
    Bottom_Quartile_Avg = as.numeric(bottom_avg),
    Difference = differences,
    stringsAsFactors = FALSE
  )
  
  # Ordenar por diferencia absoluta
  diff_df$Abs_Difference <- abs(diff_df$Difference)
  diff_df <- diff_df %>% arrange(desc(Abs_Difference))
  
  # Guardar resultados
  write_csv(diff_df, "topic_difference_by_esg_performance_2023_R.csv")
  print("Resultados guardados en: topic_difference_by_esg_performance_2023_R.csv")
  
  # 9. Visualización
  print("Creando visualización...")
  
  # Seleccionar top diferencias para visualizar
  top_positive <- diff_df %>% filter(Difference > 0) %>% head(10)
  top_negative <- diff_df %>% filter(Difference < 0) %>% head(5)
  plot_data <- bind_rows(top_positive, top_negative)
  
  # Crear gráfico
  p <- ggplot(plot_data, aes(x = reorder(Topic_Label, Difference), y = Difference)) +
    geom_col(aes(fill = Difference > 0), width = 0.7) +
    scale_fill_manual(values = c("TRUE" = "green", "FALSE" = "red"), 
                      labels = c("Más en ESG Alto", "Más en ESG Bajo"),
                      name = "") +
    coord_flip() +
    geom_hline(yintercept = 0, color = "black", linetype = "solid", alpha = 0.7) +
    labs(
      title = "Diferencias en proporciones de tópicos entre empresas\ncon alta y baja puntuación ESG (2023)",
      subtitle = "Valores corregidos por sector",
      x = "Tópicos",
      y = "Diferencia en proporciones corregidas por sector",
      caption = paste("Basado en", nrow(top_performers), "empresas ESG alto vs", 
                      nrow(bottom_performers), "empresas ESG bajo")
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 12),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    )
  
  # Guardar gráfico
  ggsave("esg_performance_topic_differences_2023_R.png", 
         plot = p, width = 12, height = 10, dpi = 300)
  print("Gráfico guardado en: esg_performance_topic_differences_2023_R.png")
  
  # 10. Resumen de resultados
  print("\n=== RESUMEN DE RESULTADOS ===")
  print("\nTópicos más prevalentes en empresas con ESG ALTO:")
  top_high_esg <- diff_df %>% 
    filter(Difference > 0) %>% 
    arrange(desc(Difference)) %>% 
    head(5)
  
  for(i in 1:nrow(top_high_esg)) {
    cat(sprintf("- %s: Diferencia = %.3f\n", 
                top_high_esg$Topic_Label[i], 
                top_high_esg$Difference[i]))
  }
  
  print("\nTópicos más prevalentes en empresas con ESG BAJO:")
  top_low_esg <- diff_df %>% 
    filter(Difference < 0) %>% 
    arrange(Difference) %>% 
    head(5)
  
  for(i in 1:nrow(top_low_esg)) {
    cat(sprintf("- %s: Diferencia = %.3f\n", 
                top_low_esg$Topic_Label[i], 
                top_low_esg$Difference[i]))
  }
  
  # 11. Crear informe en markdown
  print("\nCreando informe en markdown...")
  
  report_content <- paste0(
    "# Análisis de Tópicos vs ESG - Año 2023\n\n",
    "## Resumen del Análisis\n\n",
    "Este análisis compara las proporciones de tópicos entre empresas con puntuaciones ESG altas y bajas para el año 2023.\n\n",
    "- **Empresas analizadas**: ", nrow(analysis_df), "\n",
    "- **Empresas ESG alto (cuartil superior)**: ", nrow(top_performers), "\n",
    "- **Empresas ESG bajo (cuartil inferior)**: ", nrow(bottom_performers), "\n",
    "- **Tópicos analizados**: ", length(topic_columns), "\n\n",
    "## Metodología\n\n",
    "1. **Corrección por sector**: Las proporciones de tópicos y puntuaciones ESG se normalizaron por sector\n",
    "2. **Cuartiles**: Las empresas se dividieron en cuartiles basados en ESG corregido\n",
    "3. **Comparación**: Se compararon los cuartiles extremos (superior vs inferior)\n\n",
    "## Resultados Principales\n\n",
    "![Diferencias por rendimiento ESG](esg_performance_topic_differences_2023_R.png)\n\n",
    "### Tópicos más prevalentes en empresas con ESG ALTO:\n\n"
  )
  
  for(i in 1:nrow(top_high_esg)) {
    report_content <- paste0(report_content,
                             "- **", top_high_esg$Topic_Label[i], "**: Diferencia = ", 
                             round(top_high_esg$Difference[i], 3), "\n")
  }
  
  report_content <- paste0(report_content, "\n### Tópicos más prevalentes en empresas con ESG BAJO:\n\n")
  
  for(i in 1:nrow(top_low_esg)) {
    report_content <- paste0(report_content,
                             "- **", top_low_esg$Topic_Label[i], "**: Diferencia = ", 
                             round(top_low_esg$Difference[i], 3), "\n")
  }
  
  # Guardar informe
  writeLines(report_content, "esg_topic_analysis_report_2023_R.md")
  print("Informe guardado en: esg_topic_analysis_report_2023_R.md")
  
} else {
  print("ERROR: No hay suficientes datos en los cuartiles superior e inferior")
  print(paste("Cuartil superior:", nrow(top_performers)))
  print(paste("Cuartil inferior:", nrow(bottom_performers)))
}

print("\n=== ANÁLISIS COMPLETADO ===")
print(paste("Todos los archivos guardados en:", getwd()))
