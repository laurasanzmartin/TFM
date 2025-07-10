import os
import re
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import sent_tokenize

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from difflib import SequenceMatcher

# Ignorar advertencias
warnings.filterwarnings('ignore')

# BERTopic imports
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ========================================
# CONFIGURACIÓN DE SEMILLAS PARA REPRODUCIBILIDAD
# ========================================
RANDOM_SEED = 123

# Establecer semillas para todos los generadores de números aleatorios
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Para reproducibilidad en transformers/sentence-transformers
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f"Using random seed: {RANDOM_SEED} for reproducible results")

# Set output directory
output_dir = r"C:\Users\laura\Desktop\REPORTES\RESULTADOS"
os.makedirs(output_dir, exist_ok=True)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed but we'll continue")

# Set the directory path containing the cleaned text files
data_dir = r"C:\Users\laura\Desktop\REPORTES\REPORTES FINALES\limpios2"

# LISTA AMPLIADA Y MEJORADA DE NOMBRES DE EMPRESAS
COMPANY_NAMES = [
    # Nombres completos de empresas
    "anheuser-busch", "anheuser", "busch", "inbev", "koninklijke", "ahold", "delhaize", 
    "adidas", "adyen", "airbus", "air liquide", "societe anonyme", "allianz", "asml", 
    "holding", "axa", "basf", "bayer", "banco bilbao vizcaya argentaria", "bilbao", 
    "vizcaya", "argentaria", "bbva", "bkw", "bnp paribas", "paribas", "bnp", "danone", 
    "deutsche boerse", "deutsche post", "deutsche telekom", "deutsche", "boerse", 
    "telekom", "enel", "eni", "essilorluxottica", "essilor", "luxottica", "hermes", 
    "iberdrola", "infineon", "technologies", "ing", "groep", "intesa", "sanpaolo", 
    "industria", "diseno", "textil", "inditex", "kone", "lvmh", "moet", "hennessy", 
    "louis", "vuitton", "mercedes benz", "mercedes", "benz", "muenchener", 
    "rueckversicherungs-gesellschaft", "muenchen", "nordea", "nokia", "l'oreal", 
    "oreal", "pernod", "ricard", "kering", "prosus", "ferrari", "safran", "santander", 
    "sap", "sanofi", "schneider", "electric", "vinci", "compagnie", "saint gobain", 
    "gobain", "siemens", "stellantis", "totalenergies", "total", "energies", "vonovia", 
    "volkswagen", "daimler", "bmw", "audi", "porsche", "tesla", "ford", "toyota",
    
    # Nombres específicos que aparecen en tus resultados
    "ubi", "ubi banca", "edp", "vinci", "enel","dior","sephora","zara"
    
    # Variaciones y formas abreviadas
    "ag", "spa", "sa", "se", "nv", "oyj", "plc", "gmbh", "ltd", "llc", "inc", "corp",
    
    # Códigos bursátiles de tu lista
    "abi", "ad", "adsgn", "adyen", "air", "airp", "alvg", "asml", "axaf", "basfn", 
    "baygn", "bbva", "bkwb", "bnpp", "dano", "db1gn", "dhln", "dtegn", "enei", 
    "eni", "eslx", "hrms", "ibe", "ifxgn", "inga", "isp", "itx", "knebv", "lvmh", 
    "mbgn", "muvgn", "ndafi", "nokia", "orep", "perp", "prtp", "prx", "race", 
    "saf", "san", "sapg", "sasy", "schn", "sgef", "sgob", "siegn", "stlam", 
    "ttef", "vnan", "vowg"
]

# Convertir a minúsculas
company_names_lower = list(set([name.lower().strip() for name in COMPANY_NAMES]))

# Combinar todas las listas
all_filters = company_names_lower
all_filters = list(set([f.lower().strip() for f in all_filters if f.strip()]))

def enhanced_company_filter(text):
    """
    Función de filtrado mejorada y más agresiva para eliminar nombres de empresas y números
    """
    if not text or len(text.strip()) < 10:
        return text
    
    # Preservar el texto original para fallback
    original_text = text
    
    # Convertir a minúsculas para procesamiento
    text_lower = text.lower()
    
    # Primero, aplicar filtros de patrones específicos
    # Eliminar números (enteros, decimales, porcentajes)
    text_lower = re.sub(r'\b\d+\.?\d*%?\b', '', text_lower)
    text_lower = re.sub(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', '', text_lower)
    
    # Eliminar años
    text_lower = re.sub(r'\b(?:19|20)\d{2}\b', '', text_lower)
    
    # Eliminar códigos de ticker (2-5 letras mayúsculas)
    text_lower = re.sub(r'\b[A-Z]{2,5}\b', '', text_lower)
    
    # Eliminar URLs y emails
    text_lower = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text_lower)
    text_lower = re.sub(r'\S+@\S+', '', text_lower)
    
    # Tokenizar
    words = text_lower.split()
    filtered_words = []
    
    for word in words:
        # Limpiar palabra (mantener solo letras)
        clean_word = re.sub(r'[^\w]', '', word).strip()
        
        # Saltar si está vacía o es muy corta
        if not clean_word or len(clean_word) < 2:
            continue
            
        # Saltar si es un número
        if clean_word.isdigit():
            continue
            
        # Filtro principal: verificar contra todas nuestras listas
        if clean_word in all_filters:
            continue
            
        # Filtros adicionales por patrones
        # Saltar si parece un código (mezcla de números y letras corta)
        if len(clean_word) <= 6 and any(c.isdigit() for c in clean_word) and any(c.isalpha() for c in clean_word):
            continue
            
        # Saltar si tiene patrones específicos de identificadores de empresa
        if any(pattern in clean_word for pattern in ['ltd', 'inc', 'corp', 'llc', 'gmbh']):
            continue
            
        # Mantener la palabra original (no la limpia) para preservar estructura
        filtered_words.append(word)
    
    # Reconstruir texto
    result = ' '.join(filtered_words)
    
    # Limpieza final
    result = ' '.join(result.split())  # Normalizar espacios
    
    # Si el filtrado eliminó demasiado contenido, usar el original
    if len(result.strip()) < len(original_text.strip()) * 0.3:  # Si perdimos más del 70%
        return original_text
    
    return result if len(result.strip()) > 10 else original_text

# Custom CountVectorizer class para filtrado más agresivo
class ESGCountVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        # Establecer preprocessor personalizado
        kwargs['preprocessor'] = enhanced_company_filter
        super().__init__(**kwargs)
    
    def _word_ngrams(self, tokens, stop_words=None):
        """Override para filtrado adicional en n-gramas"""
        ngrams = super()._word_ngrams(tokens, stop_words)
        
        # Filtrar n-gramas que contengan nombres de empresas
        filtered_ngrams = []
        for ngram in ngrams:
            # Verificar si alguna parte del n-grama está en nuestros filtros
            ngram_words = ngram.split(' ') if ' ' in ngram else [ngram]
            contains_company = any(word.lower() in all_filters for word in ngram_words)
            
            if not contains_company:
                filtered_ngrams.append(ngram)
        
        return filtered_ngrams

# Read all the text files in the directory (ordenados para consistencia)
documents = []
file_names = []

# Ordenar archivos para consistencia
sorted_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])

for file in sorted_files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            text = f.read()
            documents.append(text)
            file_names.append(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")

print(f"Loaded {len(documents)} documents")

# Extract years from filenames if they contain year information
years = []
for file in file_names:
    year_match = re.search(r'20\d{2}', file)
    if year_match:
        years.append(year_match.group())
    else:
        years.append("Unknown")

# Following the paper's approach: use sentence-level analysis
print("Splitting documents into sentences for detailed topic modeling...")
sentences = []
doc_indices = []
year_indices = []

for i, doc in enumerate(documents):
    try:
        doc_sentences = sent_tokenize(doc)
        # Filter sentences that are too short or too long
        filtered_sentences = [
            sent for sent in doc_sentences 
            if 10 <= len(sent.split()) <= 200  # Reasonable sentence length
        ]
        sentences.extend(filtered_sentences)
        doc_indices.extend([i] * len(filtered_sentences))
        year_indices.extend([years[i]] * len(filtered_sentences))
    except Exception as e:
        print(f"Error tokenizing document {i}: {e}")

print(f"Total sentences extracted: {len(sentences)}")

texts_to_process = sentences

print(f"Processing {len(texts_to_process)} sentences")

# Configure BERTopic components following the paper's methodology
print("Configuring BERTopic with enhanced reproducibility...")

# Use a good embedding model with fixed random state
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# UMAP configuration for dimension reduction - TODAS LAS SEMILLAS FIJADAS
umap_model = UMAP(
    n_neighbors=10,      
    n_components=10,     
    min_dist=0.0,
    metric='cosine',
    random_state=RANDOM_SEED,  # Semilla fija
    transform_seed=RANDOM_SEED,  # Semilla adicional para transformaciones
    spread=1.5,
    verbose=False
)

# HDBSCAN configuration - SEMILLA FIJA Y ALGORITMO DETERMINÍSTICO
hdbscan_model = HDBSCAN(
    min_cluster_size=max(10, len(texts_to_process) // 200),  
    min_samples=5,  
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True,
    cluster_selection_epsilon=0.1,
    algorithm='best',  # Más determinístico
    leaf_size=40,      # Valor fijo para consistencia
    core_dist_n_jobs=1  # Un solo proceso para evitar variaciones
)

# VECTORIZADOR MEJORADO con filtrado más agresivo
print("Setting up enhanced vectorizer with aggressive company name filtering...")
vectorizer_model = ESGCountVectorizer(
    max_features=10000,    
    min_df=8,            
    max_df=0.7,          
    stop_words='english',
    ngram_range=(1, 3),  
    token_pattern=r'\b[a-zA-Z]{3,}\b',  
    lowercase=True,
    strip_accents='ascii'
)

# Initial BERTopic model con configuración determinística
print("Creating BERTopic model with enhanced reproducibility...")
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    verbose=True,
    calculate_probabilities=False,
    min_topic_size=15,  
    nr_topics=None      
)

print("Training BERTopic model with deterministic configuration...")
try:
    topics, _ = topic_model.fit_transform(texts_to_process)
    print("✓ Model training successful!")
except Exception as e:
    print(f"Error during training: {e}")
    print("Trying with fallback parameters...")
    
    # Fallback más conservador CON SEMILLAS FIJAS
    hdbscan_fallback = HDBSCAN(
        min_cluster_size=max(8, len(texts_to_process) // 250),  
        min_samples=4,
        metric='euclidean',
        cluster_selection_method='leaf',  
        prediction_data=True,
        algorithm='best',
        leaf_size=40,
        core_dist_n_jobs=1
    )
    
    vectorizer_fallback = ESGCountVectorizer(
        max_features=3000,
        min_df=5,
        max_df=0.8,
        stop_words='english',
        ngram_range=(1, 2),
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        lowercase=True
    )
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_fallback,
        vectorizer_model=vectorizer_fallback,
        verbose=True,
        calculate_probabilities=False,
        min_topic_size=8
    )
    
    topics, _ = topic_model.fit_transform(texts_to_process)

print(f"Initial number of topics found: {len(set(topics))}")

# Post-processing: Filtrar keywords de empresa de los resultados finales
def clean_topic_keywords(topic_words, num_words=10):
    """Limpia las keywords de un topic eliminando nombres de empresas"""
    cleaned_words = []
    
    for word, score in topic_words:
        word_clean = word.lower().strip()
        
        # Verificar si la palabra está en nuestros filtros
        if word_clean not in all_filters:
            # Verificaciones adicionales
            if len(word_clean) >= 3 and not word_clean.isdigit():
                cleaned_words.append((word, score))
        
        if len(cleaned_words) >= num_words:
            break
    
    return cleaned_words

# Forzar reducción a 30 topics si es necesario (DETERMINÍSTICO)
topic_info = topic_model.get_topic_info()
initial_n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)

print(f"Initial topics (excluding outliers): {initial_n_topics}")

if initial_n_topics > 30:
    print(f"Reducing from {initial_n_topics} topics to 30...")
    # Usar reducción determinística
    topic_model.reduce_topics(texts_to_process, nr_topics=30)
    final_topics = topic_model._map_predictions(topics)
    print("✓ Topic reduction completed")
elif initial_n_topics < 30:
    print(f"Found only {initial_n_topics} topics, will work with what we have...")
    final_topics = topics
else:
    print("Perfect! Already have 30 topics.")
    final_topics = topics

# Get final topic information
final_topic_info = topic_model.get_topic_info()
n_final_topics = len(final_topic_info) - 1

print(f"Final number of topics: {n_final_topics}")

# GUARDAR MODELO PARA REPRODUCIBILIDAD FUTURA
model_path = os.path.join(output_dir, 'bertopic_model_reproducible')
topic_model.save(model_path)
print(f"Model saved to {model_path} for future reproducible use")

# Print topic information with enhanced cleaning
print("\nFinal Topic Keywords (with enhanced company name filtering):")
print("=" * 80)

cleaned_topic_summaries = []

for topic_id in range(n_final_topics):
    topic_words = topic_model.get_topic(topic_id)
    if topic_words:
        # Aplicar limpieza adicional a las keywords
        cleaned_topic_words = clean_topic_keywords(topic_words, 15)
        
        if cleaned_topic_words:  # Solo mostrar si quedan palabras después de la limpieza
            keywords = [word for word, score in cleaned_topic_words[:10]]
            topic_count = sum(1 for t in final_topics if t == topic_id)
            
            print(f"Topic {topic_id}:")
            print(f"Keywords: {', '.join(keywords)}")
            print(f"Number of sentences: {topic_count}")
            print(f"Percentage: {topic_count/len(final_topics)*100:.2f}%")
            print("-" * 40)
            
            cleaned_topic_summaries.append({
                'topic_id': topic_id,
                'keywords': ', '.join(keywords),
                'sentence_count': topic_count,
                'percentage': topic_count/len(final_topics)*100
            })

# Save results
print("\nSaving results...")

# Create DataFrame with results
results_df = pd.DataFrame({
    'sentence': texts_to_process,
    'topic': final_topics,
    'document_index': doc_indices,
    'year': year_indices,
    'filename': [file_names[i] for i in doc_indices]
})

# Save to CSV
results_df.to_csv(os.path.join(output_dir, 'topic_modeling_results_reproducible.csv'), index=False)

# Save cleaned topic information
if cleaned_topic_summaries:
    topic_summary_df = pd.DataFrame(cleaned_topic_summaries)
    topic_summary_df.to_csv(os.path.join(output_dir, 'topic_summary_reproducible.csv'), index=False)

# Guardar información de configuración para reproducibilidad
config_info = {
    'random_seed': RANDOM_SEED,
    'n_final_topics': n_final_topics,
    'total_sentences_processed': len(texts_to_process),
    'umap_params': {
        'n_neighbors': 10,
        'n_components': 10,
        'min_dist': 0.0,
        'metric': 'cosine'
    },
    'hdbscan_params': {
        'min_cluster_size': max(10, len(texts_to_process) // 200),
        'min_samples': 5,
        'metric': 'euclidean'
    }
}

import json
with open(os.path.join(output_dir, 'reproducibility_config.json'), 'w') as f:
    json.dump(config_info, f, indent=2)

print(f"Results saved to {output_dir}")
print("Analysis complete with enhanced reproducibility!")

# Create visualization with fixed styling
plt.figure(figsize=(12, 8))
topic_counts = [sum(1 for t in final_topics if t == topic_id) for topic_id in range(n_final_topics)]
plt.bar(range(n_final_topics), topic_counts)
plt.xlabel('Topic ID')
plt.ylabel('Number of Sentences')
plt.title('Distribution of Sentences Across Topics (Reproducible)')
plt.xticks(range(0, n_final_topics, 2))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'topic_distribution_reproducible.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\nTopic distribution visualization saved!")
print(f"\n{'='*60}")
print("REPRODUCIBILITY NOTES:")
print(f"- Random seed used: {RANDOM_SEED}")
print("- Model saved for future use")
print("- Configuration saved in reproducibility_config.json")
print("- To reproduce exactly, use the same random seed and parameters")
print(f"{'='*60}")


def create_topic_similarity_matrix(topic_model, n_topics):
    """
    Crear matriz de similitud entre topics basada en sus vectores de palabras clave
    """
    topic_vectors = []
    
    for topic_id in range(n_topics):
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            vector = np.zeros(len(topic_words))
            for i, (word, score) in enumerate(topic_words):
                vector[i] = score
            topic_vectors.append(vector)
        else:
            topic_vectors.append(np.zeros(100))
    
    # Normalizar vectores al mismo tamaño
    max_len = max(len(v) for v in topic_vectors) if topic_vectors else 100
    normalized_vectors = []
    for vector in topic_vectors:
        if len(vector) < max_len:
            padded_vector = np.pad(vector, (0, max_len - len(vector)), 'constant')
        else:
            padded_vector = vector[:max_len]
        normalized_vectors.append(padded_vector)
    
    # Convertir a array numpy
    normalized_vectors = np.array(normalized_vectors)
    
    # Calcular matriz de similitud coseno
    similarity_matrix = cosine_similarity(normalized_vectors)
    
    # Convertir a matriz de distancia válida
    distance_matrix = 1 - similarity_matrix
    
    # Asegurar que la diagonal sea exactamente cero
    np.fill_diagonal(distance_matrix, 0.0)
    
    # Asegurar simetría
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Asegurar que todos los valores sean no negativos
    distance_matrix = np.maximum(distance_matrix, 0)
    
    return distance_matrix, normalized_vectors

def calculate_cophenetic_correlation(linkage_matrix, distance_matrix):
    """
    Calcular el coeficiente de correlación cofenética
    """
    try:
        # Verificar que la matriz de distancia sea válida
        if not np.allclose(np.diag(distance_matrix), 0):
            np.fill_diagonal(distance_matrix, 0.0)
        
        # Convertir matriz de distancia a formato condensado
        # Solo tomar la parte triangular superior (sin diagonal)
        n = distance_matrix.shape[0]
        condensed_distances = []
        
        for i in range(n):
            for j in range(i + 1, n):
                condensed_distances.append(distance_matrix[i, j])
        
        condensed_distances = np.array(condensed_distances)
        
        # Calcular distancias cofenéticas
        coph_dists = cophenet(linkage_matrix)
        
        # Verificar que ambos arrays tengan la misma longitud
        if len(condensed_distances) != len(coph_dists):
            print(f"Warning: Longitudes diferentes - condensed: {len(condensed_distances)}, coph: {len(coph_dists)}")
            min_len = min(len(condensed_distances), len(coph_dists))
            condensed_distances = condensed_distances[:min_len]
            coph_dists = coph_dists[:min_len]
        
        # Calcular correlación
        if len(condensed_distances) > 1 and len(coph_dists) > 1:
            correlation = np.corrcoef(condensed_distances, coph_dists)[0, 1]
            # Manejar casos donde la correlación es NaN
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return correlation
        
    except Exception as e:
        print(f"Error calculando correlación cofenética: {e}")
        return 0.0

def calculate_rmsstd(data_vectors, cluster_assignments):
    """
    Calcular Root Mean Square Standard Deviation (RMSSTD)
    """
    try:
        total_variance = 0
        n_points = 0
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            cluster_data = np.array([data_vectors[i] for i in cluster_indices])
            
            if len(cluster_data) > 1:
                # Calcular varianza dentro del cluster
                cluster_mean = np.mean(cluster_data, axis=0)
                cluster_variance = np.sum((cluster_data - cluster_mean) ** 2)
                total_variance += cluster_variance
                n_points += len(cluster_data)
        
        if n_points > 0:
            rmsstd = np.sqrt(total_variance / n_points)
        else:
            rmsstd = 0.0
            
        return rmsstd
        
    except Exception as e:
        print(f"Error calculando RMSSTD: {e}")
        return 0.0

def calculate_spr_and_rs(data_vectors, cluster_assignments):
    """
    Calcular Semi-Partial R-squared (SPR) y R-squared (RS)
    """
    try:
        data_array = np.array(data_vectors)
        
        # SST (Total Sum of Squares)
        overall_mean = np.mean(data_array, axis=0)
        sst = np.sum((data_array - overall_mean) ** 2)
        
        # SSE (Sum of Squares Between groups)
        # SSD (Sum of Squares Within groups)
        sse = 0
        ssd = 0
        
        for cluster_id in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]
            cluster_data = data_array[cluster_indices]
            
            if len(cluster_data) > 0:
                cluster_mean = np.mean(cluster_data, axis=0)
                
                # SSE: diferencia entre medias de clusters y media general
                sse += len(cluster_data) * np.sum((cluster_mean - overall_mean) ** 2)
                
                # SSD: varianza dentro del cluster
                ssd += np.sum((cluster_data - cluster_mean) ** 2)
        
        # R-squared
        rs = sse / sst if sst > 0 else 0
        
        # Semi-Partial R-squared (pérdida de homogeneidad relativa)
        spr = ssd / sst if sst > 0 else 0
        
        # Asegurar que los valores estén en rangos válidos
        rs = max(0, min(1, rs))
        spr = max(0, min(1, spr))
        
        return spr, rs, sse, ssd, sst
        
    except Exception as e:
        print(f"Error calculando SPR y RS: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0

def calculate_cluster_distances(linkage_matrix):
    """
    Calcular distancias de aglomeración entre clusters (CD)
    """
    distances = linkage_matrix[:, 2]  # Tercera columna contiene las distancias
    return distances

def evaluate_clustering_metrics(topic_model, cleaned_topic_summaries, output_dir, 
                               max_clusters=15, min_clusters=2):
    """
    Evaluar múltiples métricas para diferentes números de clusters
    """
    print(f"Evaluando clustering para {min_clusters} a {max_clusters} clusters...")
    
    n_topics = len(cleaned_topic_summaries)
    distance_matrix, data_vectors = create_topic_similarity_matrix(topic_model, n_topics)
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Calcular coeficiente cofenético
    coph_corr = calculate_cophenetic_correlation(linkage_matrix, distance_matrix)
    
    # Evaluar diferentes números de clusters
    results = []
    max_possible_clusters = min(max_clusters, n_topics - 1)
    
    for n_clusters in range(min_clusters, max_possible_clusters + 1):
        try:
            cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Verificar que tenemos suficientes clusters únicos
            unique_clusters = len(np.unique(cluster_assignments))
            if unique_clusters < 2:
                continue
            
            # Métricas básicas (con manejo de errores)
            try:
                silhouette = silhouette_score(data_vectors, cluster_assignments, metric='cosine')
            except:
                silhouette = 0.0
                
            try:
                calinski_harabasz = calinski_harabasz_score(data_vectors, cluster_assignments)
            except:
                calinski_harabasz = 0.0
                
            try:
                davies_bouldin = davies_bouldin_score(data_vectors, cluster_assignments)
            except:
                davies_bouldin = float('inf')
            
            # Métricas personalizadas
            rmsstd = calculate_rmsstd(data_vectors, cluster_assignments)
            spr, rs, sse, ssd, sst = calculate_spr_and_rs(data_vectors, cluster_assignments)
            cd_distances = calculate_cluster_distances(linkage_matrix)
            avg_cd = np.mean(cd_distances[:n_clusters-1]) if n_clusters > 1 and len(cd_distances) >= n_clusters-1 else 0
            
            results.append({
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin,
                'rmsstd': rmsstd,
                'spr': spr,
                'rs': rs,
                'avg_cluster_distance': avg_cd,
                'sse': sse,
                'ssd': ssd,
                'sst': sst
            })
            
        except Exception as e:
            print(f"Error evaluando {n_clusters} clusters: {e}")
            continue
    
    if not results:
        print("No se pudieron evaluar clusters. Verificar datos de entrada.")
        return None, coph_corr, 2, linkage_matrix, distance_matrix
    
    results_df = pd.DataFrame(results)
    
    # Guardar resultados
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df.to_csv(os.path.join(output_dir, 'clustering_evaluation_metrics.csv'), index=False)
    
    # Crear visualizaciones
    plot_evaluation_metrics(results_df, output_dir, coph_corr)
    
    # Determinar número óptimo de clusters
    optimal_clusters = determine_optimal_clusters(results_df)
    
    print(f"\n" + "="*80)
    print("EVALUACIÓN DE CLUSTERING - RESULTADOS:")
    print("="*80)
    print(f"Coeficiente de Correlación Cofenética: {coph_corr:.4f}")
    print(f"Número óptimo de clusters recomendado: {optimal_clusters}")
    print(f"Detalles guardados en: {output_dir}")
    
    return results_df, coph_corr, optimal_clusters, linkage_matrix, distance_matrix

def plot_evaluation_metrics(results_df, output_dir, coph_corr):
    """
    Crear visualizaciones de las métricas de evaluación
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Métricas de Evaluación de Clustering\nCoeficiente Cofenético: {coph_corr:.4f}', 
                fontsize=16, fontweight='bold')
    
    # Silhouette Score (mayor es mejor)
    axes[0,0].plot(results_df['n_clusters'], results_df['silhouette_score'], 'bo-', linewidth=2)
    axes[0,0].set_title('Silhouette Score\n(Mayor es mejor)', fontweight='bold')
    axes[0,0].set_xlabel('Número de Clusters')
    axes[0,0].set_ylabel('Silhouette Score')
    axes[0,0].grid(True, alpha=0.3)
    
    # Davies-Bouldin Index (menor es mejor)
    axes[0,1].plot(results_df['n_clusters'], results_df['davies_bouldin'], 'ro-', linewidth=2)
    axes[0,1].set_title('Davies-Bouldin Index\n(Menor es mejor)', fontweight='bold')
    axes[0,1].set_xlabel('Número de Clusters')
    axes[0,1].set_ylabel('Davies-Bouldin Index')
    axes[0,1].grid(True, alpha=0.3)
    
    # RMSSTD (menor es mejor)
    axes[0,2].plot(results_df['n_clusters'], results_df['rmsstd'], 'go-', linewidth=2)
    axes[0,2].set_title('RMSSTD\n(Menor es mejor)', fontweight='bold')
    axes[0,2].set_xlabel('Número de Clusters')
    axes[0,2].set_ylabel('RMSSTD')
    axes[0,2].grid(True, alpha=0.3)
    
    # R-squared (mayor es mejor)
    axes[1,0].plot(results_df['n_clusters'], results_df['rs'], 'mo-', linewidth=2)
    axes[1,0].set_title('R-squared (RS)\n(Mayor es mejor)', fontweight='bold')
    axes[1,0].set_xlabel('Número de Clusters')
    axes[1,0].set_ylabel('R-squared')
    axes[1,0].grid(True, alpha=0.3)
    
    # Semi-Partial R-squared (menor es mejor)
    axes[1,1].plot(results_df['n_clusters'], results_df['spr'], 'co-', linewidth=2)
    axes[1,1].set_title('Semi-Partial R-squared (SPR)\n(Menor es mejor)', fontweight='bold')
    axes[1,1].set_xlabel('Número de Clusters')
    axes[1,1].set_ylabel('SPR')
    axes[1,1].grid(True, alpha=0.3)
    
    # Distancia promedio entre clusters (menor es mejor para clusters bien separados)
    axes[1,2].plot(results_df['n_clusters'], results_df['avg_cluster_distance'], 'yo-', linewidth=2)
    axes[1,2].set_title('Distancia Promedio\nEntre Clusters (CD)', fontweight='bold')
    axes[1,2].set_xlabel('Número de Clusters')
    axes[1,2].set_ylabel('Distancia Promedio')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_evaluation_metrics.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def determine_optimal_clusters(results_df):
    """
    Determinar el número óptimo de clusters basado en múltiples métricas
    """
    # Normalizar métricas para comparación
    normalized_df = results_df.copy()
    
    # Para métricas donde "mayor es mejor"
    for col in ['silhouette_score', 'rs']:
        if col in normalized_df.columns:
            normalized_df[f'{col}_norm'] = (normalized_df[col] - normalized_df[col].min()) / \
                                         (normalized_df[col].max() - normalized_df[col].min())
    
    # Para métricas donde "menor es mejor"
    for col in ['davies_bouldin', 'rmsstd', 'spr', 'avg_cluster_distance']:
        if col in normalized_df.columns:
            normalized_df[f'{col}_norm'] = 1 - ((normalized_df[col] - normalized_df[col].min()) / \
                                              (normalized_df[col].max() - normalized_df[col].min()))
    
    # Calcular puntuación compuesta (promedio de métricas normalizadas)
    metric_cols = [col for col in normalized_df.columns if col.endswith('_norm')]
    normalized_df['composite_score'] = normalized_df[metric_cols].mean(axis=1)
    
    # Encontrar el número óptimo de clusters
    optimal_idx = normalized_df['composite_score'].idxmax()
    optimal_clusters = normalized_df.loc[optimal_idx, 'n_clusters']
    
    return int(optimal_clusters)

def plot_final_dendrogram(topic_model, cleaned_topic_summaries, output_dir, 
                         optimal_clusters, linkage_matrix):
    """
    Crear dendrograma final con el número óptimo de clusters
    """
    n_topics = len(cleaned_topic_summaries)
    cluster_assignments = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')
    
    # Crear etiquetas descriptivas
    topic_labels = []
    for tid in range(n_topics):
        topic_info = next((t for t in cleaned_topic_summaries if t['topic_id'] == tid), None)
        if topic_info:
            keywords = topic_info['keywords'].split(', ')[:3]
            percentage = topic_info.get('percentage', 0)
            label = f"T{tid}: {', '.join(keywords)} ({percentage:.1f}%)"
        else:
            label = f"Topic {tid}"
        topic_labels.append(label)
    
    # Colores para clusters
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_clusters))
    cluster_colors = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                     for c in colors]
    
    # Crear dendrograma
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    dendro = dendrogram(
        linkage_matrix,
        labels=topic_labels,
        orientation='left',
        distance_sort='descending',
        show_leaf_counts=False,
        leaf_font_size=9,
        ax=ax,
        color_threshold=0,
        above_threshold_color='black'
    )
    
    # Colorear etiquetas según cluster
    ylbls = ax.get_yticklabels()
    for i, label in enumerate(ylbls):
        topic_idx = dendro['leaves'][i]
        cluster_id = cluster_assignments[topic_idx] - 1
        label.set_color(cluster_colors[cluster_id % len(cluster_colors)])
        label.set_fontweight('bold')
        label.set_fontsize(9)
    
    ax.set_title(f'Dendrograma Óptimo - {optimal_clusters} Clusters', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'optimal_dendrogram_{optimal_clusters}_clusters.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return cluster_assignments

def comprehensive_clustering_analysis(topic_model, cleaned_topic_summaries, output_dir):
    """
    Análisis completo de clustering con evaluación de métricas
    """
    print("="*80)
    print("ANÁLISIS COMPLETO DE CLUSTERING")
    print("="*80)
    
    # Paso 1: Evaluación de métricas
    results_df, coph_corr, optimal_clusters, linkage_matrix, distance_matrix = \
        evaluate_clustering_metrics(topic_model, cleaned_topic_summaries, output_dir)
    
    # Paso 2: Crear dendrograma óptimo
    cluster_assignments = plot_final_dendrogram(
        topic_model, cleaned_topic_summaries, output_dir, optimal_clusters, linkage_matrix
    )
    
    # Paso 3: Análisis de resultados
    print(f"\n" + "="*80)
    print("INTERPRETACIÓN DE RESULTADOS:")
    print("="*80)
    
    # Interpretación del coeficiente cofenético
    if coph_corr > 0.8:
        coph_interpretation = "Excelente - La estructura jerárquica representa muy bien los datos"
    elif coph_corr > 0.7:
        coph_interpretation = "Buena - La estructura jerárquica es representativa"
    elif coph_corr > 0.6:
        coph_interpretation = "Aceptable - La estructura jerárquica es moderadamente representativa"
    else:
        coph_interpretation = "Pobre - La estructura jerárquica no representa bien los datos"
    
    print(f"Coeficiente Cofenético ({coph_corr:.4f}): {coph_interpretation}")
    print(f"Número óptimo de clusters: {optimal_clusters}")
    
    # Crear DataFrame final con clusters
    cluster_df = pd.DataFrame({
        'topic_id': range(len(cleaned_topic_summaries)),
        'cluster': cluster_assignments,
        'topic_summary': [topic['keywords'][:50] + '...' if len(topic['keywords']) > 50 
                         else topic['keywords'] for topic in cleaned_topic_summaries]
    })
    
    cluster_df.to_csv(os.path.join(output_dir, f'final_clusters_{optimal_clusters}.csv'), index=False)
    
    print(f"\nDistribución de topics por cluster:")
    cluster_counts = cluster_df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} topics")
    
    print(f"\n✓ Análisis completo guardado en: {output_dir}")
    
    return {
        'optimal_clusters': optimal_clusters,
        'cluster_assignments': cluster_assignments,
        'cophenetic_correlation': coph_corr,
        'evaluation_results': results_df,
        'cluster_df': cluster_df
    }


results = comprehensive_clustering_analysis(topic_model, cleaned_topic_summaries, output_dir)



# File paths
RESULTS_PATH = r'C:\Users\laura\Desktop\REPORTES\RESULTADOS\topic_modeling_results_reproducible.csv'
ESG_PATH = r"c:\Users\laura\Downloads\esg_eu50.xlsx"
OUTPUT_DIR = r"C:\Users\laura\Desktop\REPORTES\RESULTADOS"

def load_all_data():
    """
    Load all required data files and return them as DataFrames.
    """
    print("="*80)
    print("LOADING DATA FILES")
    print("="*80)
    
    # Load results data
    print(f"Loading results data from: {RESULTS_PATH}")
    try:
        results_df = pd.read_csv(RESULTS_PATH)
        print(f"✓ Results data loaded successfully: {results_df.shape}")
        print(f"  Columns: {list(results_df.columns)}")
        if 'topic' in results_df.columns:
            print(f"  Topics found: {sorted(results_df['topic'].unique())}")
        if 'year' in results_df.columns:
            print(f"  Years found: {sorted(results_df['year'].unique())}")
    except Exception as e:
        print(f"✗ Error loading results data: {e}")
        return None, None
    
    # Load ESG data
    print(f"\nLoading ESG data from: {ESG_PATH}")
    try:
        esg_df = pd.read_excel(ESG_PATH)
        print(f"✓ ESG data loaded successfully: {esg_df.shape}")
        print(f"  Columns: {list(esg_df.columns)}")
        if 'year' in esg_df.columns:
            print(f"  Years found: {sorted(esg_df['year'].unique())}")
        if 'company' in esg_df.columns:
            print(f"  Companies: {esg_df['company'].nunique()}")
    except Exception as e:
        print(f"✗ Error loading ESG data: {e}")
        return results_df, None
    
    return results_df, esg_df

# TIME SERIES ANALYSIS FUNCTIONS

def calculate_topic_proportions_by_year(results_df, n_final_topics=None):
    """
    Calculate topic proportions by year for time series analysis.
    """
    print("\n" + "="*80)
    print("CALCULATING TOPIC PROPORTIONS BY YEAR")
    print("="*80)
    
    print("Calculating topic proportions by year...")
    
    # Filter out noise topics (topic = -1)
    interpretable_data = results_df[results_df['topic'] != -1].copy()
    
    print(f"Total sentences: {len(results_df)}")
    print(f"Interpretable sentences: {len(interpretable_data)}")
    
    # Calculate proportions by year
    year_topic_counts = interpretable_data.groupby(['year', 'topic']).size().reset_index(name='count')
    year_totals = interpretable_data.groupby('year').size().reset_index(name='total')
    
    # Merge to get proportions
    topic_proportions = year_topic_counts.merge(year_totals, on='year')
    topic_proportions['proportion'] = topic_proportions['count'] / topic_proportions['total']
    
    # Create pivot table for easier analysis
    proportions_pivot = topic_proportions.pivot_table(
        index='year', 
        columns='topic', 
        values='proportion', 
        fill_value=0
    )
    
    print(f"Proportions pivot shape: {proportions_pivot.shape}")
    print(f"Years covered: {proportions_pivot.index.min()} to {proportions_pivot.index.max()}")
    
    return proportions_pivot, topic_proportions

def identify_most_varying_topics(proportions_pivot, top_n=10):
    """
    Identify topics with the highest variance across years.
    """
    print("\nIdentifying most varying topics...")
    
    # Calculate variance for each topic
    topic_variances = proportions_pivot.var().sort_values(ascending=False)
    most_varying_topics = topic_variances.head(top_n).index.tolist()
    
    print(f"Top {top_n} most varying topics:")
    for i, topic in enumerate(most_varying_topics, 1):
        variance = topic_variances[topic]
        print(f"  {i}. Topic {topic}: variance = {variance:.6f}")
    
    return most_varying_topics, topic_variances

def create_time_series_visualization(proportions_pivot, most_varying_topics, 
                                   topic_summaries=None, output_dir=None):
    """
    Create time series visualization of topic proportions.
    """
    print("\nCreating time series visualization...")
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Plot most varying topics
    for topic in most_varying_topics:
        if topic in proportions_pivot.columns:
            label = f"Topic {topic}"
            if topic_summaries and topic in topic_summaries:
                # Truncate long summaries
                summary = topic_summaries[topic][:50] + "..." if len(topic_summaries[topic]) > 50 else topic_summaries[topic]
                label = f"Topic {topic}: {summary}"
            
            plt.plot(proportions_pivot.index, proportions_pivot[topic], 
                    marker='o', linewidth=2, markersize=4, label=label)
    
    plt.title('Topic Proportions Over Time (Most Varying Topics)', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Topic Proportion', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'topic_time_series.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {os.path.join(output_dir, 'topic_time_series.png')}")
    
    plt.show()

# ESG CORRELATION ANALYSIS FUNCTIONS


def analyze_company_matching(results_df, esg_df):
    """
    Analyze and provide insights into company matching between datasets.
    """
    print("\n" + "="*80)
    print("COMPANY MATCHING ANALYSIS")
    print("="*80)
    
    # Get unique companies from both datasets
    results_companies = results_df['filename'].str.extract(r'([A-Z]+)')[0].unique()
    esg_companies = esg_df['company'].unique()
    
    print(f"\nResults dataset companies ({len(results_companies)}):")
    print(sorted(results_companies))
    
    print(f"\nESG dataset companies ({len(esg_companies)}):")
    print(sorted(esg_companies))
    
    # Look for potential matches using string similarity
    print("\n=== POTENTIAL MATCHES (using string similarity) ===")
    potential_matches = []
    
    for res_company in results_companies:
        best_match = None
        best_score = 0
        
        for esg_company in esg_companies:
            # Calculate similarity between company names
            similarity = SequenceMatcher(None, res_company.upper(), esg_company.upper()).ratio()
            
            # Also check if one is contained in the other
            if res_company.upper() in esg_company.upper() or esg_company.upper() in res_company.upper():
                similarity = max(similarity, 0.8)  # Boost score for containment
            
            if similarity > best_score:
                best_score = similarity
                best_match = esg_company
        
        if best_score > 0.3:  # Only show reasonably good matches
            potential_matches.append({
                'results_company': res_company,
                'esg_company': best_match,
                'similarity': best_score
            })
            print(f"{res_company:15} -> {best_match:20} (similarity: {best_score:.3f})")
    
    return potential_matches

def create_company_mapping_from_analysis(potential_matches, similarity_threshold=0.6):
    """
    Create a company mapping dictionary from the analysis results.
    """
    mapping = {}
    
    print(f"\n=== CREATING MAPPING (threshold: {similarity_threshold}) ===")
    
    for match in potential_matches:
        if match['similarity'] >= similarity_threshold:
            mapping[match['results_company']] = match['esg_company']
            print(f"Mapping: {match['results_company']} -> {match['esg_company']}")
    
    print(f"\nCreated {len(mapping)} mappings")
    return mapping

def calculate_sector_topic_correlations(results_df, esg_df, company_mapping):
    """
    Calculate correlations between ESG scores and topic proportions by sector.
    """
    print("\n=== CALCULATING SECTOR-TOPIC CORRELATIONS ===")
    
    # Add company codes to results
    results_df['company_code'] = results_df['filename'].str.extract(r'([A-Z]+)')[0]
    
    # Filter for interpretable topics only
    interpretable_data = results_df[results_df['topic'] != -1].copy()
    
    # Calculate topic proportions by company and year
    company_year_topic = interpretable_data.groupby(['company_code', 'year', 'topic']).size().reset_index(name='count')
    company_year_totals = interpretable_data.groupby(['company_code', 'year']).size().reset_index(name='total')
    
    company_proportions = company_year_topic.merge(company_year_totals, on=['company_code', 'year'])
    company_proportions['proportion'] = company_proportions['count'] / company_proportions['total']
    
    # Create merged dataset with ESG data
    merged_data = []
    
    for _, row in company_proportions.iterrows():
        company_code = row['company_code']
        year = row['year']
        
        # Check if we have a mapping for this company
        if company_code in company_mapping:
            esg_company_name = company_mapping[company_code]
            
            # Find ESG data for this company and year
            esg_match = esg_df[(esg_df['company'] == esg_company_name) & 
                              (esg_df['year'] == year)]
            
            if not esg_match.empty:
                esg_row = esg_match.iloc[0]
                merged_data.append({
                    'company_code': company_code,
                    'year': year,
                    'esg_company': esg_row['company'],
                    'sector': esg_row['sector'],
                    'topic': row['topic'],
                    'proportion': row['proportion'],
                    'esg_score': esg_row['esg']
                })
    
    merged_df = pd.DataFrame(merged_data)
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Calculate correlations by sector and topic
    correlation_results = []
    
    for sector in merged_df['sector'].unique():
        sector_data = merged_df[merged_df['sector'] == sector]
        
        for topic in sector_data['topic'].unique():
            topic_data = sector_data[sector_data['topic'] == topic]
            
            if len(topic_data) > 2:  # Need at least 3 data points for correlation
                try:
                    correlation, p_value = pearsonr(topic_data['esg_score'], topic_data['proportion'])
                    correlation_results.append({
                        'sector': sector,
                        'topic': topic,
                        'correlation': correlation,
                        'p_value': p_value,
                        'n_observations': len(topic_data)
                    })
                except:
                    # Handle cases where correlation can't be calculated
                    correlation_results.append({
                        'sector': sector,
                        'topic': topic,
                        'correlation': 0,
                        'p_value': 1,
                        'n_observations': len(topic_data)
                    })
    
    correlation_df = pd.DataFrame(correlation_results)
    print(f"Correlation results shape: {correlation_df.shape}")
    
    return correlation_df, merged_df

def create_correlation_heatmap(correlation_df, output_dir, topic_names=None):
    """
    Generate a heatmap of correlations between sectors and topics.
    """
    print("\n=== CREATING SECTOR-TOPIC CORRELATION HEATMAP ===")
    
    # Create pivot table for heatmap
    heatmap_data = correlation_df.pivot_table(
        index='sector', 
        columns='topic', 
        values='correlation', 
        fill_value=0
    )
    
    # Replace topic numbers with names if provided
    if topic_names:
        heatmap_data.columns = [topic_names.get(topic, f"Topic {topic}") 
                               for topic in heatmap_data.columns]
    
    # Create the heatmap
    plt.figure(figsize=(14, 10))
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_data, 
                annot=True, 
                cmap='RdBu_r', 
                center=0, 
                linewidths=0.5, 
                fmt='.2f', 
                cbar_kws={'label': 'Correlación de Pearson'},
                square=True)
    
    # Customize the plot
    plt.title('Correlaciones ESG-Temas por Sector', fontsize=16, fontweight='bold')
    plt.xlabel('Temas', fontsize=12)
    plt.ylabel('Sectores', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, 'esg_sector_topic_correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {heatmap_path}")
    
    # Show the plot
    plt.show()
    
    return heatmap_data

def run_esg_correlation_analysis(results_df, esg_df, output_dir, topic_names=None):
    """
    Run the complete ESG correlation analysis.
    """
    print("\n" + "="*80)
    print("ESG CORRELATION ANALYSIS")
    print("="*80)
    
    # Step 1: Analyze company matching
    potential_matches = analyze_company_matching(results_df, esg_df)
    
    # Step 2: Create mapping
    company_mapping = create_company_mapping_from_analysis(potential_matches, similarity_threshold=0.6)
    
    if not company_mapping:
        print("WARNING: No company mappings created. Cannot proceed with ESG correlation analysis.")
        return None
    
    # Step 3: Calculate correlations
    correlation_df, merged_df = calculate_sector_topic_correlations(results_df, esg_df, company_mapping)
    
    # Step 4: Create heatmap
    heatmap_data = create_correlation_heatmap(correlation_df, output_dir, topic_names)
    
    # Step 5: Save results
    correlation_path = os.path.join(output_dir, 'esg_sector_topic_correlations.csv')
    correlation_df.to_csv(correlation_path, index=False)
    print(f"Correlation results saved to: {correlation_path}")
    
    merged_path = os.path.join(output_dir, 'esg_merged_data.csv')
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged data saved to: {merged_path}")
    
    # Step 6: Print summary statistics
    print("\n=== CORRELATION SUMMARY ===")
    print(f"Total correlations calculated: {len(correlation_df)}")
    print(f"Significant correlations (p < 0.05): {len(correlation_df[correlation_df['p_value'] < 0.05])}")
    print(f"Strong positive correlations (r > 0.5): {len(correlation_df[correlation_df['correlation'] > 0.5])}")
    print(f"Strong negative correlations (r < -0.5): {len(correlation_df[correlation_df['correlation'] < -0.5])}")
    
    # Show top positive and negative correlations
    print("\n=== TOP POSITIVE CORRELATIONS ===")
    top_positive = correlation_df.nlargest(5, 'correlation')
    for _, row in top_positive.iterrows():
        print(f"Sector: {row['sector']}, Topic: {row['topic']}, r = {row['correlation']:.3f}, p = {row['p_value']:.3f}")
    
    print("\n=== TOP NEGATIVE CORRELATIONS ===")
    top_negative = correlation_df.nsmallest(5, 'correlation')
    for _, row in top_negative.iterrows():
        print(f"Sector: {row['sector']}, Topic: {row['topic']}, r = {row['correlation']:.3f}, p = {row['p_value']:.3f}")
    
    return correlation_df, merged_df, heatmap_data

# Example usage with topic names (customize as needed)
TOPIC_NAMES = {
    0: "Materiales básicos",
    1: "Bienes cíclicos de consumo", 
    2: "Bienes no cíclicos de consumo",
    3: "Energía",
    4: "Financiero",
    5: "Salud",
    6: "Industrial",
    7: "Inmobiliario",
    8: "Tecnológico",
    9: "Servicios"
    # Add more topic names as needed
}

# Modified main execution function
def run_complete_analysis():
    """
    Run the complete analysis pipeline.
    """
    print("="*80)
    print("STARTING COMPLETE ANALYSIS")
    print("="*80)
    
    # Step 1: Load data
    results_df, esg_df = load_all_data()  # This function should be defined elsewhere
    
    if results_df is None:
        print("ERROR: Could not load results data. Exiting.")
        return
    
    # Step 2: ESG correlation analysis (if ESG data is available)
    if esg_df is not None:
        try:
            correlation_df, merged_df, heatmap_data = run_esg_correlation_analysis(
                results_df, esg_df, OUTPUT_DIR, TOPIC_NAMES
            )
            
            if correlation_df is not None:
                print("\n✓ ESG correlation analysis completed successfully!")
                return correlation_df, merged_df, heatmap_data
            else:
                print("\n⚠ ESG correlation analysis completed with warnings.")
                
        except Exception as e:
            print(f"ERROR in ESG correlation analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nSkipping ESG correlation analysis (no ESG data available)")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_complete_analysis()