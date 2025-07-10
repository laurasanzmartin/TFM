import re
import os
import string
from tqdm import tqdm
import pandas as pd
import glob
from rich.console import Console

# Configuration parameters - you can modify these directly
INPUT_FOLDER = r"C:\Users\laura\Desktop\REPORTES\REPORTES FINALES\TXT"
OUTPUT_FOLDER = r"C:\Users\laura\Desktop\REPORTES\REPORTES FINALES\limpios2"
LETTER_CHAR_RATIO = 0.4  # minimum ratio between letters and characters
NUMBERS_WORDS_RATIO = 0.3  # maximum ratio between numbers and words
MIN_LENGTH = 5  # minimum length of sentence in words
VERBOSE = True  # print detailed processing information
SAMPLE_SIZE = 0  # 0 for all files, or specify a number to limit processing

# Define regex patterns for sentence splitting
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    """
    Source: https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    """
    if VERBOSE:
        print(f"Longitud del texto original: {len(text)} caracteres")
        if len(text) < 100:  # Si el texto es muy corto, imprimimos para depuración
            print(f"Texto original (completo): {text}")
        else:
            print(f"Texto original (primeros 100 chars): {text[:100]}...")
    
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = remove_url(text, '')
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if '"' in text: text = text.replace('."','".')
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("•",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    
    if VERBOSE:
        print(f"Número de oraciones extraídas: {len(sentences)}")
        if sentences:
            print(f"Ejemplo de oración: {sentences[0][:100]}...")
    
    return sentences

def remove_url(text, replace_token):
    regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)

def get_numbers_words_ratio(sent):
    """
    Calculates the ratio between the number of words and numbers in a sentence
    :param str sent: input sentence
    :return float: the ratio between words and numbers in an input sentence 
    """
    sent = sent.replace(",", " ")
    sent = sent.split()
    numeric = 0
    alphanumeric = 0
    for token in sent:
        if token.isnumeric():
            numeric += 1
        alphanumeric += 1
    if alphanumeric == 0:
        return 0
    ratio = numeric / alphanumeric
    return ratio

def get_letters_ratio(sent):
    if not sent:
        return 0
    return sum([token.isalpha() for token in sent])/len(sent)

def get_proportion_capital_words(sent):
    words = sent.split()
    if not words:
        return 0
    return len([word for word in words if word.isupper()])/len(words)
    
def check_sentence(sent, letter_char_ratio, numbers_words_ratio, min_length):
    """
    :param float letter_char_ratio: minimum ratio between characters and letter a sentence can have
    :param float number_words_ratio: maximum ratio between words and numbers a sentence can have
    :param int min_length: minimum length of the sentence in words
    """
    sent = sent.strip()
    
    # Agregamos rastreo de cada condición para depuración
    reasons = []
    
    if sent.isupper():
        reasons.append("Todo en mayúsculas")
    
    letters_ratio = get_letters_ratio(sent)
    if letters_ratio < letter_char_ratio:
        reasons.append(f"Ratio de letras bajo: {letters_ratio:.2f} < {letter_char_ratio}")
    
    if len(sent) > 400:
        reasons.append(f"Oración demasiado larga: {len(sent)} > 400")
    
    if sent and (not sent[0].isalpha() or sent[0].islower()):
        reasons.append("No comienza con letra mayúscula")
    
    if not sent or sent[-1] not in string.punctuation:
        reasons.append("No termina con signo de puntuación")
    
    numbers_ratio = get_numbers_words_ratio(sent)
    if numbers_ratio > numbers_words_ratio:
        reasons.append(f"Demasiados números: {numbers_ratio:.2f} > {numbers_words_ratio}")
    
    words_count = len(sent.split())
    if words_count < min_length:
        reasons.append(f"Pocas palabras: {words_count} < {min_length}")
    
    single_letter_count = len(re.findall(r'(\s+\w\s+\w)',sent))
    if single_letter_count > 5:
        reasons.append(f"Demasiadas letras sueltas: {single_letter_count} > 5")
    
    capital_ratio = get_proportion_capital_words(sent)
    if capital_ratio > 0.25:
        reasons.append(f"Demasiadas palabras en mayúsculas: {capital_ratio:.2f} > 0.25")
    
    if re.search("cid:", sent):
        reasons.append("Contiene 'cid:'")
    
    if reasons:
        if VERBOSE:
            print(f"Rechazada: '{sent[:50]}...' - Razones: {', '.join(reasons)}")
        return None
    else:
        sent = re.sub(r'\s+', ' ', sent)  # eliminate duplicated whitespaces
        if VERBOSE:
            print(f"Aceptada: '{sent[:50]}...'")
        return sent

def try_read_file(file_path):
    """Try different encodings to read the file"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, encoding=encoding) as f:
                return f.read(), encoding
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try reading in binary mode and decode with errors='ignore'
    try:
        with open(file_path, 'rb') as f:
            binary_data = f.read()
            return binary_data.decode('utf-8', errors='ignore'), 'utf-8-ignore'
    except Exception as e:
        raise IOError(f"Could not read file with any encoding: {str(e)}")

def process_files():
    """Main function to process files"""
    console = Console(record=True)
    
    # Verifica la estructura de directorios
    print(f"Verificando estructura de archivos en {INPUT_FOLDER}...")
    
    # Comprueba si hay archivos .txt directamente en INPUT_FOLDER
    txt_files_in_root = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
    print(f"Archivos .txt en la raíz: {len(txt_files_in_root)}")
    if txt_files_in_root and VERBOSE:
        print(f"Ejemplos: {[os.path.basename(f) for f in txt_files_in_root[:3]]}")
    
    # Comprueba si hay carpetas en INPUT_FOLDER
    subdirs = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
    print(f"Subcarpetas encontradas: {len(subdirs)}")
    # Resto del código de subdirs...
    
    # Asegurándonos de que la carpeta de salida exista
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Creada carpeta de salida: {OUTPUT_FOLDER}")
    
    # Resumen de configuración
    print("\n--- Configuración ---")
    print(f"Ratio mínimo de letras: {LETTER_CHAR_RATIO}")
    print(f"Ratio máximo de números: {NUMBERS_WORDS_RATIO}")
    print(f"Longitud mínima de oración (palabras): {MIN_LENGTH}")
    print(f"Carpeta de entrada: {INPUT_FOLDER}")
    print(f"Carpeta de salida: {OUTPUT_FOLDER}")
    print(f"Modo detallado: {'Sí' if VERBOSE else 'No'}")
    if SAMPLE_SIZE > 0:
        print(f"Procesando muestra de {SAMPLE_SIZE} archivos")
    
    stats = {
        "files_processed": 0,
        "files_with_errors": 0,
        "sentences_processed": 0,
        "sentences_accepted": 0,
        "encodings_used": {},
        "stats_by_year": {}  # Nuevo diccionario para estadísticas por año
    }
    
    # Aquí continúa el resto de tu código...
    
    # Opción 1: Si los archivos están directamente en la carpeta de entrada
    if txt_files_in_root:
        print("\nMODO: Procesando archivos directamente desde la carpeta raíz")
        
        # Limitar el número de archivos si se especifica
        if SAMPLE_SIZE > 0:
            txt_files_in_root = txt_files_in_root[:SAMPLE_SIZE]
            print(f"Procesando muestra de {SAMPLE_SIZE} archivos")
        
        for txtfile in tqdm(txt_files_in_root):
            stats["files_processed"] += 1
            filename = os.path.basename(txtfile)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            # Extraer año del nombre del archivo (suponiendo formato como "AD2017.txt")
            year = None
            match = re.search(r'(\d{4})', filename)
            if match:
                year = match.group(1)
                # Inicializar estadísticas para este año si no existen
                if year not in stats["stats_by_year"]:
                    stats["stats_by_year"][year] = {
                        "files_processed": 0,
                        "sentences_processed": 0,
                        "sentences_accepted": 0
                    }
                stats["stats_by_year"][year]["files_processed"] += 1
            
            try:
                text, encoding = try_read_file(txtfile)
                stats["encodings_used"][encoding] = stats["encodings_used"].get(encoding, 0) + 1
                
                if VERBOSE:
                    print(f"\nProcesando {filename} (encoding: {encoding})")
                
                with open(output_path, 'w+', encoding="utf-8") as output_file:
                    sentences = split_into_sentences(text)
                    stats["sentences_processed"] += len(sentences)
                    if year:
                        stats["stats_by_year"][year]["sentences_processed"] += len(sentences)
                    
                    sentences_accepted = 0
                    for sent in sentences:
                        processed_sentence = check_sentence(
                            sent, 
                            letter_char_ratio=LETTER_CHAR_RATIO, 
                            numbers_words_ratio=NUMBERS_WORDS_RATIO, 
                            min_length=MIN_LENGTH
                        )
                        if processed_sentence:
                            sentences_accepted += 1
                            output_file.write(processed_sentence + '\n')
                    
                    stats["sentences_accepted"] += sentences_accepted
                    if year:
                        stats["stats_by_year"][year]["sentences_accepted"] += sentences_accepted
                    
                    if not VERBOSE:
                        print(f"{filename}: {sentences_accepted}/{len(sentences)} oraciones aceptadas")
                
            except Exception as e:
                stats["files_with_errors"] += 1
                print(f"Error al procesar {txtfile}: {str(e)}")
                # Registrar el error en un archivo de log
                with open(os.path.join(OUTPUT_FOLDER, "errores.log"), "a", encoding="utf-8") as log_file:
                    log_file.write(f"{filename}: {str(e)}\n")
    
    # Opción 2: Si primero hay que entrar a subcarpetas
    elif subdirs:
        print("\nMODO: Procesando archivos desde subcarpetas")
        
        # Limitar el número de carpetas si se especifica
        if SAMPLE_SIZE > 0:
            subdirs = subdirs[:SAMPLE_SIZE]
            print(f"Procesando muestra de {SAMPLE_SIZE} carpetas")
        
        for company_folder in tqdm(subdirs):
            company_path = os.path.join(INPUT_FOLDER, company_folder)
            company_output_folder = os.path.join(OUTPUT_FOLDER, company_folder)
            
            # Crea carpeta de salida para esta empresa si no existe
            if not os.path.exists(company_output_folder):
                os.makedirs(company_output_folder)
            
            txt_files = glob.glob(os.path.join(company_path, "*.txt"))
            if VERBOSE:
                print(f"\nEmpresa {company_folder}: {len(txt_files)} archivos encontrados")
            
            for txtfile in txt_files:
                stats["files_processed"] += 1
                filename = os.path.basename(txtfile)
                output_path = os.path.join(company_output_folder, filename)
                
                try:
                    text, encoding = try_read_file(txtfile)
                    stats["encodings_used"][encoding] = stats["encodings_used"].get(encoding, 0) + 1
                    
                    if VERBOSE:
                        print(f"\nProcesando {company_folder}/{filename} (encoding: {encoding})")
                    
                    with open(output_path, 'w+', encoding="utf-8") as output_file:
                        sentences = split_into_sentences(text)
                        stats["sentences_processed"] += len(sentences)
                        
                        sentences_accepted = 0
                        for sent in sentences:
                            processed_sentence = check_sentence(
                                sent, 
                                letter_char_ratio=LETTER_CHAR_RATIO, 
                                numbers_words_ratio=NUMBERS_WORDS_RATIO, 
                                min_length=MIN_LENGTH
                            )
                            if processed_sentence:
                                sentences_accepted += 1
                                output_file.write(processed_sentence + '\n')
                        
                        stats["sentences_accepted"] += sentences_accepted
                        
                        if not VERBOSE:
                            print(f"{company_folder}/{filename}: {sentences_accepted}/{len(sentences)} oraciones aceptadas")
                
                except Exception as e:
                    stats["files_with_errors"] += 1
                    print(f"Error al procesar {company_folder}/{filename}: {str(e)}")
                    # Registrar el error en un archivo de log
                    with open(os.path.join(OUTPUT_FOLDER, "errores.log"), "a", encoding="utf-8") as log_file:
                        log_file.write(f"{company_folder}/{filename}: {str(e)}\n")
    
    else:
        print("¡ERROR! No se encontraron archivos .txt o carpetas con archivos .txt en la ruta especificada.")
    
    # Mostrar resumen de estadísticas
    print("\n--- Resumen de procesamiento ---")
    print(f"Archivos procesados: {stats['files_processed']}")
    print(f"Archivos con errores: {stats['files_with_errors']}")
    print(f"Oraciones procesadas en total: {stats['sentences_processed']}")
    print(f"Oraciones aceptadas: {stats['sentences_accepted']} ({stats['sentences_accepted']/max(1, stats['sentences_processed'])*100:.1f}%)")
    print("\nCodificaciones utilizadas:")
    for encoding, count in stats["encodings_used"].items():
        print(f"  {encoding}: {count} archivos")
    
    # Añadir resumen por año
    print("\n--- Estadísticas por año ---")
    years = sorted(stats["stats_by_year"].keys())
    for year in years:
        yearly_stats = stats["stats_by_year"][year]
        print(f"\nAño {year}:")
        print(f"  Archivos procesados: {yearly_stats['files_processed']}")
        print(f"  Oraciones procesadas: {yearly_stats['sentences_processed']}")
        print(f"  Oraciones aceptadas: {yearly_stats['sentences_accepted']} ({yearly_stats['sentences_accepted']/max(1, yearly_stats['sentences_processed'])*100:.1f}%)")
        print(f"  Oraciones filtradas: {yearly_stats['sentences_processed'] - yearly_stats['sentences_accepted']}")
    
    # También puedes guardar estas estadísticas por año en el CSV
    yearly_stats_list = []
    for year in years:
        yearly_stats = stats["stats_by_year"][year]
        yearly_stats_list.append({
            "año": year,
            "archivos_procesados": yearly_stats["files_processed"],
            "oraciones_procesadas": yearly_stats["sentences_processed"],
            "oraciones_aceptadas": yearly_stats["sentences_accepted"],
            "oraciones_filtradas": yearly_stats["sentences_processed"] - yearly_stats["sentences_accepted"],
            "porcentaje_aceptacion": yearly_stats["sentences_accepted"]/max(1, yearly_stats["sentences_processed"])*100
        })
    
    # Guardar estadísticas en un archivo CSV
    stats_df = pd.DataFrame([{
        "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivos_procesados": stats['files_processed'],
        "archivos_con_errores": stats['files_with_errors'],
        "oraciones_procesadas": stats['sentences_processed'],
        "oraciones_aceptadas": stats['sentences_accepted'],
        "porcentaje_aceptacion": stats['sentences_accepted']/max(1, stats['sentences_processed'])*100,
        "letter_char_ratio": LETTER_CHAR_RATIO,
        "numbers_words_ratio": NUMBERS_WORDS_RATIO,
        "min_length": MIN_LENGTH,
        "encodings": str(stats["encodings_used"])
    }])
    
    stats_df.to_csv(os.path.join(OUTPUT_FOLDER, "estadisticas_procesamiento.csv"), index=False)
    print(f"\nEstadísticas guardadas en {os.path.join(OUTPUT_FOLDER, 'estadisticas_procesamiento.csv')}")
    
    if yearly_stats_list:
        yearly_df = pd.DataFrame(yearly_stats_list)
        yearly_df.to_csv(os.path.join(OUTPUT_FOLDER, "estadisticas_por_año.csv"), index=False)
        print(f"\nEstadísticas por año guardadas en {os.path.join(OUTPUT_FOLDER, 'estadisticas_por_año.csv')}")
    
    # Devolver las estadísticas para posible uso posterior
    return stats

if __name__ == "__main__":
    # Execute the process_files function and store the returned stats
    stats = process_files()