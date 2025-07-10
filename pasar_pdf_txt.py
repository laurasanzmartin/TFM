import os
import PyPDF2
from pathlib import Path



def pdf_to_txt(pdf_folder, output_folder=None):
    """
    Convierte todos los archivos PDF en una carpeta a archivos TXT.
    
    Args:
        pdf_folder (str): Ruta a la carpeta con los archivos PDF.
        output_folder (str, opcional): Ruta donde guardar los archivos TXT.
                                      Si no se especifica, se usa la misma carpeta.
    """
    # Si no se especifica una carpeta de salida, usa la misma carpeta
    if output_folder is None:
        output_folder = pdf_folder
    
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Obtener todos los archivos PDF en la carpeta
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No se encontraron archivos PDF en {pdf_folder}")
        return
    
    print(f"Encontrados {len(pdf_files)} archivos PDF para convertir")
    
    # Procesar cada archivo PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        txt_filename = os.path.splitext(pdf_file)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)
        
        try:
            # Abrir el archivo PDF
            with open(pdf_path, 'rb') as pdf:
                # Crear objeto lector de PDF
                pdf_reader = PyPDF2.PdfReader(pdf)
                
                # Obtener el número de páginas
                num_pages = len(pdf_reader.pages)
                
                # Extraer texto de todas las páginas
                text = ""
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                
                # Guardar el texto en un archivo TXT
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
                
                print(f"Convertido: {pdf_file} -> {txt_filename}")
        
        except Exception as e:
            print(f"Error al procesar {pdf_file}: {str(e)}")
    
    print("Proceso completado.")

# Ejemplo de uso
if __name__ == "__main__":
    # Carpeta donde están los PDFs (cambia esto a tu ruta)
    pdf_folder = r"C:\Users\laura\Desktop\REPORTES\REPORTES FINALES\FALTAN"
    
    # Carpeta donde guardar los TXTs (opcional)
    # Si no quieres una carpeta separada, comenta esta línea
    output_folder = r"C:\Users\laura\Desktop\REPORTES\REPORTES FINALES\TXT"
    
    # Ejecutar la conversión
    pdf_to_txt(pdf_folder, output_folder)
    
    # Para usar la misma carpeta para entrada y salida:
    # pdf_to_txt(pdf_folder)  