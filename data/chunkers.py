import os
import json
import re
from typing import List, Dict
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict
from io import StringIO
import pandas as pd
from config import ProjectConfig

def _chunk_table(table: Dict) -> List[Dict]:
    """Convierte una tabla en un chunk de texto estructurado."""
    try:
        # Aquí está la corrección
        df = pd.read_json(StringIO(table['content']), orient='split')
        table_text = f"Tabla de la página {table['page']}:\n"
        table_text += df.to_markdown(index=False)
        return [{'content': table_text, 'source': 'table'}]
    except Exception as e:
        print(f"❌ Error convirtiendo tabla a chunk: {e}")
        return []

def _chunk_text(text: str, max_tokens: int = 500) -> List[Dict]:
    """Divide un texto en chunks, manteniendo la lógica para no cortar frases."""
    chunks = []
    words = text.split()
    current_chunk = ""
    for word in words:
        if len(current_chunk.split()) + len(word.split()) <= max_tokens:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [{'content': c, 'source': 'text'} for c in chunks]

def _chunk_table(table: Dict) -> List[Dict]:
    """Convierte una tabla en un chunk de texto estructurado."""
    try:
        df = pd.read_json(table['content'], orient='split')
        table_text = f"Tabla de la página {table['page']}:\n"
        table_text += df.to_markdown(index=False)
        return [{'content': table_text, 'source': 'table'}]
    except Exception as e:
        print(f"❌ Error convirtiendo tabla a chunk: {e}")
        return []

def _process_single_extraction(file_path: Path) -> List[Dict]:
    """Procesa un archivo de extracción y genera chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ No se pudo leer el archivo de extracción {file_path}: {e}")
        return []

    all_chunks = []
    
    # Chunking de secciones
    for section_num, section_data in data.get('sections', {}).items():
        title = section_data['title']
        content = section_data['content']
        chunked_content = _chunk_text(f"Sección {section_num}: {title}\n{content}")
        for chunk in chunked_content:
            chunk['metadata'] = {'section': section_num, 'source_file': data['source_file']}
            all_chunks.append(chunk)

    # Chunking de tablas
    for table_data in data.get('tables', []):
        chunked_table = _chunk_table(table_data)
        for chunk in chunked_table:
            chunk['metadata'] = {'table_id': table_data['table_id'], 'source_file': data['source_file']}
            all_chunks.append(chunk)

    return all_chunks

def generate_chunks_only(base_project_path: str):
    """
    Función principal para la generación de chunks.
    Lee los archivos de extracción y crea chunks procesables.
    """
    config = ProjectConfig(base_project_path)
    extracted_dir = config.get_path('extracted_content')
    processed_dir = config.get_path('processed_chunks')
    
    if not extracted_dir.is_dir():
        print(f"❌ Carpeta de contenido extraído no encontrada: {extracted_dir}")
        return

    extracted_files = list(extracted_dir.glob('*.json'))
    if not extracted_files:
        print("🔍 No se encontraron archivos de extracción JSON. Finalizando.")
        return

    all_documents_chunks = []
    for file in extracted_files:
        print(f"🔄 Generando chunks para {file.name}...")
        chunks = _process_single_extraction(file)
        if chunks:
            all_documents_chunks.extend(chunks)

    # Guardar los chunks combinados en un solo archivo
    output_path = processed_dir / f"chunks_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_documents_chunks, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Proceso de chunking completado. Total de chunks generados: {len(all_documents_chunks)}")
    print(f"   • Chunks guardados en: {output_path}")

    return all_documents_chunks

