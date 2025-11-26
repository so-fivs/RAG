"""
image_vision_processor.py

Procesador de imágenes usando modelos de visión multimodal.
Actualiza chunks de imágenes con descripciones mejoradas.
"""

import ollama
import json
from pathlib import Path
from typing import Dict, List


class VisionImageProcessor:
    """Procesa imágenes con modelo de visión multimodal."""
    
    SAFETY_PROMPT = """Describe este elemento visual de una ficha de seguridad química.
Identifica:
1. Tipo (pictograma GHS, etiqueta, diagrama, tabla)
2. Símbolos o iconos visibles
3. Colores significativos
4. Texto legible
5. Significado de seguridad

Responde en español, máximo 3 líneas."""
    
    def __init__(self, model: str = "llava:7b"):
        """
        Args:
            model: Modelo de visión de Ollama
        """
        self.model = model
    
    def describe_image(self, img_path: Path) -> str:
        """
        Genera descripción de imagen.
        
        Args:
            img_path: Ruta a la imagen
            
        Returns:
            Descripción textual
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': self.SAFETY_PROMPT,
                    'images': [str(img_path)]
                }]
            )
            
            return response['message']['content'].strip()
        
        except Exception as e:
            print(f"Error procesando {img_path.name}: {e}")
            return f"Imagen: {img_path.name}"
    
    def detect_pictogram_type(self, description: str) -> str:
        """Clasifica tipo de pictograma."""
        description_lower = description.lower()
        
        pictogram_types = {
            'inflamable': ['llama', 'fuego', 'inflamable'],
            'corrosivo': ['corrosión', 'tubo', 'ácido'],
            'toxico': ['calavera', 'cráneo', 'tóxico'],
            'irritante': ['signo exclamación', 'irritante'],
            'peligro_salud': ['silueta', 'cuerpo humano', 'salud'],
            'explosivo': ['explosión', 'bomba'],
            'comburente': ['llama círculo', 'oxígeno'],
            'gas_presion': ['cilindro', 'gas', 'presión'],
            'medio_ambiente': ['árbol', 'pez', 'ambiental']
        }
        
        for tipo, keywords in pictogram_types.items():
            if any(kw in description_lower for kw in keywords):
                return tipo
        
        return 'otro'


class ChunksImageUpdater:
    """Actualiza chunks de imágenes con descripciones mejoradas."""
    
    def __init__(self, config, model: str = "llava:7b"):
        """
        Args:
            config: Instancia de ProjectConfig
            model: Modelo de visión
        """
        self.config = config
        self.processor = VisionImageProcessor(model=model)
    def _generate_embedding(self, text: str) -> List[float]:
        """Genera embedding para texto."""
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )
        return response['embedding']
    
    def update_chunks_file(self, chunks_path: Path) -> Dict[str, int]:
        """Actualiza un archivo de chunks."""
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraer nombre base del documento desde el archivo
        # FDS 21 - Esmalte_Uretano_AR_metadata_chunks.json -> FDS 21 - Esmalte_Uretano_AR
        nombre_base = chunks_path.stem.replace('_metadata_chunks', '')
        
        # Carpeta de imágenes
        img_folder = self.config.get_folder('images')
        
        # Buscar todas las imágenes de este documento
        imagenes_doc = list(img_folder.glob(f"{nombre_base}*"))
        
        if not imagenes_doc:
            print(f"  No hay imágenes para {nombre_base}")
            return {'updated': 0, 'skipped': 0, 'errors': 0}
        
        print(f"  Encontradas {len(imagenes_doc)} imágenes")
        
        updated = 0
        skipped = 0
        errors = 0
        img_index = 0
        
        for chunk in data['chunks']:
            if chunk['metadata'].get('tipo_contenido') != 'imagen':
                continue
            
            # Asignar imagen secuencialmente
            if img_index >= len(imagenes_doc):
                skipped += 1
                continue
            
            img_path = imagenes_doc[img_index]
            img_index += 1
            
            try:
                new_desc = self.processor.describe_image(img_path)
                chunk['content'] = new_desc
                
                pictogram_type = self.processor.detect_pictogram_type(new_desc)
                chunk['metadata']['pictograma_tipo'] = pictogram_type
                chunk['metadata']['imagen_path'] = str(img_path)
                chunk['metadata']['imagen_nombre'] = img_path.name
                
                embedding = self._generate_embedding(new_desc)
                chunk['embedding'] = embedding
                
                updated += 1
                print(f"  Actualizada: {img_path.name}")
            
            except Exception as e:
                print(f"  Error: {e}")
                errors += 1
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {'updated': updated, 'skipped': skipped, 'errors': errors}
    
    def update_all_chunks(self) -> Dict[str, int]:
        """
        Actualiza todos los archivos de chunks.
        
        Returns:
            Estadísticas totales
        """
        chunks_folder = self.config.get_folder('processed_chunks')
        chunk_files = list(chunks_folder.glob("*_chunks.json"))
        
        total_updated = 0
        total_skipped = 0
        total_errors = 0
        
        print(f"\nActualizando {len(chunk_files)} archivos...")
        
        for chunk_file in chunk_files:
            print(f"\nProcesando: {chunk_file.name}")
            
            stats = self.update_chunks_file(chunk_file)
            
            total_updated += stats['updated']
            total_skipped += stats['skipped']
            total_errors += stats['errors']
            
            print(f"  Actualizado: {stats['updated']}")
            print(f"  Omitido: {stats['skipped']}")
            print(f"  Errores: {stats['errors']}")
        
        return {
            'total_updated': total_updated,
            'total_skipped': total_skipped,
            'total_errors': total_errors
        }


def main():
    """Ejecuta actualización de todos los chunks de imágenes."""
    from RAG.data.config import ProjectConfig
    
    config = ProjectConfig()
    
    print("ACTUALIZADOR DE CHUNKS DE IMAGENES")
    print("=" * 80)
    print("\nNOTA: Requiere modelo llava:7b instalado")
    print("Instala con: ollama pull llava:7b\n")
    
    try:
        updater = ChunksImageUpdater(config)
        stats = updater.update_all_chunks()
        
        print("\n" + "=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)
        print(f"Total actualizado: {stats['total_updated']}")
        print(f"Total omitido: {stats['total_skipped']}")
        print(f"Total errores: {stats['total_errors']}")
        
        if stats['total_updated'] > 0:
            print("\nIMPORTANTE: Re-ejecuta chromadb_ingestor.py para actualizar base de datos")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("Verifica que el modelo llava:7b este instalado")


if __name__ == "__main__"