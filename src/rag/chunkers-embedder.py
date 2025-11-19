"""
chunkers_embedder.py

Módulo para chunking semántico de documentos FDS y generación de embeddings.
Procesa texto, tablas e imágenes manteniendo metadata original enriquecida.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import ollama
from PIL import Image
import pytesseract


class TextChunker:
    """Divide texto en chunks semánticos con overlapping."""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        """
        Args:
            chunk_size: Tamaño máximo en tokens por chunk
            overlap: Tokens de solapamiento entre chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def estimate_tokens(self, text: str) -> int:
        """Estimación rápida de tokens (1 token ≈ 4 caracteres)."""
        return len(text) // 4
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Divide texto por párrafos respetando límite de tokens.
        
        Args:
            text: Texto completo a dividir
            
        Returns:
            Lista de chunks de texto
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                
                # Mantener overlap con último párrafo
                overlap_paras = []
                overlap_tokens = 0
                for p in reversed(current_chunk):
                    p_tokens = self.estimate_tokens(p)
                    if overlap_tokens + p_tokens <= self.overlap:
                        overlap_paras.insert(0, p)
                        overlap_tokens += p_tokens
                    else:
                        break
                
                current_chunk = overlap_paras
                current_tokens = overlap_tokens
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def add_context(self, chunk: str, metadata: Dict[str, Any]) -> str:
        """
        Agrega contexto del documento al chunk.
        
        Args:
            chunk: Texto del chunk
            metadata: Metadata del documento original
            
        Returns:
            Chunk con contexto prepended
        """
        producto = metadata.get('nombre_producto', 'Producto desconocido')
        seccion = metadata.get('seccion_actual', 'General')
        
        contexto = f"Documento: {producto}\nSección: {seccion}\n\n"
        return contexto + chunk


class TableChunker:
    """Convierte tablas CSV a texto estructurado."""
    
    def csv_to_text(self, csv_path: Path) -> str:
        """
        Convierte CSV a representación textual.
        
        Args:
            csv_path: Ruta al archivo CSV
            
        Returns:
            Tabla en formato texto
        """
        df = pd.read_csv(csv_path)
        
        # Construir texto estructurado
        lines = []
        lines.append(f"Tabla con {len(df)} filas y {len(df.columns)} columnas")
        lines.append(f"Columnas: {', '.join(df.columns)}")
        lines.append("")
        
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            lines.append(row_text)
        
        return '\n'.join(lines)
    
    def add_table_metadata(self, table_text: str, csv_path: Path, doc_metadata: Dict) -> Dict:
        """
        Enriquece metadata de tabla.
        
        Args:
            table_text: Texto de la tabla
            csv_path: Ruta al CSV original
            doc_metadata: Metadata del documento padre
            
        Returns:
            Metadata completa de la tabla
        """
        df = pd.read_csv(csv_path)
        
        metadata = doc_metadata.copy()
        metadata.update({
            'tipo_contenido': 'tabla',
            'tabla_rows': len(df),
            'tabla_cols': len(df.columns),
            'tabla_headers': list(df.columns),
            'tabla_source': csv_path.name
        })
        
        return metadata


class ImageProcessor:
    """Procesa imágenes extrayendo texto con OCR."""
    
    def extract_text_from_image(self, img_path: Path) -> str:
        """
        Extrae texto de imagen usando OCR.
        
        Args:
            img_path: Ruta a la imagen
            
        Returns:
            Texto extraído
        """
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img, lang='spa')
            return text.strip()
        except Exception as e:
            print(f"Error OCR en {img_path}: {e}")
            return ""
    
    def generate_image_description(self, img_path: Path) -> str:
        """
        Genera descripción de imagen.
        
        Args:
            img_path: Ruta a la imagen
            
        Returns:
            Descripción textual
        """
        ocr_text = self.extract_text_from_image(img_path)
        
        # Detectar tipo de imagen por nombre
        filename = img_path.stem.lower()
        if 'pictograma' in filename or 'ghs' in filename:
            tipo = "Pictograma de seguridad"
        elif 'etiqueta' in filename or 'label' in filename:
            tipo = "Etiqueta del producto"
        else:
            tipo = "Imagen"
        
        desc = f"{tipo}. "
        if ocr_text:
            desc += f"Texto visible: {ocr_text}"
        
        return desc


class OllamaEmbedder:
    """Genera embeddings usando Ollama."""
    
    def __init__(self, model: str = "nomic-embed-text"):
        """
        Args:
            model: Modelo de Ollama para embeddings
        """
        self.model = model
        self._verificar_modelo()
    
    def _verificar_modelo(self):
        """Verifica que el modelo esté disponible."""
        try:
            ollama.embeddings(model=self.model, prompt="test")
        except Exception as e:
            raise RuntimeError(
                f"Modelo {self.model} no disponible. "
                f"Ejecuta: ollama pull {self.model}\n"
                f"Error: {e}"
            )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Genera embedding de un texto.
        
        Args:
            text: Texto a embedder
            
        Returns:
            Vector de embedding
        """
        response = ollama.embeddings(model=self.model, prompt=text)
        return response['embedding']
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings de múltiples textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de vectores
        """
        return [self.embed_text(text) for text in texts]


class ChunkMetadataBuilder:
    """Construye metadata enriquecida para cada chunk."""
    
    @staticmethod
    def generate_chunk_id(content: str, doc_id: str, index: int) -> str:
        """
        Genera ID único para chunk.
        
        Args:
            content: Contenido del chunk
            doc_id: ID del documento padre
            index: Índice del chunk
            
        Returns:
            ID único del chunk
        """
        hash_content = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{index:03d}_{hash_content}"
    
    @staticmethod
    def build_metadata(
        chunk: str,
        doc_metadata: Dict,
        chunk_index: int,
        total_chunks: int,
        tipo_contenido: str = "texto"
    ) -> Dict:
        """
        Construye metadata completa del chunk.
        
        Args:
            chunk: Texto del chunk
            doc_metadata: Metadata del documento original
            chunk_index: Índice del chunk actual
            total_chunks: Total de chunks del documento
            tipo_contenido: Tipo de contenido (texto, tabla, imagen)
            
        Returns:
            Metadata enriquecida
        """
        doc_id = doc_metadata.get('codigo', 'unknown')
        chunk_id = ChunkMetadataBuilder.generate_chunk_id(chunk, doc_id, chunk_index)
        
        metadata = {
            # Metadata heredada del documento
            'producto': doc_metadata.get('nombre_producto'),
            'fabricante': doc_metadata.get('proveedor'),
            'codigo_producto': doc_metadata.get('codigo'),
            'fecha_fds': doc_metadata.get('fecha_emision'),
            
            # Metadata del chunk
            'chunk_id': chunk_id,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'tipo_contenido': tipo_contenido,
            'char_count': len(chunk),
            'token_count': len(chunk) // 4,
            
            # Metadata de sección
            'seccion': doc_metadata.get('seccion_actual', 'general'),
            
            # Metadata de clasificación
            'componentes_cas': doc_metadata.get('componentes_quimicos', []),
            'codigos_h': doc_metadata.get('indicaciones_peligro_detalle', []),
            'codigos_p': doc_metadata.get('frases_precaucion_detalle', []),
        }
        
        # Eliminar valores None
        return {k: v for k, v in metadata.items() if v is not None}


class DocumentProcessor:
    """Orquesta el procesamiento completo de documentos."""
    
    def __init__(self, config):
        """
        Args:
            config: Instancia de ProjectConfig
        """
        self.config = config
        self.text_chunker = TextChunker()
        self.table_chunker = TableChunker()
        self.image_processor = ImageProcessor()
        self.embedder = OllamaEmbedder()
        self.metadata_builder = ChunkMetadataBuilder()
    
    def process_document(self, metadata_path: Path) -> List[Dict]:
        """
        Procesa un documento completo generando chunks con embeddings.
        
        Args:
            metadata_path: Ruta al archivo JSON de metadata
            
        Returns:
            Lista de chunks procesados con embeddings y metadata
        """
        with open(metadata_path, 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        
        doc_metadata = doc_data.get('metadata', {})
        nombre_base = metadata_path.stem.replace('_metadata', '')
        
        all_chunks = []
        
        # Procesar texto por secciones
        secciones = doc_metadata.get('secciones', {})
        for seccion_nombre, seccion_texto in secciones.items():
            doc_metadata['seccion_actual'] = seccion_nombre
            
            text_chunks = self.text_chunker.chunk_by_paragraphs(seccion_texto)
            
            for idx, chunk_text in enumerate(text_chunks):
                chunk_with_context = self.text_chunker.add_context(chunk_text, doc_metadata)
                
                metadata = self.metadata_builder.build_metadata(
                    chunk_with_context,
                    doc_metadata,
                    len(all_chunks),
                    len(text_chunks),
                    tipo_contenido="texto"
                )
                
                embedding = self.embedder.embed_text(chunk_with_context)
                
                all_chunks.append({
                    'id': metadata['chunk_id'],
                    'content': chunk_with_context,
                    'embedding': embedding,
                    'metadata': metadata
                })
        
        # Procesar tablas
        carpeta_tablas = self.config.get_folder('tables')
        tablas_paths = list(carpeta_tablas.glob(f"{nombre_base}_tabla_*.csv"))
        
        for tabla_path in tablas_paths:
            tabla_text = self.table_chunker.csv_to_text(tabla_path)
            metadata = self.table_chunker.add_table_metadata(tabla_text, tabla_path, doc_metadata)
            
            metadata = self.metadata_builder.build_metadata(
                tabla_text,
                metadata,
                len(all_chunks),
                1,
                tipo_contenido="tabla"
            )
            
            embedding = self.embedder.embed_text(tabla_text)
            
            all_chunks.append({
                'id': metadata['chunk_id'],
                'content': tabla_text,
                'embedding': embedding,
                'metadata': metadata
            })
        
        # Procesar imágenes
        carpeta_imagenes = self.config.get_folder('images')
        imagenes_paths = list(carpeta_imagenes.glob(f"{nombre_base}_p*_img*.png")) + \
                        list(carpeta_imagenes.glob(f"{nombre_base}_p*_img*.jpeg"))
        
        for img_path in imagenes_paths:
            img_text = self.image_processor.generate_image_description(img_path)
            
            if not img_text:
                continue
            
            img_metadata = doc_metadata.copy()
            img_metadata.update({
                'tipo_contenido': 'imagen',
                'imagen_path': str(img_path),
                'imagen_nombre': img_path.name
            })
            
            metadata = self.metadata_builder.build_metadata(
                img_text,
                img_metadata,
                len(all_chunks),
                1,
                tipo_contenido="imagen"
            )
            
            embedding = self.embedder.embed_text(img_text)
            
            all_chunks.append({
                'id': metadata['chunk_id'],
                'content': img_text,
                'embedding': embedding,
                'metadata': metadata
            })
        
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[Dict], output_path: Path):
        """
        Guarda chunks procesados en JSON.
        
        Args:
            chunks: Lista de chunks procesados
            output_path: Ruta de salida
        """
        output_data = {
            'total_chunks': len(chunks),
            'chunks': chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"Guardados {len(chunks)} chunks en {output_path}")


def main():
    """Ejecuta procesamiento de todos los documentos."""
    from config import ProjectConfig
    
    config = ProjectConfig()
    processor = DocumentProcessor(config)
    
    carpeta_metadata = config.get_folder('metadata')
    carpeta_chunks = config.get_folder('processed_chunks')
    carpeta_chunks.mkdir(parents=True, exist_ok=True)
    
    metadata_files = list(carpeta_metadata.glob("*_metadata.json"))
    
    print(f"Procesando {len(metadata_files)} documentos...")
    
    for metadata_path in metadata_files:
        print(f"\nProcesando: {metadata_path.name}")
        
        try:
            chunks = processor.process_document(metadata_path)
            
            output_path = carpeta_chunks / f"{metadata_path.stem}_chunks.json"
            processor.save_processed_chunks(chunks, output_path)
            
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Texto: {sum(1 for c in chunks if c['metadata']['tipo_contenido'] == 'texto')}")
            print(f"  Tablas: {sum(1 for c in chunks if c['metadata']['tipo_contenido'] == 'tabla')}")
            print(f"  Imágenes: {sum(1 for c in chunks if c['metadata']['tipo_contenido'] == 'imagen')}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\nProcesamiento completado")


if __name__ == "__main__":
    main()