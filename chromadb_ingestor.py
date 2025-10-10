"""
chromadb_ingestor.py

Módulo para ingestar chunks procesados en ChromaDB.
Lee JSONs de processed_chunks/ y crea colecciones separadas por tipo de contenido.
"""

import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ChromaDBIngestor:
    """Gestiona la ingesta de chunks en ChromaDB."""
    
    def __init__(self, config, reset_db: bool = True):
        """
        Args:
            config: Instancia de ProjectConfig
            reset_db: Si True, elimina base de datos existente
        """
        self.config = config
        self.db_path = config.get_folder('vector_db')
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        if reset_db and self.db_path.exists():
            import shutil
            shutil.rmtree(self.db_path)
            self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collections = self._initialize_collections()
    
    def _initialize_collections(self) -> Dict[str, chromadb.Collection]:
        """
        Crea tres colecciones separadas por tipo de contenido.
        
        Returns:
            Diccionario con colecciones {tipo: collection}
        """
        collections = {}
        
        for tipo in ['texto', 'tabla', 'imagen']:
            collection_name = f"fds_{tipo}s"
            
            collections[tipo] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": f"Chunks de {tipo} de FDS",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            print(f"Colección '{collection_name}' inicializada")
        
        return collections
    
    def load_chunks_from_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Carga chunks desde archivo JSON.
        
        Args:
            json_path: Ruta al archivo JSON de chunks
            
        Returns:
            Lista de chunks
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('chunks', [])
    
    def prepare_batch_for_collection(self, chunks: List[Dict]) -> Dict[str, List]:
        """
        Prepara datos en formato batch para ChromaDB.
        
        Args:
            chunks: Lista de chunks del mismo tipo
            
        Returns:
            Dict con keys: ids, embeddings, documents, metadatas
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk['id'])
            embeddings.append(chunk['embedding'])
            documents.append(chunk['content'])
            
            metadata = chunk['metadata'].copy()
            
            # ChromaDB solo acepta: str, int, float, bool
            # Convertir listas a strings separados por comas
            for key, value in metadata.items():
                if isinstance(value, list):
                    metadata[key] = ','.join(map(str, value))
                elif value is None:
                    metadata[key] = ''
            
            metadatas.append(metadata)
        
        return {
            'ids': ids,
            'embeddings': embeddings,
            'documents': documents,
            'metadatas': metadatas
        }
    
    def ingest_document(self, json_path: Path) -> Dict[str, int]:
        """
        Ingesta todos los chunks de un documento.
        
        Args:
            json_path: Ruta al JSON de chunks procesados
            
        Returns:
            Conteo de chunks por tipo
        """
        chunks = self.load_chunks_from_json(json_path)
        
        # Agrupar chunks por tipo de contenido
        chunks_by_type = {
            'texto': [],
            'tabla': [],
            'imagen': []
        }
        
        for chunk in chunks:
            tipo = chunk['metadata'].get('tipo_contenido', 'texto')
            if tipo in chunks_by_type:
                chunks_by_type[tipo].append(chunk)
        
        # Insertar en cada colección
        counts = {}
        for tipo, tipo_chunks in chunks_by_type.items():
            if not tipo_chunks:
                counts[tipo] = 0
                continue
            
            batch_data = self.prepare_batch_for_collection(tipo_chunks)
            collection = self.collections[tipo]
            
            collection.add(
                ids=batch_data['ids'],
                embeddings=batch_data['embeddings'],
                documents=batch_data['documents'],
                metadatas=batch_data['metadatas']
            )
            
            counts[tipo] = len(tipo_chunks)
        
        return counts
    
    def get_collection_stats(self) -> Dict[str, Dict]:
        """
        Obtiene estadísticas de todas las colecciones.
        
        Returns:
            Dict con stats por colección
        """
        stats = {}
        
        for tipo, collection in self.collections.items():
            count = collection.count()
            stats[tipo] = {
                'name': collection.name,
                'count': count,
                'metadata': collection.metadata
            }
        
        return stats
    
    def verify_ingestion(self, chunk_id: str, tipo: str = 'texto') -> Dict:
        """
        Verifica que un chunk específico fue ingestado correctamente.
        
        Args:
            chunk_id: ID del chunk a verificar
            tipo: Tipo de contenido (texto, tabla, imagen)
            
        Returns:
            Datos del chunk recuperado
        """
        collection = self.collections[tipo]
        
        result = collection.get(
            ids=[chunk_id],
            include=['embeddings', 'documents', 'metadatas']
        )
        
        if not result['ids']:
            return None
        
        return {
            'id': result['ids'][0],
            'document': result['documents'][0],
            'metadata': result['metadatas'][0],
            'embedding_dim': len(result['embeddings'][0]) if len(result['embeddings']) > 0 else 0
        }


def main():
    """Ejecuta ingesta de todos los chunks procesados."""
    from config import ProjectConfig
    
    config = ProjectConfig()
    
    # Inicializar ingestor (reset_db=True solo primera vez)
    print("Inicializando ChromaDB...")
    ingestor = ChromaDBIngestor(config, reset_db=False)
    
    # Buscar archivos de chunks
    chunks_folder = config.get_folder('processed_chunks')
    chunk_files = list(chunks_folder.glob("*_chunks.json"))
    
    if not chunk_files:
        print(f"No se encontraron archivos en {chunks_folder}")
        return
    
    print(f"\nEncontrados {len(chunk_files)} documentos para ingestar\n")
    
    # Ingestar cada documento
    total_counts = {'texto': 0, 'tabla': 0, 'imagen': 0}
    
    for chunk_file in chunk_files:
        print(f"Ingiriendo: {chunk_file.name}")
        
        try:
            counts = ingestor.ingest_document(chunk_file)
            
            for tipo, count in counts.items():
                total_counts[tipo] += count
            
            print(f"  Texto: {counts['texto']}, Tablas: {counts['tabla']}, Imágenes: {counts['imagen']}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Mostrar estadísticas finales
    print("\n" + "="*50)
    print("RESUMEN DE INGESTA")
    print("="*50)
    
    stats = ingestor.get_collection_stats()
    
    for tipo, tipo_stats in stats.items():
        print(f"\nColección: {tipo_stats['name']}")
        print(f"  Total chunks: {tipo_stats['count']}")
    
    print(f"\nTotal general: {sum(s['count'] for s in stats.values())} chunks")
    print(f"Base de datos guardada en: {config.get_folder('vector_db')}")
    
    # Verificar primer chunk de texto como prueba
    if stats['texto']['count'] > 0:
        print("\n" + "="*50)
        print("VERIFICACIÓN DE MUESTRA")
        print("="*50)
        
        # Obtener primer chunk
        first_collection = ingestor.collections['texto']
        sample = first_collection.get(limit=1, include=['metadatas'])
        
        if sample['ids']:
            chunk_id = sample['ids'][0]
            verified = ingestor.verify_ingestion(chunk_id, 'texto')
            
            if verified:
                print(f"\nChunk ID: {verified['id']}")
                print(f"Producto: {verified['metadata'].get('producto', 'N/A')}")
                print(f"Sección: {verified['metadata'].get('seccion', 'N/A')}")
                print(f"Dimensión embedding: {verified['embedding_dim']}")
                print(f"Preview: {verified['document'][:200]}...")


if __name__ == "__main__":
    main()