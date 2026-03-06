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
import warnings
import logging

# ✅ Suprimir warnings de telemetría de ChromaDB
warnings.filterwarnings("ignore")
logging.getLogger('chromadb').setLevel(logging.CRITICAL)


class ChromaDBIngestor:
    """Gestiona la ingesta de chunks en ChromaDB."""
    
    def __init__(self, config, reset_db: bool = False):
        """
        Args:
            config: Instancia de ProjectConfig
            reset_db: Si True, elimina base de datos existente
        """
        self.config = config
        self.db_path = config.get_folder('vector_db')
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # ✅ SOLO eliminar si reset_db=True explícitamente
        if reset_db:
            print(f"⚠️  RESET MODE: Eliminando base de datos existente...")
            import shutil
            if self.db_path.exists():
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
            
            print(f"📚 Colección '{collection_name}' lista (total: {collections[tipo].count()} chunks)")
        
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
    
    def ingest_document(self, json_path: Path, skip_existing: bool = True) -> Dict[str, int]:
        """
        Ingesta todos los chunks de un documento.
        
        Args:
            json_path: Ruta al JSON de chunks procesados
            skip_existing: Si True, omite chunks que ya existen en la BD
            
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
        counts = {'texto': 0, 'tabla': 0, 'imagen': 0, 'skipped': 0}
        
        for tipo, tipo_chunks in chunks_by_type.items():
            if not tipo_chunks:
                counts[tipo] = 0
                continue
            
            collection = self.collections[tipo]
            
            # ✅ Verificar chunks existentes
            if skip_existing:
                try:
                    # Obtener IDs existentes en la colección
                    existing_result = collection.get(include=[])
                    existing_ids = set(existing_result['ids']) if existing_result['ids'] else set()
                    
                    # Filtrar chunks nuevos
                    new_chunks = [c for c in tipo_chunks if c['id'] not in existing_ids]
                    skipped = len(tipo_chunks) - len(new_chunks)
                    
                    if skipped > 0:
                        counts['skipped'] += skipped
                        print(f"  ⏭️  {skipped} chunks de {tipo} ya existen (omitidos)")
                    
                    tipo_chunks = new_chunks
                    
                except Exception as e:
                    print(f"  ⚠️  Error verificando IDs: {e}")
                    # Si falla la verificación, continuar sin filtrar
            
            # Si no hay chunks nuevos, continuar
            if not tipo_chunks:
                counts[tipo] = 0
                continue
            
            # Preparar e insertar
            batch_data = self.prepare_batch_for_collection(tipo_chunks)
            
            try:
                collection.add(
                    ids=batch_data['ids'],
                    embeddings=batch_data['embeddings'],
                    documents=batch_data['documents'],
                    metadatas=batch_data['metadatas']
                )
                counts[tipo] = len(tipo_chunks)
                print(f"  ✅ {counts[tipo]} chunks de {tipo} añadidos")
            except Exception as e:
                print(f"  ❌ Error al añadir chunks de {tipo}: {e}")
                counts[tipo] = 0
        
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
            'embedding_dim': len(result['embeddings'][0]) if result['embeddings'] else 0
        }


def main():
    """Ejecuta ingesta de todos los chunks procesados."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))  
    from config import ProjectConfig

    config = ProjectConfig()
    
    print("🚀 Inicializando ChromaDB (modo incremental)...")
    print("   Si necesitas resetear todo, usa: cleanup_chromadb.py\n")
    
    ingestor = ChromaDBIngestor(config, reset_db=False)
    
    # Buscar archivos de chunks
    chunks_folder = config.get_folder('processed_chunks')
    chunk_files = list(chunks_folder.glob("*_chunks.json"))
    
    if not chunk_files:
        print(f"❌ No se encontraron archivos en {chunks_folder}")
        return
    
    print(f"📁 Encontrados {len(chunk_files)} documentos\n")
    
    # Ingestar cada documento
    total_counts = {'texto': 0, 'tabla': 0, 'imagen': 0, 'skipped': 0}
    
    for i, chunk_file in enumerate(chunk_files, 1):
        print(f"[{i}/{len(chunk_files)}] {chunk_file.name}")
        
        try:
            counts = ingestor.ingest_document(chunk_file, skip_existing=True)
            
            for tipo, count in counts.items():
                total_counts[tipo] += count
            
            print(f"     Texto: {counts['texto']}, Tablas: {counts['tabla']}, Imágenes: {counts['imagen']}\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    # Estadísticas finales
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL")
    print("="*60)
    
    stats = ingestor.get_collection_stats()
    
    for tipo, tipo_stats in stats.items():
        print(f"\n📚 {tipo_stats['name']}: {tipo_stats['count']} chunks")
    
    print(f"\n✅ Total en BD: {sum(s['count'] for s in stats.values())} chunks")
    print(f"➕ Nuevos añadidos: {sum(total_counts[k] for k in ['texto', 'tabla', 'imagen'])}")
    print(f"⏭️  Omitidos (duplicados): {total_counts['skipped']}")
    print(f"\n💾 Base de datos: {config.get_folder('vector_db')}")
    
    # Verificación de muestra
    if stats['texto']['count'] > 0:
        print("\n" + "="*60)
        print("🔍 VERIFICACIÓN DE MUESTRA")
        print("="*60)
        
        first_collection = ingestor.collections['texto']
        sample = first_collection.get(limit=1, include=['metadatas'])
        
        if sample['ids']:
            chunk_id = sample['ids'][0]
            verified = ingestor.verify_ingestion(chunk_id, 'texto')
            
            if verified:
                print(f"\n📄 Chunk ID: {verified['id']}")
                print(f"   Producto: {verified['metadata'].get('producto', 'N/A')}")
                print(f"   Sección: {verified['metadata'].get('seccion', 'N/A')}")
                print(f"   Embedding: {verified['embedding_dim']}D")
                print(f"   Preview: {verified['document'][:150]}...")


if __name__ == "__main__":
    main()