"""
rag_retriever.py

Módulo para recuperación de información del RAG.
Implementa búsqueda híbrida con filtros de metadata y ranking.
"""

import chromadb
import ollama
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Representa un resultado de búsqueda."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    tipo_contenido: str


class QueryRouter:
    """Determina qué colecciones buscar según la query."""
    
    KEYWORDS_TABLA = [
        'tabla', 'componente', 'concentración', 'porcentaje', 'cas',
        'límite', 'valor', 'composición', 'ingrediente'
    ]
    
    KEYWORDS_IMAGEN = [
        'pictograma', 'símbolo', 'etiqueta', 'señal', 'ghs',
        'advertencia visual', 'imagen'
    ]
    
    @staticmethod
    def route_query(query: str) -> List[str]:
        """
        Determina en qué colecciones buscar.
        
        Args:
            query: Pregunta del usuario
            
        Returns:
            Lista de tipos de colección ['texto', 'tabla', 'imagen']
        """
        query_lower = query.lower()
        collections = ['texto']
        
        if any(kw in query_lower for kw in QueryRouter.KEYWORDS_TABLA):
            collections.append('tabla')
        
        if any(kw in query_lower for kw in QueryRouter.KEYWORDS_IMAGEN):
            collections.append('imagen')
        
        return collections


class MetadataFilter:
    """Construye filtros de metadata para ChromaDB."""
    
    @staticmethod
    def build_filter(
        producto: Optional[str] = None,
        fabricante: Optional[str] = None,
        seccion: Optional[str] = None,
        codigo_cas: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Construye diccionario de filtros para ChromaDB.
        
        Args:
            producto: Nombre del producto
            fabricante: Nombre del fabricante
            seccion: Sección específica de FDS
            codigo_cas: Número CAS de componente
            
        Returns:
            Dict de filtros o None si no hay filtros
        """
        conditions = []
        
        if producto:
            conditions.append({"producto": {"$eq": producto}})
        
        if fabricante:
            conditions.append({"fabricante": {"$eq": fabricante}})
        
        if seccion:
            conditions.append({"seccion": {"$eq": seccion}})
        
        if codigo_cas:
            conditions.append({"componentes_cas": {"$contains": codigo_cas}})
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        return {"$and": conditions}


class RAGRetriever:
    """Recuperador principal del sistema RAG."""
    
    def __init__(self, config, embedding_model: str = "nomic-embed-text"):
        """
        Args:
            config: Instancia de ProjectConfig
            embedding_model: Modelo de Ollama para embeddings
        """
        self.config = config
        self.embedding_model = embedding_model
        
        db_path = config.get_folder('vector_db')
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        self.collections = {
            'texto': self.client.get_collection('fds_textos'),
            'tabla': self.client.get_collection('fds_tablas'),
            'imagen': self.client.get_collection('fds_imagens')
        }
        
        self.query_router = QueryRouter()
        self.metadata_filter = MetadataFilter()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Genera embedding de la query.
        
        Args:
            query: Pregunta del usuario
            
        Returns:
            Vector de embedding
        """
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=query
        )
        return response['embedding']
    
    def search_collection(
        self,
        collection_type: str,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Busca en una colección específica.
        
        Args:
            collection_type: Tipo de colección (texto, tabla, imagen)
            query_embedding: Vector de embedding de la query
            n_results: Número de resultados a retornar
            where_filter: Filtros de metadata
            
        Returns:
            Lista de SearchResult
        """
        collection = self.collections[collection_type]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        search_results = []
        
        if not results['ids'][0]:
            return search_results
        
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                similarity=1 - results['distances'][0][i],
                tipo_contenido=collection_type
            ))
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 5,
        producto: Optional[str] = None,
        fabricante: Optional[str] = None,
        seccion: Optional[str] = None,
        codigo_cas: Optional[str] = None,
        force_collections: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Búsqueda híbrida en múltiples colecciones con filtros.
        
        Args:
            query: Pregunta del usuario
            n_results: Número de resultados por colección
            producto: Filtro por producto
            fabricante: Filtro por fabricante
            seccion: Filtro por sección
            codigo_cas: Filtro por código CAS
            force_collections: Forzar búsqueda en colecciones específicas
            
        Returns:
            Lista combinada y rankeada de resultados
        """
        query_embedding = self.embed_query(query)
        
        collections_to_search = force_collections or self.query_router.route_query(query)
        
        where_filter = self.metadata_filter.build_filter(
            producto=producto,
            fabricante=fabricante,
            seccion=seccion,
            codigo_cas=codigo_cas
        )
        
        all_results = []
        
        for collection_type in collections_to_search:
            if collection_type not in self.collections:
                continue
            
            results = self.search_collection(
                collection_type=collection_type,
                query_embedding=query_embedding,
                n_results=n_results,
                where_filter=where_filter
            )
            
            all_results.extend(results)
        
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        
        return all_results[:n_results * len(collections_to_search)]
    
    def get_context_for_generation(
        self,
        search_results: List[SearchResult],
        max_tokens: int = 3000
    ) -> str:
        """
        Construye contexto para el generador.
        
        Args:
            search_results: Resultados de búsqueda
            max_tokens: Máximo de tokens en el contexto
            
        Returns:
            Contexto formateado
        """
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(search_results, 1):
            chunk_tokens = len(result.content) // 4
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            header = f"[Fragmento {i} - {result.tipo_contenido.upper()}]"
            metadata_line = f"Producto: {result.metadata.get('producto', 'N/A')} | Sección: {result.metadata.get('seccion', 'N/A')}"
            separator = "-" * 80
            
            context_part = f"{header}\n{metadata_line}\n{separator}\n{result.content}\n\n"
            
            context_parts.append(context_part)
            current_tokens += chunk_tokens
        
        return "\n".join(context_parts)


class RetrievalStats:
    """Estadísticas de recuperación."""
    
    @staticmethod
    def print_results(results: List[SearchResult]):
        """
        Imprime resultados de búsqueda formateados.
        
        Args:
            results: Lista de SearchResult
        """
        if not results:
            print("No se encontraron resultados")
            return
        
        print(f"\nEncontrados {len(results)} resultados:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{'='*80}")
            print(f"Resultado #{i}")
            print(f"{'='*80}")
            print(f"Tipo: {result.tipo_contenido.upper()}")
            print(f"Similaridad: {result.similarity:.4f}")
            print(f"Producto: {result.metadata.get('producto', 'N/A')}")
            print(f"Sección: {result.metadata.get('seccion', 'N/A')}")
            print(f"Chunk ID: {result.chunk_id}")
            print(f"\nContenido:")
            print(f"{result.content[:300]}...")
            print()


def main():
    """Interfaz de prueba del retriever."""
    from config import ProjectConfig
    
    config = ProjectConfig()
    retriever = RAGRetriever(config)
    
    print("RAG Retriever - Sistema de Búsqueda")
    print("="*80)
    
    test_queries = [
        "¿Cuáles son los componentes peligrosos del esmalte epóxico?",
        "¿Qué pictogramas de seguridad tiene el producto?",
        "Muéstrame la información sobre límites de exposición"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        results = retriever.hybrid_search(
            query=query,
            n_results=3
        )
        
        RetrievalStats.print_results(results)
        
        print("\nContexto para generador:")
        print("-"*80)
        context = retriever.get_context_for_generation(results, max_tokens=1000)
        print(context[:500] + "...")
    
    print("\n" + "="*80)
    print("Pruebas completadas")


if __name__ == "__main__":
    main()