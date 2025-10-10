"""
rag_retriever_advanced.py

Sistema RAG con búsqueda híbrida y Fast Reranking (basado en similitud),
además de Clarificación de Queries y Contexto Conversacional.
"""

import chromadb
import ollama
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """Resultado de búsqueda."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    tipo_contenido: str
    rank_score: Optional[float] = None


@dataclass
class RetrievalMetrics:
    """Métricas de evaluación de retrieval."""
    precision_at_k: Dict[int, float]
    mrr_at_10: float
    recall_at_k: Dict[int, float]
    latency_ms: float

# ==============================================================================
# CLASES DE CLARIFICACIÓN Y CONTEXTO
# ==============================================================================

@dataclass
class Clarification:
    """Representa una solicitud de clarificación."""
    needs_clarification: bool
    question: str
    reason: str
    suggestions: List[str]

class QueryClarifier:
    """Detecta ambigüedad y genera contra-preguntas."""
    
    AMBIGUOUS_TERMS = {
        'producto_generico': ['el producto', 'este producto', 'ese', 'componentes', 'ingredientes'],
        'info_generica': ['información', 'datos', 'detalles', 'qué tiene', 'cuéntame'],
        'limites_ambiguos': ['límites', 'exposición', 'valores'],
        'seguridad_vaga': ['seguridad', 'peligros', 'riesgos', 'precauciones']
    }
    
    SPECIFIC_SECTIONS = {
        'composicion': ['composición', 'componentes', 'ingredientes', 'cas'],
        'peligros': ['peligros', 'clasificación', 'pictogramas', 'códigos h'],
        'primeros_auxilios': ['primeros auxilios', 'intoxicación', 'contacto'],
        'incendio': ['incendio', 'fuego', 'extinción', 'combustión'],
        'manipulacion': ['manipulación', 'almacenamiento', 'precauciones'],
        'exposicion': ['exposición', 'protección', 'epi', 'ventilación'],
        'propiedades': ['propiedades', 'físicas', 'químicas', 'aspecto'],
        'toxicologia': ['toxicología', 'toxicidad', 'efectos salud'],
    }
    
    def detect_ambiguity(self, query: str, search_results: List[SearchResult], context_product: Optional[str]) -> Clarification:
        """
        Detecta si la query necesita clarificación.
        """
        query_lower = query.lower()
        
        # 1. Verificar múltiples productos en resultados
        if search_results:
            productos = set(r.metadata.get('producto') for r in search_results if r.metadata.get('producto'))
            if len(productos) > 2:
                return Clarification(
                    needs_clarification=True,
                    question=f"Encontré información de {len(productos)} productos diferentes. ¿Cuál te interesa?",
                    reason="multiple_products",
                    suggestions=list(productos)[:5]
                )
        
        # 2. Verificar términos genéricos de producto SÓLO si no hay contexto previo
        if not context_product and any(term in query_lower for term in self.AMBIGUOUS_TERMS['producto_generico']):
            if not self._has_specific_product(query_lower):
                return Clarification(
                    needs_clarification=True,
                    question="¿A qué producto químico te refieres?",
                    reason="generic_product",
                    suggestions=["Esmalte Epóxico", "Pintura Texturizada", "Diluyente Xileno"]
                )
        
        # 3. Verificar consulta de límites ambigua
        if any(term in query_lower for term in self.AMBIGUOUS_TERMS['limites_ambiguos']):
            if not any(spec in query_lower for spec in ['dnel', 'pnec', 'explosividad', 'onu']):
                return Clarification(
                    needs_clarification=True,
                    question="¿Qué tipo de límite necesitas?",
                    reason="ambiguous_limits",
                    suggestions=["Límites de exposición ocupacional", "Límites de explosividad"]
                )
        
        # 4. Verificar información demasiado general
        if any(term in query_lower for term in self.AMBIGUOUS_TERMS['info_generica']):
            matched_section = self._match_section(query_lower)
            if not matched_section:
                return Clarification(
                    needs_clarification=True,
                    question="¿Sobre qué aspecto específico necesitas información (composición, peligros, etc.)?",
                    reason="too_general",
                    suggestions=["Composición química", "Peligros y clasificación", "Manipulación y almacenamiento"]
                )
        
        return Clarification(
            needs_clarification=False,
            question="",
            reason="specific_enough",
            suggestions=[]
        )
    
    def _has_specific_product(self, query: str) -> bool:
        """Verifica si la query menciona un producto específico por nombre."""
        specific_indicators = ['epóxico', 'epoxi', 'uretano', 'alquídico', 'texturizada']
        return any(ind in query for ind in specific_indicators)
    
    def _match_section(self, query: str) -> Optional[str]:
        """Identifica si la query menciona una sección específica."""
        for section, keywords in self.SPECIFIC_SECTIONS.items():
            if any(kw in query for kw in keywords):
                return section
        return None
    
    def format_clarification_message(self, clarification: Clarification) -> str:
        """
        Formatea mensaje de clarificación para el usuario.
        """
        if not clarification.needs_clarification:
            return ""
        
        message = f"⚠️ {clarification.question}"
        
        if clarification.suggestions:
            message += "\n\nOpciones sugeridas:\n"
            for i, sugg in enumerate(clarification.suggestions, 1):
                message += f" - {sugg}\n"
        
        return message

class ConversationalContext:
    """Mantiene contexto de conversación para queries de seguimiento."""
    
    def __init__(self):
        self.history: List[Dict] = []
        self.current_product: Optional[str] = None
        
    def add_turn(self, query: str, entities: Dict):
        """
        Registra turno de conversación y actualiza el producto actual.
        SOLO actualiza current_product si se encuentra un producto específico en las entidades.
        """
        producto_extraido = entities.get('producto')

        # Lógica de preservación: Si el extractor no identificó un producto real (ej: 'Ninguno', 'Sin especificar', etc.), 
        # mantenemos el contexto anterior (self.current_product).
        if producto_extraido and producto_extraido.lower() not in ['sin especificar', 'ninguno', 'el producto', '']:
            self.current_product = producto_extraido
            
        self.history.append({'query': query, 'metadata': entities})
    
    def get_context_for_query(self, query: str) -> str:
        """
        Enriquece query con el contexto de producto conversacional ANTES de enviarla al rewriter.
        """
        # La nueva query es enriquecida con el contexto actual si existe.
        if self.current_product and not self._is_new_product_mentioned(query):
            # Usamos un formato que el rewriter puede parsear o que al menos ayuda al embedding.
            return f"{query} | producto: {self.current_product}"
        
        return query

    def _is_new_product_mentioned(self, query: str) -> bool:
        """Heurística simple para ver si la query ya contiene un producto real específico."""
        query_lower = query.lower()
        specific_indicators = ['epóxico', 'epoxi', 'uretano', 'alquídico', 'texturizada', 'diluyente', 'xileno']
        # Si la query contiene un indicador de producto específico, asumimos que el usuario está cambiando de tema.
        return any(ind in query_lower for ind in specific_indicators)


# ==============================================================================
# CLASES RAG ESENCIALES
# ==============================================================================

class FastReranker:
    """Re-ranker rápido basado en similarity original."""
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """
        Re-rankea usando similarity original de ChromaDB.
        """
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        for r in results[:top_k]:
            r.rank_score = r.similarity
        
        return results[:top_k]


class HybridRetriever:
    """Sistema de recuperación híbrido usando Fast Reranker."""
    
    def __init__(
        self,
        config,
        embedding_model: str = "nomic-embed-text"
    ):
        self.config = config
        self.embedding_model = embedding_model
        
        db_path = config.get_folder('vector_db')
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        self.collections = {
            'texto': self.client.get_collection('fds_textos'),
            'tabla': self.client.get_collection('fds_tablas'),
            'imagen': self.client.get_collection('fds_imagens')
        }
        
        self.reranker = FastReranker()
    
    def retrieve(
        self,
        query: str,
        n_candidates: int = 20,
        n_final: int = 5,
        collections: Optional[List[str]] = None,
        metadata_filter: Optional[Dict] = None
    ) -> Tuple[List[SearchResult], RetrievalMetrics]:
        """
        Pipeline de recuperación usando Fast Reranking.
        """
        import time
        start_time = time.time()
        
        query_embedding = self._embed(query)
        
        collections_to_search = collections or ['texto', 'tabla', 'imagen']
        
        all_candidates = []
        for coll_name in collections_to_search:
            if coll_name not in self.collections:
                continue
            
            results = self._search_collection(
                coll_name,
                query_embedding,
                n_candidates,
                metadata_filter
            )
            all_candidates.extend(results)
        
        all_candidates.sort(key=lambda x: x.similarity, reverse=True)
        top_candidates = all_candidates[:n_candidates]
        
        final_results = self.reranker.rerank(query, top_candidates, n_final)
        
        latency = (time.time() - start_time) * 1000
        
        metrics = self._compute_metrics(final_results, latency)
        
        return final_results, metrics
    
    def _search_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int,
        where_filter: Optional[Dict]
    ) -> List[SearchResult]:
        """Busca en colección individual."""
        collection = self.collections[collection_name]
        
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
                tipo_contenido=collection_name
            ))
        
        return search_results
    
    def _embed(self, text: str) -> List[float]:
        """Genera embedding."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Error al generar embedding con Ollama: {e}")
            # Devuelve un vector de ceros para evitar fallos de tipo
            return [0.0] * 768


    def _compute_metrics(self, results: List[SearchResult], latency: float) -> RetrievalMetrics:
        """Calcula métricas de evaluación usando 0.7 como umbral de relevancia."""
        k_values = [1, 3, 5, 10]
        
        precision_at_k = {}
        for k in k_values:
            relevant = sum(1 for r in results[:k] if r.similarity > 0.7)
            precision_at_k[k] = relevant / k if k <= len(results) and k > 0 else 0
        
        reciprocal_ranks = []
        # MRR@10: Reciprocal Rank del primer resultado relevante (sim > 0.7) en top 10
        for i, result in enumerate(results[:10], 1):
            if result.similarity > 0.7:
                reciprocal_ranks.append(1 / i)
                break
        
        mrr_at_10 = reciprocal_ranks[0] if reciprocal_ranks else 0
        
        recall_at_k = {}
        # Asume que todos los resultados con similarity > 0.7 son el "set total de relevantes"
        total_relevant = sum(1 for r in results if r.similarity > 0.7)
        for k in k_values:
            relevant_at_k = sum(1 for r in results[:k] if r.similarity > 0.7)
            recall_at_k[k] = relevant_at_k / total_relevant if total_relevant > 0 else 0
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            mrr_at_10=mrr_at_10,
            recall_at_k=recall_at_k,
            latency_ms=latency
        )


class EntityExtractor:
    """Extractor de entidades de queries."""
    
    ENTITY_SCHEMA = {
        "producto": {"type": "string", "examples": ["Esmalte Epóxico", "Pintura Texturizada"]},
        "componente_quimico": {"type": "string", "examples": ["Xileno", "Acetato de butilo"]},
        "codigo_cas": {"type": "string", "pattern": r"\d+-\d+-\d+"},
        "seccion_fds": {"type": "string", "enum": [
            "composicion", "peligros", "primeros_auxilios", "incendio",
            "manipulacion", "exposicion", "propiedades", "toxicologia"
        ]},
        "tipo_informacion": {"type": "string", "enum": [
            "componentes", "concentracion", "peligros", "pictogramas",
            "limites_exposicion", "propiedades_fisicas", "transporte"
        ]}
    }
    
    def __init__(self, model: str = "mistral:7b"):
        self.model = model
    
    def extract(self, query: str) -> Dict[str, Any]:
        prompt = f"""Extrae información estructurada de esta pregunta sobre fichas de seguridad química.

Pregunta: {query}

Identifica:
- producto: nombre del producto químico
- componente_quimico: componente específico mencionado
- seccion_fds: sección de FDS relevante
- tipo_informacion: tipo de información solicitada

Responde SOLO en formato JSON sin explicaciones.
"""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={'temperature': 0},
                format='json'
            )
            import json
            entities = json.loads(response['response'])
            validated = self._validate_entities(entities)
            return validated
        except Exception as e:
            # print(f"Error en EntityExtractor: {e}") # Se comenta para mantener la salida limpia
            return {}
            
    def _validate_entities(self, entities: Dict) -> Dict:
        validated = {}
        for key, value in entities.items():
            if key in self.ENTITY_SCHEMA and value:
                schema = self.ENTITY_SCHEMA[key]
                if 'enum' in schema:
                    if value.lower() in [e.lower() for e in schema['enum']]:
                        validated[key] = value
                else:
                    validated[key] = value
        return validated

class QueryRewriter:
    """Reescribe queries ambiguas con contexto extraído."""
    
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
    
    def rewrite(self, query: str) -> Tuple[str, Dict, float]:
        """
        Reescribe query agregando contexto.
        """
        entities = self.entity_extractor.extract(query)
        
        if not entities:
            return query, {}, 0.0
        
        enriched_parts = [query]
        
        if entities.get('producto'):
            enriched_parts.append(f"producto: {entities['producto']}")
        
        if entities.get('seccion_fds'):
            enriched_parts.append(f"sección: {entities['seccion_fds']}")
        
        enriched_query = " | ".join(enriched_parts)
        
        specificity = len(entities) / len(self.entity_extractor.ENTITY_SCHEMA)
        
        return enriched_query, entities, specificity

# ==============================================================================
# FUNCIÓN MAIN Y LÓGICA DE INTERACCIÓN
# ==============================================================================

def main():
    """Prueba del sistema avanzado con contexto y clarificación."""
    from config import ProjectConfig
    
    config = ProjectConfig()
    
    retriever = HybridRetriever(config, embedding_model="nomic-embed-text")
    entity_extractor = EntityExtractor()
    query_rewriter = QueryRewriter(entity_extractor)
    clarifier = QueryClarifier()
    context = ConversationalContext()

    print("SISTEMA RAG CON CONTEXTO Y CLARIFICACIÓN")
    print("=" * 80)
    
    # Simulación de conversación:
    conversation_steps = [
        "Cuales son los componentes peligrosos del esmalte epoxico?",
        "Dame sus límites de exposición",
        "Dime todo sobre el producto",
        "¿Qué componentes tiene?",
        "¿Qué componentes tiene el diluyente Xileno?" 
    ]
    
    for i, query_original in enumerate(conversation_steps):
        print(f"\n--- Turno {i+1} ---")
        print(f"Query Original: {query_original}")

        # 1. Aplicar contexto conversacional (si aplica)
        query_with_context = context.get_context_for_query(query_original)
        
        # 2. Extracción de entidades y reescritura
        rewritten_for_search, entities, specificity = query_rewriter.rewrite(query_with_context)

        # 3. Detectar si la query original es ambigua
        clarification = clarifier.detect_ambiguity(
            query_original, 
            search_results=[], 
            context_product=context.current_product
        )

        if clarification.needs_clarification:
            print(f"**NECESITA CLARIFICACIÓN ({clarification.reason})**")
            print(clarifier.format_clarification_message(clarification))
            continue
        
        print(f"Query Final de Búsqueda: {rewritten_for_search}")
        
        # 4. Retrieval (Fast Reranker)
        results, metrics = retriever.retrieve(
            rewritten_for_search,
            n_candidates=20,
            n_final=5,
            metadata_filter=None
        )
        
        # 5. Actualizar Contexto
        # Solo se actualiza si la query no era ambigua y se hizo retrieval
        context.add_turn(query_original, entities)
        
        # 6. Mostrar métricas y resultados (Formato de salida restaurado)
        print(f"\nMetricas:")
        print(f"  P@5: {metrics.precision_at_k[5]:.3f}")
        print(f"  MRR@10: {metrics.mrr_at_10:.3f}")
        print(f"  Latencia: {metrics.latency_ms:.1f}ms")

        print("\nTop 3 resultados:")
        for j, r in enumerate(results[:3], 1):
            # Usamos rank_score que es igual a similarity en FastReranker
            print(f"  {j}. Similarity: {r.similarity:.4f} | Rank: {r.rank_score:.4f}")
            print(f"     [Producto: {r.metadata.get('producto', 'N/A')} | Tipo: {r.tipo_contenido}]")
            print(f"     Fragmento: {r.content[:100]}...")
        
        # Muestra el contexto actual
        print(f"\n> Contexto actualizado: Producto actual es '{context.current_product}'")


if __name__ == "__main__":
    main()