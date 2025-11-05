"""
rag_retriever_advanced.py

Sistema RAG con búsqueda híbrida, Fast Reranking, Contexto Conversacional,
Clarificación de Queries y Extracción de Metadata Estructurada.

CORRECCIÓN CLAVE: 
- El filtro de producto de contexto se aplica ahora como un filtro de 
  METADATOS ESTRICTO (where) en la clave 'producto' para asegurar 
  que los chunks relevantes sean recuperados en los turnos de seguimiento.
- El filtro where_document (sobre el texto) está desactivado para evitar 
  descartar chunks válidos.
"""

import chromadb
import ollama
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import time


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
class StructuredMetadata:
    """Metadata estructurada extraída de chunks."""
    query_type: str
    codigos_h: List[Dict[str, str]] = field(default_factory=list)
    codigos_p: List[Dict[str, str]] = field(default_factory=list)
    componentes_cas: List[Dict[str, str]] = field(default_factory=list)
    extracted_from_chunks: int = 0

@dataclass
class RetrievalMetrics:
    """Métricas de evaluación de retrieval."""
    precision_at_k: Dict[int, float]
    mrr_at_10: float
    recall_at_k: Dict[int, float]
    latency_ms: float


@dataclass
class Clarification:
    """Representa una solicitud de clarificación."""
    needs_clarification: bool
    question: str
    reason: str
    suggestions: List[str]


# ----------------------------------------------------------------------
# SectionMapper (Filtro Desactivado)
# ----------------------------------------------------------------------
class SectionMapper:
    """Mapea keywords a secciones de metadata de FDS (código omitido por brevedad, es el mismo)."""
    
    SECTION_MAPPING = {
        'identificacion': {
            'keywords_es': ['identificación', 'proveedor', 'fabricante', 'distribuidor', 'emergencia', 'contacto'],
            'keywords_en': ['identification', 'supplier', 'manufacturer', 'distributor', 'emergency', 'contact'],
            'sections': ['identificacion_producto']
        },
        'peligros': {
            'keywords_es': ['peligro', 'riesgo', 'clasificación', 'pictograma', 'ghs', 'sga', 'codigo h', 'h226', 'h304', 'h316', 'h317', 'h336', 'h412', 'palabra advertencia', 'indicacion peligro'],
            'keywords_en': ['hazard', 'risk', 'classification', 'pictogram', 'ghs', 'h code', 'signal word', 'hazard statement'],
            'sections': ['identificacion_peligros']
        },
        'componentes': {
            'keywords_es': ['componente', 'composición', 'ingrediente', 'cas', 'concentración', 'xileno', 'acetato', 'metacrilato', 'sustancia', 'porcentaje', 'mezcla', 'familia quimica'],
            'keywords_en': ['component', 'composition', 'ingredient', 'cas', 'concentration', 'substance', 'percentage', 'mixture', 'chemical family'],
            'sections': ['composicion_componentes']
        },
        'primeros_auxilios': {
            'keywords_es': ['primeros auxilios', 'auxilio', 'intoxicación', 'ingestión', 'contacto', 'inhalación', 'ojos', 'piel', 'sintomas', 'medico', 'tratamiento'],
            'keywords_en': ['first aid', 'ingestion', 'contact', 'inhalation', 'eyes', 'skin', 'symptoms', 'physician', 'treatment'],
            'sections': ['primeros_auxilios']
        },
        'incendio': {
            'keywords_es': ['incendio', 'fuego', 'extinción', 'combustión', 'inflamable', 'punto inflamacion', 'extintor', 'bombero', 'llama'],
            'keywords_en': ['fire', 'firefighting', 'combustion', 'flammable', 'flash point', 'extinguisher', 'firefighter', 'flame'],
            'sections': ['medidas_contra_incendios']
        },
        'vertido': {
            'keywords_es': ['vertido', 'derrame', 'fuga', 'accidental', 'limpieza', 'contención', 'absorbente'],
            'keywords_en': ['spill', 'leak', 'accidental', 'cleanup', 'containment', 'absorbent'],
            'sections': ['vertido_accidental']
        },
        'manipulacion': {
            'keywords_es': ['manipulación', 'almacenamiento', 'manejo', 'precaución', 'higiene', 'almacenar', 'ventilacion'],
            'keywords_en': ['handling', 'storage', 'precaution', 'hygiene', 'store', 'ventilation'],
            'sections': ['manipulacion_almacenamiento']
        },
        'exposicion': {
            'keywords_es': ['exposición', 'protección', 'epi', 'ppe', 'límite', 'twa', 'stel', 'ventilación', 'guantes', 'respirador', 'limite exposicion', 'dnel', 'pnec', 'ocupacional'],
            'keywords_en': ['exposure', 'protection', 'ppe', 'limit', 'twa', 'stel', 'ventilacion', 'guantes', 'respirator', 'exposure limit', 'dnel', 'pnec', 'occupational'],
            'sections': ['controles_exposicion_proteccion']
        },
        'propiedades': {
            'keywords_es': ['propiedades', 'física', 'química', 'aspecto', 'densidad', 'punto', 'ebullición', 'fusión', 'viscosidad', 'ph', 'olor', 'color', 'estado'],
            'keywords_en': ['properties', 'physical', 'chemical', 'appearance', 'density', 'point', 'boiling', 'melting', 'viscosity', 'ph', 'odor', 'color', 'state'],
            'sections': ['propiedades_fisicas_quimicas']
        },
        'estabilidad': {
            'keywords_es': ['estabilidad', 'reactividad', 'incompatible', 'descomposición', 'reaccion peligrosa', 'polimerizacion'],
            'keywords_en': ['stability', 'reactivity', 'incompatible', 'decomposition', 'hazardous reaction', 'polymerization'],
            'sections': ['estabilidad_reactividad']
        },
        'toxicologia': {
            'keywords_es': ['toxicología', 'toxicidad', 'efectos', 'salud', 'dl50', 'cl50', 'carcinogeno', 'mutagenico', 'sensibilizante'],
            'keywords_en': ['toxicology', 'toxicity', 'effects', 'health', 'ld50', 'lc50', 'carcinogen', 'mutagenic', 'sensitizer'],
            'sections': ['informacion_toxicologica']
        },
        'ecologia': {
            'keywords_es': ['ecología', 'ecológica', 'ambiental', 'acuático', 'biodegradación', 'bioacumulación', 'pez', 'alga', 'dafnia'],
            'keywords_en': ['ecology', 'ecological', 'environmental', 'aquatic', 'biodegradation', 'bioaccumulation', 'fish', 'algae', 'daphnia'],
            'sections': ['informacion_ecologica']
        },
        'eliminacion': {
            'keywords_es': ['eliminación', 'disposición', 'residuo', 'desecho', 'tratamiento residuo', 'incineración'],
            'keywords_en': ['disposal', 'waste', 'treatment', 'incineration'],
            'sections': ['consideraciones_eliminacion']
        },
        'transporte': {
            'keywords_es': ['transporte', 'onu', 'adr', 'imdg', 'iata', 'embalaje', 'numero onu', 'clase transporte'],
            'keywords_en': ['transport', 'un', 'adr', 'imdg', 'iata', 'packaging', 'un number', 'transport class'],
            'sections': ['informacion_transporte']
        },
        'reglamentaria': {
            'keywords_es': ['reglamentaria', 'legislación', 'normativa', 'reach', 'cov', 'regulación'],
            'keywords_en': ['regulatory', 'legislation', 'regulation', 'reach', 'voc'],
            'sections': ['informacion_reglamentaria']
        },
        'otra': {
            'keywords_es': ['otra información', 'abreviatura', 'acrónimo', 'aviso', 'glosario'],
            'keywords_en': ['other information', 'abbreviation', 'acronym', 'notice', 'glosario'],
            'sections': ['otra_informacion']
        }
    }
    
    def detect_section(self, query: str) -> Optional[List[str]]:
        """Detecta secciones relevantes desde la query."""
        query_lower = query.lower()
        
        for section_key, config in self.SECTION_MAPPING.items():
            all_keywords = config['keywords_es'] + config['keywords_en']
            
            if any(kw in query_lower for kw in all_keywords):
                return config['sections']
        
        return None
    
    def build_section_filter(self, sections: List[str]) -> Optional[Dict]:
        """
        Retorna None para DESACTIVAR el filtro de sección.
        """
        return None


# ----------------------------------------------------------------------
# MetadataExtractor (Funciones de agregación robustas)
# ----------------------------------------------------------------------
class MetadataExtractor:
    """Extrae y estructura metadata de chunks recuperados (código omitido por brevedad, es el mismo)."""
    
    QUERY_TYPE_MAPPING = {
        'peligros': {
            'keywords_es': ['peligro', 'riesgo', 'codigo h', 'h226', 'h304', 'h316', 'h317', 'h336', 'h412', 'clasificacion', 'pictograma'],
            'keywords_en': ['hazard', 'danger', 'h code', 'classification', 'pictogram'],
            'metadata_fields': ['codigos_h']
        },
        'precauciones': {
            'keywords_es': ['precaucion', 'codigo p', 'p201', 'p210', 'p241', 'p260', 'p273', 'p280', 'medida', 'prevencion'],
            'keywords_en': ['precaution', 'p code', 'precautionary', 'measure', 'prevention'],
            'metadata_fields': ['codigos_p']
        },
        'componentes': {
            'keywords_es': ['componente', 'composicion', 'ingrediente', 'cas', 'sustancia', 'concentracion', 'xileno', 'acetato', 'metacrilato'],
            'keywords_en': ['component', 'composition', 'ingredient', 'cas', 'substance', 'concentration'],
            'metadata_fields': ['componentes_cas']
        }
    }
    
    def detect_query_type(self, query: str) -> Optional[str]:
        """Detecta tipo de información solicitada."""
        query_lower = query.lower()
        
        for query_type, config in self.QUERY_TYPE_MAPPING.items():
            all_keywords = config['keywords_es'] + config['keywords_en']
            
            if any(kw in query_lower for kw in all_keywords):
                return query_type
        
        return None
    
    def extract(self, results: List[SearchResult], query_type: str) -> StructuredMetadata:
        """Extrae metadata estructurada de chunks recuperados."""
        
        if query_type == 'peligros':
            codigos_h = self._aggregate_codigos_h(results)
            return StructuredMetadata(
                query_type='peligros',
                codigos_h=codigos_h,
                extracted_from_chunks=len(results)
            )
        
        elif query_type == 'precauciones':
            codigos_p = self._aggregate_codigos_p(results)
            return StructuredMetadata(
                query_type='precauciones',
                codigos_p=codigos_p,
                extracted_from_chunks=len(results)
            )
        
        elif query_type == 'componentes':
            componentes = self._aggregate_componentes(results)
            return StructuredMetadata(
                query_type='componentes',
                componentes_cas=componentes,
                extracted_from_chunks=len(results)
            )
        
        return StructuredMetadata(query_type='unknown')
    
    def _aggregate_codigos_h(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """
        ROBUSTO: Agrega códigos H únicos, validando el tipo de dato.
        """
        codigos_h_dict = {}
        
        for result in results:
            codigos_h = result.metadata.get('codigos_h', [])
            
            if not isinstance(codigos_h, list):
                continue

            for codigo in codigos_h:
                if not isinstance(codigo, dict):
                    continue
                    
                codigo_id = codigo.get('codigo')
                if codigo_id and codigo_id not in codigos_h_dict:
                    codigos_h_dict[codigo_id] = {
                        'codigo': codigo_id,
                        'descripcion': codigo.get('descripcion', '')
                    }
        
        return sorted(codigos_h_dict.values(), key=lambda x: x['codigo'])
    
    def _aggregate_codigos_p(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """
        ROBUSTO: Agrega códigos P únicos, validando el tipo de dato.
        """
        codigos_p_dict = {}
        
        for result in results:
            codigos_p = result.metadata.get('codigos_p', [])
            
            if not isinstance(codigos_p, list):
                continue
            
            for codigo in codigos_p:
                if not isinstance(codigo, dict):
                    continue
                    
                codigo_id = codigo.get('codigo')
                if codigo_id and codigo_id not in codigos_p_dict:
                    codigos_p_dict[codigo_id] = {
                        'codigo': codigo_id,
                        'descripcion': codigo.get('descripcion', '')
                    }
        
        return sorted(codigos_p_dict.values(), key=lambda x: x['codigo'])
    
    def _aggregate_componentes(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """
        ROBUSTO: Agrega componentes CAS únicos con concentraciones, validando el tipo de dato.
        """
        componentes_dict = {}
        
        for result in results:
            componentes_cas = result.metadata.get('componentes_cas', [])
            
            if not isinstance(componentes_cas, list):
                continue
            
            for comp in componentes_cas:
                if not isinstance(comp, dict):
                    continue
                    
                cas_id = comp.get('cas')
                if cas_id and cas_id not in componentes_dict:
                    componentes_dict[cas_id] = {
                        'nombre': comp.get('nombre', ''),
                        'cas': cas_id,
                        'concentracion': comp.get('concentracion', 'No especificada')
                    }
        
        return sorted(componentes_dict.values(), key=lambda x: x['nombre'])


# ----------------------------------------------------------------------
# ConversationalContext
# ----------------------------------------------------------------------
class QueryClarifier:
    """Detecta ambigüedad y genera contra-preguntas (código omitido por brevedad, es el mismo)."""
    
    def detect_ambiguity(
        self,
        query: str,
        search_results: List[SearchResult],
        context_product: Optional[str]
    ) -> Clarification:
        """Detecta si la query necesita clarificación."""
        query_lower = query.lower()
        
        if search_results:
            productos = set(r.metadata.get('producto') for r in search_results if r.metadata.get('producto'))
            
            if len(productos) > 3:
                return Clarification(
                    needs_clarification=True,
                    question=f"Encontré información de {len(productos)} productos. ¿Cuál te interesa?",
                    reason="multiple_products",
                    suggestions=list(productos)[:5]
                )
        
        generic_terms = ['el producto', 'este producto', 'ese', 'producto']
        if not context_product and any(term in query_lower for term in generic_terms):
            if not self._has_specific_product(query_lower):
                return Clarification(
                    needs_clarification=True,
                    question="¿A qué producto químico te refieres?",
                    reason="generic_product",
                    suggestions=["Esmalte Epóxico", "Pintura Texturizada", "Esmalte Uretano"]
                )
        
        return Clarification(
            needs_clarification=False,
            question="",
            reason="specific_enough",
            suggestions=[]
        )
    
    def _has_specific_product(self, query: str) -> bool:
        """Verifica si menciona producto específico."""
        specific = ['epóxico', 'epoxi', 'uretano', 'alquídico', 'texturizada', 'xileno']
        return any(prod in query for prod in specific)


class ConversationalContext:
    """Mantiene contexto conversacional."""
    
    def __init__(self):
        self.current_product: Optional[str] = None
        self.history: List[Dict] = []
    
    def update_product(self, results: List[SearchResult]):
        """Actualiza producto actual desde resultados."""
        if results:
            producto = results[0].metadata.get('producto')
            if producto:
                self.current_product = producto
    
    def enrich_query(self, query: str) -> str:
        """
        Devuelve solo la pregunta clave para el embedding vectorial (Query Limpia).
        El filtrado por producto se hará a través del filtro de metadatos (HybridRetriever).
        """
        if not self.current_product:
            return query
        
        query_lower = query.lower()
        
        # Si es un seguimiento (Turnos 2, 3, 4: 'Y sus peligros?', 'Dame sus precauciones')
        if self._is_followup_query(query_lower):
            
            # Limpiar y estandarizar la pregunta clave
            if 'componente' in query_lower or 'composición' in query_lower:
                return "¿Cuáles son los componentes principales?"
            
            if 'peligro' in query_lower or 'riesgo' in query_lower:
                return "¿Cuáles son los peligros principales?"
            
            if 'precauc' in query_lower or 'codigo p' in query_lower:
                return "¿Cuáles son las precauciones de manejo?"
            
            # Para seguimientos generales, simplemente se elimina la primera palabra (y, sus, dame, etc.)
            words = query.split()
            return ' '.join(words[1:]) if len(words) > 1 else query
        
        return query
    
    def _is_followup_query(self, query: str) -> bool:
        """Detecta si es query de seguimiento."""
        followup_indicators = [
            'y', 'sus', 'su', 'ese', 'este', 'el mismo', 'también',
            'and', 'its', 'that', 'this', 'same', 'also', 'dame'
        ] 
        
        words = query.lower().split()
        if not words:
            return False
        
        return words[0] in followup_indicators or len(words) < 4

    def add_turn(self, query: str, results: List[SearchResult]):
        """Registra turno de conversación y actualiza el producto actual."""
        self.history.append({'query': query})
        self.update_product(results)


class FastReranker:
    """Re-ranker basado en similarity original (código omitido por brevedad, es el mismo)."""
    
    def rerank(self, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Re-rankea por similarity."""
        results.sort(key=lambda x: x.similarity, reverse=True)
        
        for r in results[:top_k]:
            r.rank_score = r.similarity
        
        return results[:top_k]


# ----------------------------------------------------------------------
# HybridRetriever MODIFICADO: Lógica de Filtro de Contexto a Metadatos
# ----------------------------------------------------------------------
class HybridRetriever:
    """Sistema de recuperación híbrido."""
    
    def __init__(self, config, embedding_model: str = "nomic-embed-text"):
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
        self.metadata_extractor = MetadataExtractor()
    
    def retrieve(
        self,
        query: str,
        n_candidates: int = 20,
        n_final: int = 5,
        producto: Optional[str] = None, # Producto en contexto
        section_filter: Optional[Dict] = None 
    ) -> Tuple[List[SearchResult], Optional[StructuredMetadata], RetrievalMetrics]:
        """Pipeline de recuperación con extracción de metadata."""
        start_time = time.time()
        
        query_embedding = self._embed(query)
        
        # 1. Determinar el producto a usar en el filtro
        producto_filtro = self._determine_product_filter(query, producto)
        
        # 2. Construir los filtros (producto_filtro va a where_metadata)
        where_metadata, where_document = self._build_combined_filter(producto_filtro, section_filter) 
        
        all_candidates = []
        for coll_name in ['texto', 'tabla', 'imagen']:
            results = self._search_collection(
                coll_name,
                query_embedding,
                n_candidates,
                where_metadata,
                where_document
            )
            all_candidates.extend(results)
        
        all_candidates.sort(key=lambda x: x.similarity, reverse=True)
        top_candidates = all_candidates[:n_candidates]
        
        final_results = self.reranker.rerank(top_candidates, n_final)
        
        query_type = self.metadata_extractor.detect_query_type(query)
        
        structured_metadata = None
        if query_type:
            structured_metadata = self.metadata_extractor.extract(final_results, query_type) 
        
        latency = (time.time() - start_time) * 1000
        metrics = self._compute_metrics(final_results, latency)
        
        return final_results, structured_metadata, metrics

    def _determine_product_filter(self, query: str, context_product: Optional[str]) -> Optional[str]:
        """
        Decide qué producto usar para el filtro de metadatos (contexto vs nueva query).
        """
        if not context_product:
            return None
        
        context_lower = context_product.lower()
        query_lower = query.lower()

        if ('uretano' in query_lower and 'epoxico' in context_lower) or \
           ('epoxico' in query_lower and 'uretano' in context_lower):
            return None
            
        return context_product # Si no hay cambio, se mantiene el filtro de contexto

    def _build_combined_filter(
        self,
        producto: Optional[str],
        section_filter: Optional[Dict]
    ) -> Tuple[Optional[Dict], Optional[Dict]]:

        # Inicializamos el filtro de metadatos con el filtro de sección (que es None)
        where_metadata = section_filter 
        where_document = None # Desactivamos el filtro sobre el texto
        
        if producto:
            # Filtro estricto: la clave 'producto' en los metadatos DEBE ser igual.
            product_filter = {"producto": {"$eq": producto}}
            
            if where_metadata:
                 # Si ya hay un filtro de metadata (sección), combinamos con AND.
                where_metadata = {"$and": [where_metadata, product_filter]}
            else:
                # Si no hay filtro de sección, el filtro de producto es el único filtro de metadata.
                where_metadata = product_filter
            
        return where_metadata, where_document
    
    def _search_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int,
        where_metadata: Optional[Dict],
        where_document: Optional[Dict] = None
    ) -> List[SearchResult]:
        
        """
        Busca en la colección usando los filtros de metadatos (where) y de texto (where_document).
        """
        collection = self.collections[collection_name]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_metadata, # Ahora contiene el filtro de producto
            where_document=where_document, 
            include=['documents', 'metadatas', 'distances']
        )
        
        search_results = []
        
        if not results['ids'] or not results['ids'][0]:
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
        """Genera embedding (código omitido por brevedad)."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response['embedding']
    
    def _compute_metrics(self, results: List[SearchResult], latency: float) -> RetrievalMetrics:
        """Calcula métricas (código omitido por brevedad)."""
        k_values = [1, 3, 5, 10]
        
        precision_at_k = {}
        for k in k_values:
            relevant = sum(1 for r in results[:k] if r.similarity > 0.7)
            precision_at_k[k] = relevant / k if k <= len(results) else 0
        
        reciprocal_ranks = []
        for i, result in enumerate(results[:10], 1):
            if result.similarity > 0.7:
                reciprocal_ranks.append(1 / i)
                break
        
        mrr_at_10 = reciprocal_ranks[0] if reciprocal_ranks else 0
        
        recall_at_k = {}
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


def main():
    """Sistema RAG con contexto y extracción de metadata (código omitido por brevedad, es el mismo)."""
    from config import ProjectConfig
    
    @dataclass
    class MockProjectConfig:
        def get_folder(self, name: str) -> Path:
            # AJUSTA ESTA RUTA A LA UBICACIÓN REAL DE TU BASE DE DATOS CHROMA
            return Path('./chroma_db_mock') 

    try:
        config = ProjectConfig()
    except:
        print("ADVERTENCIA: Usando Mock ProjectConfig. Asegúrate de que el path a la DB de Chroma es correcto y que OLLAMA está corriendo.")
        config = MockProjectConfig()

    
    retriever = HybridRetriever(config)
    context = ConversationalContext()
    clarifier = QueryClarifier()
    section_mapper = SectionMapper()
    
    print("SISTEMA RAG CON CONTEXTO Y METADATA ESTRUCTURADA")
    print("=" * 80)
    
    conversation = [
        "Cuales son los componentes peligrosos del esmalte epoxico?",
        "Y sus peligros?",
        "Dame sus precauciones",
        "Que componentes tiene?",
        "Que peligros tiene el esmalte uretano?"
    ]
    
    for i, query_original in enumerate(conversation, 1):
        print(f"\n--- Turno {i} ---")
        print(f"Query: {query_original}")
        
        # 1. Obtener la query limpia/reformulada para el embedding
        query_enriched = context.enrich_query(query_original)
        
        if query_enriched != query_original:
            print(f"Query enriquecida/limpia (para embedding): {query_enriched}")
        
        sections = section_mapper.detect_section(query_enriched)
        section_filter = section_mapper.build_section_filter(sections)
        
        if sections:
            print(f"Secciones detectadas (Filtro Desactivado): {sections}")
        
        # 2. Retrieval: El producto de contexto se usa para el filtro de METADATOS.
        results, structured_metadata, metrics = retriever.retrieve(
            query_enriched,
            n_candidates=20,
            n_final=5,
            producto=context.current_product,
            section_filter=section_filter
        )
        
        clarification = clarifier.detect_ambiguity(
            query_original,
            results,
            context.current_product
        )
        
        if clarification.needs_clarification:
            print(f"CLARIFICACION ({clarification.reason}): {clarification.question}")
            if clarification.suggestions:
                for sugg in clarification.suggestions:
                    print(f"  - {sugg}")
            continue
        
        context.add_turn(query_original, results)
        
        if structured_metadata and structured_metadata.extracted_from_chunks > 0:
            print(f"\nMetadata extraida ({structured_metadata.extracted_from_chunks} chunks):")
            
            if structured_metadata.codigos_h:
                print("\nCODIGOS H:")
                for codigo in structured_metadata.codigos_h[:5]:
                    print(f"  {codigo['codigo']}: {codigo['descripcion']}")
            
            if structured_metadata.codigos_p:
                print("\nCODIGOS P:")
                for codigo in structured_metadata.codigos_p[:5]:
                    print(f"  {codigo['codigo']}: {codigo['descripcion']}")
            
            if structured_metadata.componentes_cas:
                print("\nCOMPONENTES:")
                for comp in structured_metadata.componentes_cas[:5]:
                    conc = f" [{comp['concentracion']}]" if comp['concentracion'] != 'No especificada' else ""
                    print(f"  {comp['nombre']} (CAS: {comp['cas']}){conc}")
        
        print(f"\nMetricas: P@5={metrics.precision_at_k[5]:.3f} | MRR={metrics.mrr_at_10:.3f} | {metrics.latency_ms:.0f}ms")
        
        print(f"\nTop 3 chunks:")
        for j, r in enumerate(results[:3], 1):
            # Asegúrate de imprimir el contenido completo si quieres verlo, aunque se truncó a 'Sim' aquí.
            print(f"  {j}. Sim={r.similarity:.3f} | Producto: {r.metadata.get('producto', 'N/A')[:30]} | Tipo: {r.tipo_contenido}")
        
        print(f"\nContexto Actual: {context.current_product}")


if __name__ == "__main__":
    main()