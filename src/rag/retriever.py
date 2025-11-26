"""
retriever.py

RAG Retriever para tu estructura de ChromaDB:
- Parsea metadatos que están en formato string
- Búsqueda por nombre parcial de producto
- Secciones opcionales (muchos chunks en 'otra_informacion')
"""

import chromadb
from difflib import get_close_matches
import ollama
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import ast
import warnings
import logging

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Add of existing embedding ID")
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('chromadb.db.impl.sqlite').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="capture() takes 1 positional argument")

@dataclass
class SearchResult:
    """Resultado de búsqueda."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    similarity: float
    tipo_contenido: str
    source_type: str = "semantic"
@dataclass
class RetrievalMetrics:
    """Métricas simplificadas de recuperación."""
    avg_similarity: float
    top_similarity: float
    latency_ms: float
    total_chunks: int

@dataclass
class StructuredMetadata:
    """Metadata estructurada extraída de chunks."""
    query_type: str
    codigos_h: List[Dict[str, str]] = field(default_factory=list)
    codigos_p: List[Dict[str, str]] = field(default_factory=list)
    componentes_cas: List[Dict[str, str]] = field(default_factory=list)
    extracted_from_chunks: int = 0


@dataclass
class MetadataParser:
    """Parsea metadatos que están en formato string."""
    
    @staticmethod
    def parse_codigos_h(codigos_str: str) -> List[Dict[str, str]]:
        """
        Parsea códigos H desde string.
        CASOS SOPORTADOS:
        - "H315,H319,H317,H225"
        - "{'codigo': 'H226', 'descripcion': '...'}"
        - "[{'codigo': 'H226'}, {'codigo': 'H304'}]"
        """
        if not codigos_str or codigos_str == '':
            return []
        
        codigos = []
        
        # CASO 1: String simple "H315,H319,H317"
        if '{' not in codigos_str and '[' not in codigos_str:
            for codigo in codigos_str.split(','):
                codigo = codigo.strip()
                if codigo and codigo.startswith('H'):
                    codigos.append({
                        'codigo': codigo,
                        'descripcion': MetadataParser._get_h_description(codigo)
                    })
            return codigos
        
        # CASO 2: String con diccionarios "{'codigo': 'H226', 'descripcion': '...'}"
        try:
            # Limpiar y normalizar
            clean_str = codigos_str.replace("'codigo':", '"codigo":').replace("'descripcion':", '"descripcion":')
            clean_str = clean_str.replace("'", '"')  # Comillas simples a dobles
            
            # Si no está en array, convertir
            if not clean_str.startswith('['):
                clean_str = '[' + clean_str + ']'
            
            # Parsear JSON
            import json
            parsed = json.loads(clean_str)
            
            for item in parsed:
                if isinstance(item, dict) and 'codigo' in item:
                    codigo_id = item['codigo']
                    if codigo_id and codigo_id.startswith('H'):
                        codigos.append({
                            'codigo': codigo_id,
                            'descripcion': MetadataParser._get_h_description(codigo_id)
                        })
        except Exception as e:
            print(f"  [WARN] Error parseando códigos H: {e}")
            print(f"  [DEBUG] String recibido: {codigos_str[:200]}")
            
            # FALLBACK: Buscar códigos H con regex
            import re
            matches = re.findall(r'H\d{3}', codigos_str)
            for codigo in set(matches):  # set() para eliminar duplicados
                codigos.append({
                    'codigo': codigo,
                    'descripcion': MetadataParser._get_h_description(codigo)
                })
        
        return codigos
    
    @staticmethod
    def parse_codigos_p(codigos_str: str) -> List[Dict[str, str]]:
        """Parsea códigos P con el mismo enfoque que parse_codigos_h."""
        if not codigos_str or codigos_str == '':
            return []
        
        codigos = []
        
        # CASO 1: String simple "P370 + P378,P305 + P351 + P338"
        if '{' not in codigos_str and '[' not in codigos_str:
            for codigo_group in codigos_str.split(','):
                for codigo in codigo_group.split('+'):
                    codigo = codigo.strip()
                    if codigo and codigo.startswith('P'):
                        codigos.append({
                            'codigo': codigo,
                            'descripcion': MetadataParser._get_p_description(codigo)
                        })
            return codigos
        
        # CASO 2: Con diccionarios (mismo enfoque que códigos H)
        try:
            clean_str = codigos_str.replace("'codigo':", '"codigo":').replace("'descripcion':", '"descripcion":')
            clean_str = clean_str.replace("'", '"')
            
            if not clean_str.startswith('['):
                clean_str = '[' + clean_str + ']'
            
            import json
            parsed = json.loads(clean_str)
            
            for item in parsed:
                if isinstance(item, dict) and 'codigo' in item:
                    codigo_id = item['codigo']
                    if codigo_id and codigo_id.startswith('P'):
                        codigos.append({
                            'codigo': codigo_id,
                            'descripcion': MetadataParser._get_p_description(codigo_id)
                        })
        except Exception as e:
            # FALLBACK: Buscar códigos P con regex
            import re
            matches = re.findall(r'P\d{3}', codigos_str)
            for codigo in set(matches):
                codigos.append({
                    'codigo': codigo,
                    'descripcion': MetadataParser._get_p_description(codigo)
                })
        
        return codigos
    
    @staticmethod
    def parse_componentes_cas(componentes_str: str) -> List[Dict[str, str]]:
        """Parsea componentes CAS."""
        if not componentes_str or componentes_str == '':
            return []
        
        componentes = []
        
        try:
            # Normalizar formato
            clean_str = componentes_str.replace("'nombre':", '"nombre":').replace("'cas':", '"cas":')
            clean_str = clean_str.replace("'concentracion':", '"concentracion":')
            clean_str = clean_str.replace("'", '"')
            
            if not clean_str.startswith('['):
                # Dividir por "},{"
                parts = clean_str.split('},{')
                clean_str = '[' + ','.join('{' + p.strip('{}') + '}' for p in parts) + ']'
            
            import json
            parsed = json.loads(clean_str)
            
            for item in parsed:
                if isinstance(item, dict) and 'nombre' in item and 'cas' in item:
                    componentes.append({
                        'nombre': item['nombre'],
                        'cas': item['cas'],
                        'concentracion': item.get('concentracion', 'No especificada')
                    })
        except Exception as e:
            print(f"  [WARN] Error parseando componentes: {e}")
        
        return componentes
    
    @staticmethod
    def _get_h_description(codigo: str) -> str:
        """Descripción de códigos H."""
        h_descriptions = {
            'H225': 'Líquido y vapores muy inflamables',
            'H226': 'Líquidos y vapores inflamables',
            'H304': 'Puede ser mortal en caso de ingestión',
            'H315': 'Provoca irritación cutánea',
            'H316': 'Provoca una leve irritación cutánea',
            'H317': 'Puede provocar una reacción alérgica en la piel',
            'H319': 'Provoca irritación ocular grave',
            'H335': 'Puede irritar las vías respiratorias',
            'H336': 'Puede provocar somnolencia o vértigo',
            'H373': 'Puede provocar daños en los órganos',
            'H412': 'Nocivo para los organismos acuáticos'
        }
        return h_descriptions.get(codigo, 'Descripción no disponible')
    
    @staticmethod
    def _get_p_description(codigo: str) -> str:
        """Descripción de códigos P."""
        p_descriptions = {
            'P210': 'Mantener alejado de fuentes de ignición',
            'P233': 'Mantener el recipiente cerrado herméticamente',
            'P240': 'Conectar a tierra el recipiente',
            'P241': 'Utilizar material eléctrico/equipos a prueba de explosión',
            'P242': 'Utilizar herramientas que no produzcan chispas',
            'P260': 'No respirar vapores',
            'P261': 'Evitar respirar el polvo/humo/gas',
            'P264': 'Lavarse las manos después de manipular',
            'P270': 'No comer, beber ni fumar durante su utilización',
            'P272': 'La ropa contaminada no debe salir del lugar de trabajo',
            'P273': 'Evitar su liberación al medio ambiente',
            'P280': 'Llevar guantes/ropa/gafas de protección',
            'P303': 'EN CASO DE CONTACTO CON LA PIEL (o el pelo)',
            'P305': 'EN CASO DE CONTACTO CON LOS OJOS',
            'P333': 'En caso de irritación o erupción cutánea',
            'P337': 'Si persiste la irritación ocular',
            'P351': 'Enjuagar con agua',
            'P353': 'Aclararse la piel con agua',
            'P361': 'Quitarse inmediatamente la ropa contaminada',
            'P362': 'Quitarse la ropa contaminada',
            'P364': 'Lavar la ropa contaminada antes de usarla',
            'P370': 'En caso de incendio',
            'P378': 'Utilizar arena seca, polvo químico o espuma resistente',
            'P403': 'Almacenar en un lugar bien ventilado',
            'P405': 'Guardar bajo llave',
            'P501': 'Eliminar el contenido/recipiente según normativa'
        }
        return p_descriptions.get(codigo, 'Descripción no disponible')


from difflib import get_close_matches  # ← AGREGAR AL INICIO DEL ARCHIVO

# ----------------------------------------------------------------------
# MetadataExtractor con Parser integrado
# ----------------------------------------------------------------------
class MetadataExtractor:
    """Extrae y estructura metadata de chunks recuperados con fuzzy matching."""
    
    QUERY_TYPE_MAPPING = {
        # GRUPO 1: Peligros (Sección 2 + 11)
        'peligros': {
            'keywords': [
                'peligro', 'peligros', 'riesgo', 'riesgos',
                'codigo h', 'codigos h', 'h315', 'h319', 'h317', 'h225', 'h226', 'h304', 'h336', 'h412',
                'clasificacion', 'categoria', 'pictograma', 'simbolo',
                'inflamable', 'toxico', 'irritante', 'corrosivo', 'nocivo', 'toxicidad'
            ],
            'secciones': ['identificacion_peligros', 'informacion_toxicologica'],
            'metadata_fields': ['codigos_h'],
            'search_level': 'hybrid'
        },
        
        # GRUPO 2: Protección/EPP (Sección 2 + 8) - NUEVO
        'proteccion': {
            'keywords': [
                'proteccion', 'precaucion', 'precauciones', 'codigo p', 'codigos p',
                'p280', 'p210', 'p233', 'p260', 'p264', 'p270', 'p273', 'p280',
                'p303', 'p305', 'p333', 'p337', 'p370', 'p378', 'p403', 'p405', 'p501',
                'epp', 'equipo', 'guantes', 'gafas', 'mascara', 'respirador',
                'ropa', 'traje', 'botas', 'ventilacion', 'medidas', 'seguridad'
            ],
            'secciones': ['identificacion_peligros', 'controles_exposicion_proteccion'],
            'metadata_fields': ['codigos_p'],
            'search_level': 'hybrid'
        },
        
        # GRUPO 3: Emergencias (4, 5, 6) - NUEVO
        'emergencias': {
            'keywords': [
                'primeros auxilios', 'emergencia', 'emergencias', 'accidente',
                'contacto piel', 'contacto ojos', 'contacto con la piel', 'contacto con los ojos',
                'inhalacion', 'ingestion', 'tragar', 'respirar',
                'incendio', 'fuego', 'llama', 'combustion', 'extincion',
                'derrame', 'vertido', 'fuga', 'escape', 'liberacion',
                'que hacer', 'tratamiento', 'sintomas', 'efectos'
            ],
            'secciones': ['primeros_auxilios', 'medidas_contra_incendios', 'medidas_vertido_accidental'],
            'metadata_fields': [],
            'search_level': 'hybrid'
        },
        
        # GRUPO 4: Componentes (Sección 3)
        'componentes': {
            'keywords': [
                'componente', 'componentes', 'composicion', 'ingrediente', 'ingredientes',
                'cas', 'numero cas', 'sustancia', 'sustancias',
                'concentracion', 'porcentaje',
                'xileno', 'acetato', 'solvente', 'resina', 'pigmento',
                'formula', 'quimico'
            ],
            'secciones': ['composicion_componentes'],
            'metadata_fields': ['componentes_cas'],
            'search_level': 'hybrid'
        },
        
        # GRUPO 5: Manipulación (7 + 13) - NUEVO
        'manipulacion': {
            'keywords': [
                'manipulacion', 'manejo', 'usar', 'uso', 'aplicacion',
                'almacenamiento', 'almacenar', 'guardar', 'conservar',
                'temperatura', 'temperatura almacenamiento', 'conservacion',
                'ventilacion', 'lugar seco', 'lugar fresco',
                'incompatibilidad', 'incompatible', 'reacciones', 'estabilidad',
                'eliminacion', 'disposicion', 'desecho', 'residuo',
                'caducidad', 'vida util', 'vencimiento'
            ],
            'secciones': ['manipulacion_almacenamiento', 'consideraciones_eliminacion', 'estabilidad_reactividad'],
            'metadata_fields': [],
            'search_level': 'hybrid'
        },
        
        # GRUPO 6: Propiedades (Sección 9) - NUEVO
        'propiedades': {
            'keywords': [
                'propiedad', 'propiedades', 'fisico', 'quimico',
                'densidad', 'viscosidad', 'ph', 'color', 'olor', 'aspecto', 'estado',
                'punto inflamacion', 'punto ebullicion', 'punto de inflamacion',
                'solubilidad', 'soluble', 'miscible',
                'liquido', 'solido', 'pastoso', 'viscoso',
                'caracteristicas'
            ],
            'secciones': ['propiedades_fisicas_quimicas'],
            'metadata_fields': [],
            'search_level': 'hybrid'
        },
        
        # GRUPO 7: Identificación (Sección 1) - NUEVO
        'identificacion': {
            'keywords': [
                'identificacion', 'producto', 'nombre', 'codigo', 'codigo producto',
                'fabricante', 'proveedor', 'distribuidor', 'sika',
                'uso', 'aplicacion', 'para que sirve',
                'fecha', 'fecha emision', 'version'
            ],
            'secciones': ['identificacion_producto'],
            'metadata_fields': [],
            'search_level': 'hybrid'
        }
    }
    
    def __init__(self):
        self.parser = MetadataParser()
    
    def detect_query_type(self, query: str) -> Optional[str]:
        """Detecta tipo de query con tolerancia a typos usando fuzzy matching."""
        query_lower = query.lower()
        
        # 1️⃣ PASO 1: Búsqueda exacta (más rápida)
        for query_type, config in self.QUERY_TYPE_MAPPING.items():
            if any(kw in query_lower for kw in config['keywords']):
                return query_type
        
        # 2️⃣ PASO 2: Fuzzy matching solo si no hubo match exacto
        query_words = [w for w in query_lower.split() if len(w) >= 4]  # Ignorar palabras cortas
        
        for query_type, config in self.QUERY_TYPE_MAPPING.items():
            for word in query_words:
                # Buscar coincidencia aproximada (75% similitud)
                matches = get_close_matches(word, config['keywords'], n=1, cutoff=0.75)
                if matches:
                    print(f"  [FUZZY MATCH] '{word}' → '{matches[0]}' (tipo: {query_type})")
                    return query_type
        
        return None
    
    def get_search_level(self, query_type: str) -> str:
        if query_type in self.QUERY_TYPE_MAPPING:
            return self.QUERY_TYPE_MAPPING[query_type]['search_level']
        return 'semantic_only'
    
    def extract(self, results: List[SearchResult], query_type: str) -> StructuredMetadata:
        """Extrae metadata estructurada según el tipo de query."""
        if query_type == 'peligros':
            return StructuredMetadata(
                query_type=query_type,
                codigos_h=self._aggregate_codigos_h(results),
                extracted_from_chunks=len(results)
            )
        elif query_type in ['precauciones', 'proteccion']:  # ← Agregar 'proteccion'
            return StructuredMetadata(
                query_type=query_type,
                codigos_p=self._aggregate_codigos_p(results),
                extracted_from_chunks=len(results)
            )
        elif query_type == 'componentes':
            return StructuredMetadata(
                query_type=query_type,
                componentes_cas=self._aggregate_componentes(results),
                extracted_from_chunks=len(results)
            )
        else:

            return StructuredMetadata(query_type=query_type)
    
    def _aggregate_codigos_h(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        codigos_dict = {}
        for result in results:
            codigos_str = result.metadata.get('codigos_h', '')  
            codigos_list = self.parser.parse_codigos_h(codigos_str)  
            for codigo in codigos_list:
                codigo_id = codigo.get('codigo')
                if codigo_id and codigo_id not in codigos_dict:
                    codigos_dict[codigo_id] = codigo
        return sorted(codigos_dict.values(), key=lambda x: x['codigo'])
    
    def _aggregate_codigos_p(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        codigos_dict = {}
        for result in results:
            codigos_str = result.metadata.get('codigos_p', '')  
            codigos_list = self.parser.parse_codigos_p(codigos_str) 
            for codigo in codigos_list:
                codigo_id = codigo.get('codigo')
                if codigo_id and codigo_id not in codigos_dict:
                    codigos_dict[codigo_id] = codigo
        return sorted(codigos_dict.values(), key=lambda x: x['codigo'])
    
    def _aggregate_componentes(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        componentes_dict = {} 
        for result in results:
            comp_str = result.metadata.get('componentes_cas', '')  
            comp_list = self.parser.parse_componentes_cas(comp_str)  
            for comp in comp_list:
                cas_id = comp.get('cas')  
                if cas_id and cas_id not in componentes_dict:
                    componentes_dict[cas_id] = comp
        return sorted(componentes_dict.values(), key=lambda x: x.get('nombre', ''))
# ----------------------------------------------------------------------
# ConversationalContext
# ----------------------------------------------------------------------
class ConversationalContext:
    """Mantiene contexto conversacional simplificado."""
    
    def __init__(self):
        self.current_product: Optional[str] = None

    
    def set_product(self, producto: str):
        """Establece el producto actual."""
        if producto:
            self.current_product = producto
            print(f"  [CONTEXTO] Producto establecido: {producto}")
    
    def clear_product(self):
        """Limpia el producto actual."""
        self.current_product = None
    
    def enrich_query(self, query: str) -> str:
        if not self.current_product:
            return query
        
        # Si menciona producto, NO enriquecer (permite cambio de producto)
        product_keywords = ['epoxi', 'uretano', 'alquid', 'textur', 'sika']
        if any(kw in query.lower() for kw in product_keywords):
            return query
        
        if len(query.split()) < 4:
            return f"{query} {self.current_product}"
        
        return query
    
    def _is_followup_query(self, query: str) -> bool:
        """Detecta si es query de seguimiento."""
        followup_indicators = ['y', 'sus', 'su', 'ese', 'este', 'dame', 'tambien', 'cuales', 'que', 'es']
        words = query.split()
        
        if not words:
            return False
        
        return words[0] in followup_indicators or len(words) < 5


# ----------------------------------------------------------------------
# HybridRetriever
# ----------------------------------------------------------------------
class HybridRetriever:
    """Sistema de recuperación híbrido sin usar $contains."""
    
    def __init__(self, config, embedding_model: str = "nomic-embed-text"):
        self.config = config
        self.embedding_model = embedding_model
        
        db_path = Path('/Users/sofiavelandiasierra/Documents/RAG/RAG/data/data/vector_db')
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        self.collections = {
            'texto': self.client.get_collection('fds_textos'),
            'tabla': self.client.get_collection('fds_tablas'),
            'imagen': self.client.get_collection('fds_imagens')
        }
        
        self.metadata_extractor = MetadataExtractor()
        
    def _extract_full_product_info(self, results: List[SearchResult], producto: str) -> Dict[str, Any]:
            """Extrae info completa del producto del primer chunk relevante."""
            for result in results:
                if producto.lower() in result.metadata.get('producto', '').lower():
                    return {
                        'producto': result.metadata.get('producto', 'N/A'),
                        'fabricante': result.metadata.get('fabricante', 'N/A'),
                        'codigo_producto': result.metadata.get('codigo_producto', 'N/A'),
                        'fecha_fds': result.metadata.get('fecha_fds', 'N/A'),
                        'codigos_h': result.metadata.get('codigos_h', []),
                        'codigos_p': result.metadata.get('codigos_p', []),
                        'componentes_cas': result.metadata.get('componentes_cas', [])
                    }
            return {}
        
    def retrieve(
        self,
        query: str,
        context_product: Optional[str] = None,
        n_candidates: int = 15, 
        n_final: int = 10  
    ) -> Tuple[List[SearchResult], Optional[str], List[StructuredMetadata], RetrievalMetrics, Dict[str, Any], List[Dict[str, str]]]:
        """Pipeline de recuperación híbrido con multi-type y extracción de imágenes."""
        start_time = time.time()
        
        # 1. Detectar tipo de query (puede ser múltiples en el futuro)
        query_types = []
        detected_type = self.metadata_extractor.detect_query_type(query)
        if detected_type:
            query_types.append(detected_type)
        
        print(f"  [ANÁLISIS] Tipos: {query_types or 'general'}")
        
        # 2. Extraer producto de la query
        detected_product = self._extract_product_from_query(query)
        producto_filtro = detected_product if detected_product else context_product
        
        if producto_filtro:
            print(f"  [FILTRO] Producto: {producto_filtro}")
        
        # 3. Búsqueda semántica ← PRIMERO CREAR all_results
        all_results = self._search_semantic(query, n_candidates)
        
        # 4. Filtrar por producto si es necesario
        if producto_filtro:
            all_results = self._filter_by_product(all_results, producto_filtro)
        
        for result in all_results:
            if result.tipo_contenido == 'imagen':
                result.similarity *= 1.3 
                print(f"  [BOOST IMAGEN] {result.metadata.get('imagen_nombre', 'N/A')}: {result.similarity:.3f}")
        
        # 5. Ordenar y seleccionar top-N
        all_results.sort(key=lambda x: x.similarity, reverse=True)
        final_results = all_results[:n_final]
        
        # 6. Extraer metadata estructurada para CADA tipo
        structured_metadatas = []
        for qtype in query_types:
            metadata = self.metadata_extractor.extract(final_results, qtype)
            structured_metadatas.append(metadata)
        
        # 7. Extraer imágenes de ChromaDB
        images = self._extract_images_from_results(final_results)
        
        # 8. Métricas
        latency = (time.time() - start_time) * 1000
        metrics = self._compute_metrics(final_results, latency)
        product_info = self._extract_full_product_info(final_results, producto_filtro) if producto_filtro else {}
        
        return final_results, producto_filtro, structured_metadatas, metrics, product_info, images
    def _search_semantic(self, query: str, n_results: int) -> List[SearchResult]:
        """Búsqueda semántica sin filtros (recupera todo)."""
        query_embedding = self._embed(query)
        
        results = []
        for coll_name in ['texto', 'tabla', 'imagen']:
            coll_results = self._search_collection(coll_name, query_embedding, n_results)
            results.extend(coll_results)
        
        return results
    
    def _filter_by_product(self, results: List[SearchResult], producto: str) -> List[SearchResult]:
        """Filtra resultados manualmente por producto (búsqueda parcial)."""
        filtered = []
        producto_lower = producto.lower()
        
        for result in results:
            producto_chunk = result.metadata.get('producto', '').lower()
            if producto_lower in producto_chunk:
                filtered.append(result)
        
        return filtered
    
    def _search_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int
    ) -> List[SearchResult]:
        """Busca en una colección sin filtros."""
        
        collection = self.collections[collection_name]
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            print(f"  [ERROR] Búsqueda en {collection_name}: {e}")
            return []
        
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
    
    def _extract_images_from_results(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """Extrae imágenes directamente de los metadatos de chunks tipo 'imagen'."""
        images = []
        
        for result in results:
            # Solo procesar chunks de tipo imagen
            if result.tipo_contenido != 'imagen':
                continue
            
            # Extraer nombre de imagen del metadata (ya viene de ChromaDB)
            imagen_nombre = result.metadata.get('imagen_nombre', '')
            
            if not imagen_nombre:
                print(f"  [WARN] Chunk imagen sin 'imagen_nombre': {result.chunk_id}")
                continue
            
            images.append({
                'filename': imagen_nombre,
                'similarity': round(result.similarity, 3),
                'producto': result.metadata.get('producto', 'N/A'),
                'tipo': 'imagen',
                'chunk_id': result.chunk_id
            })
            
            print(f"  [✓ IMAGEN] {imagen_nombre} (sim={result.similarity:.3f})")
        
        print(f"  [RESUMEN] {len(images)} imágenes extraídas")
        return images
    
    def _extract_product_from_query(self, query: str) -> Optional[str]:
        """Extrae nombre de producto de la query."""
        query_lower = query.lower()

        
        # Nombres parciales para búsqueda
        product_terms = {
            # Epóxico Aluminio (NUEVO - más específico primero)
            'esmalte epoxico aluminio': 'Epóxico Aluminio',
            'esmalte epóxico aluminio': 'Epóxico Aluminio',
            'epoxico aluminio': 'Epóxico Aluminio',
            'epóxico aluminio': 'Epóxico Aluminio',
            
            # Pintura Texturizada (NUEVO)
            'pintura texturizada': 'Texturizada',
            'texturizada': 'Texturizada',
            
            # Esmalte Uretano (NUEVO)
            'esmalte uretano': 'Uretano',
            
            # Esmalte Alquídico (NUEVO)
            'esmalte alquidico': 'Alquídico',
            'esmalte alquídico': 'Alquídico',
            
            # Epóxico genérico
            'epoxico': 'Epóxico',
            'epóxico': 'Epóxico',
            'epoxi': 'Epóxico',
            
            # Uretano genérico
            'uretano': 'Uretano',
            
            # Alquídico genérico
            'alquidico': 'Alquídico',
            'alquídico': 'Alquídico',
            
            # SikaWall
            'sikawall': 'SikaWall',
        }
        
        # Buscar coincidencia exacta (más largo primero)
        for term in sorted(product_terms.keys(), key=len, reverse=True):
            if term in query_lower:
                return product_terms[term]
        
        # Fuzzy matching como fallback
        query_words = [w for w in query_lower.split() if len(w) >= 5]
        for word in query_words:
            matches = get_close_matches(word, product_terms.keys(), n=1, cutoff=0.8)
            if matches:
                print(f"  [FUZZY PRODUCTO] '{word}' → '{matches[0]}'")
                return product_terms[matches[0]]
        
        return None
        
    def _embed(self, text: str) -> List[float]:
        """Genera embedding."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response['embedding']
    
    def _compute_metrics(self, results: List[SearchResult], latency: float) -> RetrievalMetrics:
        """Calcula métricas simplificadas."""
        if not results:
            return RetrievalMetrics(avg_similarity=0.0, top_similarity=0.0, latency_ms=latency, total_chunks=0)
        
        avg_sim = sum(r.similarity for r in results) / len(results)
        top_sim = results[0].similarity if results else 0.0
        
        return RetrievalMetrics(
            avg_similarity=avg_sim,
            top_similarity=top_sim,
            latency_ms=latency,
            total_chunks=len(results)
        )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    from config import ProjectConfig
    
    @dataclass
    class MockProjectConfig:
        def get_folder(self, name: str) -> Path:
            return Path('./data/vector_db')
    
    try:
        config = ProjectConfig()
    except:
        print("⚠️ Usando Mock Config")
        config = MockProjectConfig()
    
    retriever = HybridRetriever(config)
    context = ConversationalContext()
    
    print("\n" + "=" * 80)
    print("SISTEMA RAG HÍBRIDO - VERSIÓN CORREGIDA")
    print("=" * 80)
    
    conversation = [
        "Cuales son los componentes peligrosos del esmalte epoxico?",
        "Y sus peligros?",
        "Dame sus precauciones",
        "Que componentes tiene?",
        "Y es inflamable?",
        "Que peligros tiene el esmalte uretano?",
        "Que componentes tiene el esmalte alquidico?",
        "Que precauciones tiene la pintura texturizada?"
    ]
    
    for i, query_original in enumerate(conversation, 1):
        print(f"\n{'='*80}")
        print(f"TURNO {i}: {query_original}")
        print(f"{'='*80}")
        
        query_enriched = context.enrich_query(query_original)
        if query_enriched != query_original:
            print(f"  [REFORMULADA] {query_enriched}")
        
        results, detected_product, structured_metadatas, metrics, product_info, images = retriever.retrieve(
            query_enriched,
            context_product=context.current_product,
            n_candidates=15,
            n_final=5
        )
        
        if detected_product:
            context.set_product(detected_product)
        
        
        # Métricas
        print(f"\n📊 MÉTRICAS:")
        print(f"  ├─ Similitud Promedio: {metrics.avg_similarity:.3f}")
        print(f"  ├─ Mejor Similitud: {metrics.top_similarity:.3f}")
        print(f"  ├─ Chunks Recuperados: {metrics.total_chunks}")
        print(f"  └─ Latencia: {metrics.latency_ms:.0f}ms")
        
        print(f"\n📄 CHUNKS RECUPERADOS ({len(results)}):")
        for j, r in enumerate(results, 1):
            producto_chunk = r.metadata.get('producto', 'N/A')
            seccion_chunk = r.metadata.get('seccion', 'N/A')
            print(f"  {j}. [{r.tipo_contenido.upper()}] Sim={r.similarity:.3f}")
            print(f"     ├─ Producto: {producto_chunk[:60]}")
            print(f"     ├─ Sección: {seccion_chunk}")
            print(f"     └─ Chunk ID: {r.chunk_id}")
                
        if images:
            print(f"\n🖼️ IMÁGENES ENCONTRADAS ({len(images)}):")
            for img in images:  # Mostrar TODAS
                print(f"  • {img['filename']}")
                print(f"    └─ Similitud: {img['similarity']:.3f} | Tipo: {img['tipo']} | Producto: {img['producto']}")
        else:
            print(f"\n🖼️ IMÁGENES: No se encontraron imágenes en los resultados")
        # Metadata Estructurada
        if structured_metadatas:
            for metadata in structured_metadatas:
                if metadata.extracted_from_chunks > 0:
                    print(f"\n📦 METADATA ESTRUCTURADA - {metadata.query_type.upper()} ({metadata.extracted_from_chunks} chunks):")
                    
                    if metadata.codigos_h:
                        print(f"\n  🔴 CÓDIGOS H ({len(metadata.codigos_h)}):")
                        for codigo in metadata.codigos_h:
                            print(f"    • {codigo['codigo']}: {codigo['descripcion']}")
                    
                    if metadata.codigos_p:
                        print(f"\n  🔵 CÓDIGOS P ({len(metadata.codigos_p)}):")
                        for codigo in metadata.codigos_p:
                            print(f"    • {codigo['codigo']}: {codigo['descripcion']}")
                    
                    if metadata.componentes_cas:
                        print(f"\n  🧪 COMPONENTES ({len(metadata.componentes_cas)}):")
                        for comp in metadata.componentes_cas:
                            conc = f" [{comp['concentracion']}]" if comp['concentracion'] != 'No especificada' else ""
                            print(f"    • {comp['nombre']} (CAS: {comp['cas']}){conc}")
        # Top Chunks
        print(f"\n🎯 TOP-3 CHUNKS:")
        for j, r in enumerate(results[:3], 1):
            producto_chunk = r.metadata.get('producto', 'N/A')
            seccion_chunk = r.metadata.get('seccion', 'N/A')
            print(f"  {j}. Sim={r.similarity:.3f} | {r.tipo_contenido}")
            print(f"     └─ {producto_chunk[:50]}")
            print(f"     └─ Sección: {seccion_chunk}")
        
        print(f"\n💡 CONTEXTO ACTUAL: {context.current_product or '(ninguno)'}")
    
    print("\n" + "=" * 80)
    print("FIN DE LA CONVERSACIÓN")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
