"""
rag_pipeline.py - Retriever + Generator unificados con contexto conversacional.
"""

import re
import time
import warnings
import logging
import ollama
import chromadb
import unicodedata

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")
logging.getLogger('chromadb').setLevel(logging.ERROR)

# ─────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────
@dataclass
class SearchResult:
    chunk_id:       str
    content:        str
    metadata:       Dict[str, Any]
    similarity:     float
    tipo_contenido: str

@dataclass
class RetrievalMetrics:
    avg_similarity: float
    top_similarity: float
    latency_ms:     float
    total_chunks:   int

@dataclass
class GeneratedResponse:
    answer:              str
    sources:             List[Dict[str, Any]]
    structured_metadata: Dict[str, Any]
    pictogramas:         List[str]
    latency_ms:          float
    metrics:             RetrievalMetrics


# ─────────────────────────────────────────────
# CONTEXTO CONVERSACIONAL
# ─────────────────────────────────────────────
class ConversationalContext:
    """Mantiene el producto activo entre turnos de conversación."""

    # Términos conocidos para detectar producto en la query
    PRODUCT_TERMS = [
        'epoxico', 'epóxico', 'epoxi', 'epoxy',
        'uretano', 'poliuretano',
        'alquidico', 'alquídico',
        'texturiz', 'textura',
        'anticorrosiv',
        'sika', 'pintuco', 'axalta', 'sherwin', 'comex',
        'esmalte', 'pintura', 'barniz', 'laca', 'imprimante',
        'señalizacion', 'trafico',
    ]

    def __init__(self):
        self.current_product: Optional[str] = None

    def detect_product(self, query: str) -> Optional[str]:
        """Extrae término de producto de la query si existe."""
        q = query.lower()
        for term in self.PRODUCT_TERMS:
            if term in q:
                return term
        return None

    def update(self, query: str) -> Optional[str]:
        """Actualiza contexto y retorna el producto a usar para filtrar."""
        detected = self.detect_product(query)
        if detected:
            self.current_product = detected
            print(f"  [CONTEXTO] Producto detectado: {detected}")
        elif self.current_product:
            print(f"  [CONTEXTO] Usando producto previo: {self.current_product}")
        return self.current_product

    def reset(self):
        self.current_product = None


# ─────────────────────────────────────────────
# PARSER DE METADATA
# ─────────────────────────────────────────────
class MetadataParser:

    @staticmethod
    def parse_pipe_codes(codes_str: str, prefix: str) -> List[Dict[str, str]]:
        if not codes_str or codes_str.strip() in ('', '[]', '{}'):
            return []
        result, seen = [], set()
        for part in codes_str.split('|'):
            part = part.strip()
            m = re.match(
                rf'^({prefix}\d{{3}}(?:\+{prefix}\d{{3}})*)\s*[:\-]\s*(.+)$',
                part, re.IGNORECASE
            )
            if m:
                codigo = m.group(1).upper().replace(' ', '')
                if codigo not in seen:
                    seen.add(codigo)
                    result.append({'codigo': codigo, 'descripcion': m.group(2).strip()})
        return result

    @staticmethod
    def parse_pictogramas(pic_str: str) -> List[str]:
        if not pic_str or pic_str.strip() in ('', '[]'):
            return []
        return [p.strip() for p in pic_str.split(',') if p.strip()]

    @staticmethod
    def parse_h_codes(s: str) -> List[Dict[str, str]]:
        return MetadataParser.parse_pipe_codes(s, 'H')

    @staticmethod
    def parse_p_codes(s: str) -> List[Dict[str, str]]:
        return MetadataParser.parse_pipe_codes(s, 'P')


# ─────────────────────────────────────────────
# QUERY TYPE DETECTOR
# ─────────────────────────────────────────────
QUERY_TYPES = {
    'peligros':      ['peligro', 'riesgo', 'clasificacion', 'inflamable',
                      'toxico', 'irritante', 'corrosivo', 'pictograma'],
    'proteccion':    ['proteccion', 'precaucion', 'epp', 'guantes', 'mascara',
                      'respirador', 'medidas', 'seguridad'],
    'emergencias':   ['primeros auxilios', 'emergencia', 'derrame', 'incendio',
                      'contacto piel', 'contacto ojos', 'inhalacion', 'ingestion'],
    'componentes':   ['componente', 'composicion', 'cas', 'sustancia',
                      'concentracion', 'ingrediente'],
    'manipulacion':  ['manipulacion', 'almacenamiento', 'almacenar', 'guardar',
                      'temperatura', 'incompatible', 'estabilidad'],
    'propiedades':   ['densidad', 'viscosidad', 'ph', 'color', 'olor',
                      'punto inflamacion', 'solubilidad', 'propiedades', 'fisicas'],
    'identificacion':['fabricante', 'proveedor', 'nombre', 'codigo',
                      'para que sirve', 'fecha'],
}

def detect_query_type(query: str) -> str:
    q = query.lower()
    for qtype, keywords in QUERY_TYPES.items():
        if any(kw in q for kw in keywords):
            return qtype
    return 'general'


# ─────────────────────────────────────────────
# METADATA DESDE CHUNKS
# ─────────────────────────────────────────────
def agregar_metadata_desde_chunks(results: List[SearchResult]) -> Dict[str, Any]:
    # Ya no dependemos estrictamente de parser externo si los datos vienen como strings
    h_dict, p_dict, pic_set = {}, {}, set()
    producto, fabricante, cas_principal = '', '', ''

    for r in results:
        m = r.metadata
        
        # 1. Captura de Identidad (Strings directos)
        if not producto: 
            producto = m.get('producto') or m.get('nombre_comercial') or m.get('archivo', '')
        if not fabricante: 
            fabricante = m.get('fabricante') or m.get('proveedor', '')
        if not cas_principal: 
            cas_principal = m.get('cas_principal') or m.get('cas', '')

        # 2. Procesar H_CODES (Vienen como: "H225: desc | H315: desc")
        h_raw = m.get('h_codes', '')
        if h_raw and isinstance(h_raw, str):
            # Separamos por el pipe '|' que usaste en la ingesta
            partes_h = [h.strip() for h in h_raw.split('|') if h.strip()]
            for item in partes_h:
                if ':' in item:
                    codigo, desc = item.split(':', 1)
                    codigo = codigo.strip()
                    if codigo not in h_dict:
                        h_dict[codigo] = {'codigo': codigo, 'descripcion': desc.strip()}

        # 3. Procesar P_CODES (Vienen igual que los H)
        p_raw = m.get('p_codes', '')
        if p_raw and isinstance(p_raw, str):
            partes_p = [p.strip() for p in p_raw.split('|') if p.strip()]
            for item in partes_p:
                if ':' in item:
                    codigo, desc = item.split(':', 1)
                    codigo = codigo.strip()
                    if codigo not in p_dict:
                        p_dict[codigo] = {'codigo': codigo, 'descripcion': desc.strip()}

        # 4. Procesar PICTOGRAMAS (Vienen como: "Inflamable, Toxicidad")
        pic_raw = m.get('pictogramas', '')
        if pic_raw and isinstance(pic_raw, str):
            # Separamos por coma
            nombres_pic = [p.strip() for p in pic_raw.split(',') if p.strip()]
            for p in nombres_pic:
                pic_set.add(p)

    return {
        'producto':      producto,
        'fabricante':    fabricante,
        'cas_principal': cas_principal,
        'h_codes':       sorted(h_dict.values(), key=lambda x: x['codigo']),
        'p_codes':       sorted(p_dict.values(), key=lambda x: x['codigo']),
        'pictogramas':   sorted(list(pic_set)),

    }
# ─────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────
class RAGPipeline:

    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL       = "qwen2.5:1.5b"

    def __init__(self, config):
        self.config      = config
        db_path          = str(config.get_folder('vector_db'))
        self.images_path = config.get_folder('images')
        self.context     = ConversationalContext()

        self.client    = chromadb.PersistentClient(path=db_path)
        self.col_texto = self.client.get_collection('fds_textos')
        self.col_tabla = self.client.get_collection('fds_tablas')

        ollama.list()
        print(f"RAGPipeline listo — LLM: {self.LLM_MODEL} | Embeddings: {self.EMBEDDING_MODEL}")

    def _embed(self, text: str) -> List[float]:
        return ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=text)['embedding']

    def _retrieve(self, query: str,
                  n_candidates: int = 80,
                  n_final: int = 70) -> List[SearchResult]:
        emb     = self._embed(query)
        results = []

        for col, tipo in [(self.col_texto, 'texto'), (self.col_tabla, 'tabla')]:
            try:
                r = col.query(
                    query_embeddings=[emb],
                    n_results=n_candidates,
                    include=['documents', 'metadatas', 'distances']
                )
                for i in range(len(r['ids'][0])):
                    results.append(SearchResult(
                        chunk_id       = r['ids'][0][i],
                        content        = r['documents'][0][i],
                        metadata       = r['metadatas'][0][i],
                        similarity = round(max(0, 1 - (r['distances'][0][i] / 100)), 3),
                        tipo_contenido = tipo
                    ))
            except Exception as e:
                print(f"  [WARN] Colección {tipo}: {e}")

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:n_final]

    def _filter_by_product(self, results, producto):
        def normalize(s):
            return unicodedata.normalize('NFD', s.lower()).encode('ascii', 'ignore').decode()
        
        prod_norm = normalize(producto)
        filtered = [
            r for r in results
            if prod_norm in normalize(r.metadata.get('producto', ''))
        ]
        if filtered:
            print(f"  [FILTRO] {len(filtered)} chunks para '{producto}'")
            return filtered
        print(f"  [FILTRO] Sin coincidencias — usando general")
        return results

    def _get_images(self, metadata: Dict[str, Any]) -> List[str]:
        if not self.images_path.exists():
            return []
        producto    = metadata.get('producto', '')
        nombre_base = re.sub(r'[^\w\s\-]', '', producto)[:40].strip()
        imgs        = list(self.images_path.glob(f"*{nombre_base}*"))
        return [p.name for p in imgs[:5]]

    def _build_prompt(self, query: str,
                      results: List[SearchResult],
                      metadata: Dict[str, Any],
                      query_type: str) -> str:

        h_text   = '\n'.join(f"- {c['codigo']}: {c['descripcion']}"
                             for c in metadata['h_codes']) or 'No disponible'
        p_text   = '\n'.join(f"- {c['codigo']}: {c['descripcion']}"
                             for c in metadata['p_codes']) or 'No disponible'
        pic_text = ', '.join(metadata['pictogramas']) or 'No disponible'
        contexto = '\n\n'.join(
            f"[Sección: {r.metadata.get('seccion_fds', 'N/A')}]\n{r.content[:800]}"
            for r in results[:6] if r.tipo_contenido == 'texto'
        )

        prompt = (
            "Eres un experto en seguridad química especializado en FDS.\n\n"
            f"PRODUCTO: {metadata['producto']}\n"
            f"FABRICANTE: {metadata['fabricante']}\n"
            f"CAS PRINCIPAL: {metadata['cas_principal']}\n"
            f"PICTOGRAMAS: {pic_text}\n\n"
            f"CÓDIGOS DE PELIGRO (H):\n{h_text}\n\n"
            f"MEDIDAS DE PRECAUCIÓN (P):\n{p_text}\n\n"
            f"CONTEXTO FDS (tipo: {query_type}):\n{contexto}\n\n"
            f"PREGUNTA: {query}\n\n"
            "INSTRUCCIONES:\n"
            "1. Responde ÚNICAMENTE con información presente en el contexto.\n"
            "2. Si un dato no está, di: La FDS no especifica este dato.\n"
            "3. Integra los códigos H/P caundo lo veas necesario de forma natural explicados.\n"
            "4. Cita la sección de donde proviene cada dato.\n"
            "5. Responde en texto plano sin markdown ni asteriscos.\n"
            "6. Respuesta concisa, completa y precisa.\n\n"
            "RESPUESTA:"
        )
        return prompt

    def _call_llm(self, prompt: str) -> str:
        resp = ollama.generate(
            model=self.LLM_MODEL,
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': 1024, 'num_ctx': 4096}
        )
        text = resp.get('response', '')
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*',     r'\1', text)
        text = re.sub(r'#+\s*',          '',    text)
        return text.strip()

    def run(self,
            query:        str,
            session_ctx:  Optional[ConversationalContext] = None,
            n_candidates: int = 80,
            n_final:      int = 70) -> GeneratedResponse:

        t0         = time.time()
        query_type = detect_query_type(query)
        print(f"  [TIPO] {query_type}")

        # Usar contexto de sesión si se pasa, si no usar el interno
        ctx     = session_ctx if session_ctx is not None else self.context
        producto = ctx.update(query)

        results = self._retrieve(query, n_candidates, n_final)

        if producto:
            results = self._filter_by_product(results, producto)

        metadata = agregar_metadata_desde_chunks(results)
        imagenes = self._get_images(metadata)
        prompt   = self._build_prompt(query, results, metadata, query_type)
        answer   = self._call_llm(prompt)
        latency  = (time.time() - t0) * 1000

        metrics = RetrievalMetrics(
            avg_similarity = sum(r.similarity for r in results) / max(len(results), 1),
            top_similarity = results[0].similarity if results else 0.0,
            latency_ms     = latency,
            total_chunks   = len(results)
        )

        return GeneratedResponse(
            answer   = answer,
            sources  = [{
                'producto':   r.metadata.get('producto', ''),
                'seccion':    r.metadata.get('seccion_fds', ''),
                'similarity': round(r.similarity, 3),
                'tipo':       r.tipo_contenido,
            } for r in results],
            structured_metadata = metadata,
            pictogramas         = imagenes,
            latency_ms          = latency,
            metrics             = metrics
        )

    def reset_context(self):
        self.context.reset()


# ─────────────────────────────────────────────
# GEMINI SWAP (descomentar para usar)
# ─────────────────────────────────────────────
# import google.generativeai as genai
# genai.configure(api_key="TU_API_KEY")
# class RAGPipelineGemini(RAGPipeline):
#     def _embed(self, text):
#         return genai.embed_content(model="models/text-embedding-004",content=text)['embedding']
#     def _call_llm(self, prompt):
#         return genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt).text


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import ProjectConfig

    config   = ProjectConfig()
    pipeline = RAGPipeline(config)

    # Test conversacional: segunda pregunta sin mencionar producto
    conversacion = [
        "Cuáles son los peligros del esmalte epóxico?",
        "Y qué medidas de protección se necesitan?",   # sin mencionar producto
        "Qué hacer en caso de derrame?",                # sin mencionar producto
        "Ahora dime los peligros del esmalte alquídico?",  # cambio de producto
        "Cuáles son sus componentes?",                  # sin mencionar producto
    ]

    for q in conversacion:
        print(f"\n{'='*70}\nQUERY: {q}\n{'='*70}")
        resp = pipeline.run(q)
        print(f"RESPUESTA:\n{resp.answer[:300]}")
        print(f"Producto activo: {pipeline.context.current_product}")
        print(f"Latencia: {resp.latency_ms:.0f}ms")