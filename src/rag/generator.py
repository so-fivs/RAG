"""
src/rag/generator.py 
"""

import ollama
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from config import ProjectConfig
import sys
import warnings
warnings.filterwarnings("ignore", message="Add of existing embedding ID")
try:
    from .retriever import (
        HybridRetriever, 
        ConversationalContext, 
        SearchResult, 
        StructuredMetadata
    )
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.rag.retriever import (
        HybridRetriever,
        ConversationalContext,
        SearchResult,
        StructuredMetadata
    )
config = ProjectConfig()
db_path = config.get_folder('vector_db')
images_path = config.get_folder('images')
pdf_path = config.get_folder('raw_documents')

@dataclass
class GeneratedResponse:
    """Respuesta generada por el sistema."""
    answer: str
    sources: List[Dict[str, Any]]
    fds_reference: Optional[Dict[str, str]]  # ← NUEVO CAMPO
    structured_metadata: Optional[Dict[str, Any]]
    pictogramas: List[str]
    latency_ms: float
    retrieval_metrics: Dict[str, Any]

class RAGGenerator:
    """Generador RAG optimizado con modelo liviano."""
    
    def __init__(self, config, use_gemini: bool = False, temperature: float = 0.3):  
        self.config = config
        self.temperature = temperature
        self.use_gemini = False
        
  
        self.llm_model = "qwen2.5:1.5b" 
        #tinydolphin
        self.pictograma_mapping = {
            'H225': 'flame',
            'H226': 'flame',
            'H304': 'health_hazard',
            'H315': 'exclamation',
            'H317': 'exclamation',
            'H319': 'exclamation',
            'H335': 'exclamation',
            'H336': 'health_hazard',
            'H373': 'health_hazard',
            'H412': 'environment'
        }
        
        try:
            self.retriever = HybridRetriever(config)
            self.context = ConversationalContext()
            print("✅ Retriever y contexto inicializados")
        except Exception as e:
            print(f"❌ Error inicializando retriever: {e}")
            raise
        
        print(f"🚀 Usando modelo OPTIMIZADO: {self.llm_model}")
        try:
            ollama.list()
            print("✅ Ollama verificado")
        except Exception as e:
            print(f"❌ ERROR: Ollama no disponible. Ejecuta: ollama serve")
            raise

    def generate_response(self, query: str, use_context: bool = True) -> GeneratedResponse:
        start_time = time.time()
        
        try:
            results, detected_product, structured_metadatas, metrics, product_info, images = self.retriever.retrieve(
            self.context.enrich_query(query) if use_context else query,
            context_product=self.context.current_product if use_context else None,
            n_candidates=30,
            n_final=15    
            )
            
            if detected_product and use_context:
                self.context.set_product(detected_product)
            
            metadata_dict = self._format_metadata_deduplicated(structured_metadatas) 
            prompt = self._build_prompt(query, results, metadata_dict, product_info)
            text_chunks = [r for r in results if r.tipo_contenido == 'texto']
            if not text_chunks:
                return GeneratedResponse(
                    answer="No encontré información de texto en la FDS para responder esta pregunta con precisión.",
                    sources=self._format_sources(results),
                    fds_reference=None,
                    structured_metadata=metadata_dict,
                    pictogramas=[],
                    latency_ms=0,
                    retrieval_metrics={}
                )
            llm_response = self._call_llm(prompt)
            
            # Extraer referencia FDS explícita
            fds_reference = None
            if detected_product:
                fds_reference = {
                    'producto': detected_product,
                    'fabricante': product_info.get('fabricante', 'N/A'),
                    'codigo': detected_product,  
                    'fecha': product_info.get('fecha_fds', 'N/A')
                }
                print(f"  [FDS REFERENCE] {fds_reference}")
            
            pictogramas_unicos = self._extract_pictogramas(structured_metadatas)
            imagenes_unicas = list({img['filename'] for img in images})
            combined_images = list(set(pictogramas_unicos + imagenes_unicas))
            print(f"  [IMÁGENES FINALES] {len(combined_images)} únicas: {combined_images}")
            
            sources = self._format_sources(results)
            latency = (time.time() - start_time) * 1000

            return GeneratedResponse(
                answer=llm_response,
                sources=sources,
                fds_reference=fds_reference,
                structured_metadata=metadata_dict,
                pictogramas=combined_images,
                latency_ms=latency,
                retrieval_metrics={
                    'avg_similarity': metrics.avg_similarity,
                    'top_similarity': metrics.top_similarity,
                    'total_chunks': metrics.total_chunks
                }
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def _build_prompt(self, query: str, results: List[SearchResult], 
                structured_metadata_dict: Optional[Dict[str, Any]], 
                product_info: Dict[str, Any]) -> str:
        """Prompt optimizado para respuestas completas y detalladas."""
        
        producto_nombre = product_info.get('producto', 'N/A')
        fabricante = product_info.get('fabricante', 'N/A')
        codigo = product_info.get('codigo_producto', 'N/A')
        fecha_fds = product_info.get('fecha_fds', 'N/A')
        prompt = f"""Eres un experto en seguridad química especializado en Fichas de Datos de Seguridad (FDS).

    {'='*60}
    INFORMACIÓN DEL PRODUCTO
    {'='*60}
Producto: {producto_nombre}
Fabricante: {fabricante}
Código del Producto: {codigo}
Fecha de Emisión FDS: {fecha_fds}

    PREGUNTA: {query}

    """
        
        if structured_metadata_dict:
            
            # Agregar códigos H/P y componentes
            if structured_metadata_dict.get('codigos_h'):
                prompt += "\nCÓDIGOS DE PELIGRO (H):\n"
                for codigo in structured_metadata_dict['codigos_h']:
                    prompt += f"- {codigo['codigo']}: {codigo['descripcion']}\n"
            
            if structured_metadata_dict.get('codigos_p'):
                prompt += "\nMEDIDAS DE PRECAUCIÓN (P):\n"
                for codigo in structured_metadata_dict['codigos_p']:
                    prompt += f"- {codigo['codigo']}: {codigo['descripcion']}\n"
            
            if structured_metadata_dict.get('componentes_cas'):
                prompt += "\nCOMPONENTES QUÍMICOS:\n"
                for comp in structured_metadata_dict['componentes_cas']:
                    # Asegurarse de que 'concentracion' esté disponible
                    conc = comp.get('concentracion', 'No especificada')
                    conc_str = f" ({conc})" if conc != 'No especificada' else ""
                    prompt += f"- {comp['nombre']} - CAS: {comp['cas']}{conc_str}\n"

        prompt += "\nCONTEXTO DE LA FDS:\n"
        prompt += "\nCONTEXTO DE LA FDS:\n"
        text_results = [r for r in results if r.tipo_contenido == 'texto']
        for i, result in enumerate(text_results[:5], 1):
            prompt += f"\n[Fragmento {i} - Sección: {result.metadata.get('seccion', 'N/A')}]\n{result.content[:800]}\n"
        
        prompt += """

    INSTRUCCIONES:
    0. Inicia mencionando: "El {producto_nombre} fabricado por {fabricante} (Código: {codigo}, FDS vigente desde {fecha_fds})..." y esta información debe aparecer naturalmente en el primer párrafo
    1. Proporciona una respuesta COMPLETA CONCISA
    2. Estructura tu respuesta con: introducción y uso del producto → detalles técnicos correspondientes a la pregunta → recomendaciones prácticas (sin títulos explícitos)
    3. Integra los códigos H/P y componentes CAS de manera natural en el texto, sin mencionar explícitamente los códigos sino parafraseados 
    4. Si la información es insuficiente, especifica QUÉ datos faltan exactamente
    5. Usa lenguaje técnico pero accesible, sin jerga innecesaria
    6. Para información sobre precauciones, revisa también la sección de controles de exposición/protección individual
    7. No incluyas información de otros productos o documentos si no la encuentras directamente
    8. Responde ÚNICAMENTE con información presente en los fragmentos anteriores
    9. Si un dato no está en los fragmentos, di exactamente: "La FDS no especifica este dato"
    10. NO completes con conocimiento general ni suposiciones
    11. NO menciones materiales, temperaturas ni condiciones que no estén literalmente en el texto
    12. Cita la sección de donde proviene cada dato (ej: "Según la sección de manipulación...")
    13. Si los fragmentos no contienen información relevante para la pregunta, responde: 
    "No encontré información suficiente en la FDS sobre este tema"`
    14. IMPORTANTE: Responde en texto plano sin formato Markdown, sin asteriscos, negritas ni listas numeradas

    RESPUESTA DETALLADA:
    """
        return prompt
        
    def _call_llm(self, prompt: str) -> str:
        """Llamada al LLM."""
        try:
            print(f"  [QWEN] Generando respuesta...")
            
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': 500,
                    'top_p': 0.9,
                    'num_ctx': 4093
                }
            )
            
            if not response or 'response' not in response:
                return "Error: Ollama no devolvió respuesta válida"
            
            # Limpiar markdown agresivamente
            import re
            text = response['response']
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
            text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
            text = re.sub(r'#+\s*', '', text)             # ### headers
            text = re.sub(r'^-\s+', '', text, flags=re.MULTILINE)  # - listas
            text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)  # 1. listas
            
            if len(text.strip()) < 50:
                return "Error: La base de datos no tiene información suficiente."
            
            print(f"  [QWEN] ✅ Respuesta limpia: {len(text)} chars")
            return text
        except Exception as e:
            print(f"  [QWEN ERROR] {e}")
            return f"Error: {str(e)}"

    def _extract_pictogramas(self, structured_metadatas: List[StructuredMetadata]) -> List[str]:
        """Extrae pictogramas únicos."""
        if not structured_metadatas: 
            return []
        
        pictogramas = set()
        
        try:            
            if not images_path.exists():
                print(f"⚠️ Carpeta de imágenes no encontrada: {images_path}")
                return []
            
            for metadata in structured_metadatas:
                if not metadata.codigos_h:  
                    continue
                
                for codigo in metadata.codigos_h:
                    codigo_id = codigo.get('codigo', '')
                    
                    if codigo_id in self.pictograma_mapping:
                        base_name = self.pictograma_mapping[codigo_id]
                        
                        matching_files = (
                            list(images_path.glob(f"*{base_name}*.png")) + 
                            list(images_path.glob(f"*{base_name}*.jpg"))
                        )
                        
                        if matching_files:
                            pictogramas.add(matching_files[0].name)
            
        except Exception as e:
            print(f"❌ Error extrayendo pictogramas: {e}")
        
        return list(pictogramas)
    
    def _format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Formatea fuentes para respuesta."""
        sources = []
        for result in results:
            sources.append({
                'producto': result.metadata.get('producto', 'N/A'),
                'seccion': result.metadata.get('seccion', 'N/A'),
                'similarity': round(result.similarity, 3),
                'tipo': result.tipo_contenido,
                'chunk_id': result.chunk_id,
                'content_preview': result.content[:200] + '...'
            })
        return sources
    
    # ✅ CAMBIO 3: Nueva función de deduplicación
    def _format_metadata_deduplicated(self, structured_metadatas: List[StructuredMetadata]) -> Optional[Dict[str, Any]]:
        """
        Formatea metadata eliminando duplicados por código único.
        
        ANTES: ['H225', 'H315', 'H225', 'H315'] (duplicados)
        AHORA: ['H225', 'H315'] (únicos)
        """
        if not structured_metadatas:
            return None
        
        # Diccionarios para deduplicar (clave = código)
        codigos_h_dict = {}
        codigos_p_dict = {}
        componentes_cas_dict = {}
        query_types = []
        total_chunks = 0
        
        for metadata in structured_metadatas:
            query_types.append(metadata.query_type)
            total_chunks += metadata.extracted_from_chunks
            
            # Agregar códigos H únicos
            for codigo in metadata.codigos_h:
                codigo_id = codigo.get('codigo')
                if codigo_id and codigo_id not in codigos_h_dict:
                    codigos_h_dict[codigo_id] = codigo
            
            # Agregar códigos P únicos
            for codigo in metadata.codigos_p:
                codigo_id = codigo.get('codigo')
                if codigo_id and codigo_id not in codigos_p_dict:
                    codigos_p_dict[codigo_id] = codigo
            
            # Agregar componentes CAS únicos
            for comp in metadata.componentes_cas:
                cas_id = comp.get('cas')
                if cas_id and cas_id not in componentes_cas_dict:
                    componentes_cas_dict[cas_id] = comp
        
        # Convertir a listas ordenadas
        return {
            'query_types': list(set(query_types)),
            'codigos_h': sorted(codigos_h_dict.values(), key=lambda x: x['codigo']),
            'codigos_p': sorted(codigos_p_dict.values(), key=lambda x: x['codigo']),
            'componentes_cas': sorted(componentes_cas_dict.values(), key=lambda x: x.get('nombre', '')),
            'extracted_from_chunks': total_chunks
        }
    
    def reset_context(self):
        """Reinicia contexto conversacional."""
        self.context.clear_product()
        print("🔄 Contexto reiniciado")


if __name__ == "__main__":
    try:
        from RAG.config import ProjectConfig
        
        config = ProjectConfig()
        generator = RAGGenerator(config, use_gemini=False)
        
        # Test rápido
        query = "Cuales son los peligros del esmalte epoxico?"
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        
        response = generator.generate_response(query)
        
        print(f"\nRESPUESTA:\n{response.answer}")
        print(f"\nPICTOGRAMAS: {response.pictogramas}")
        print(f"CÓDIGOS H: {[c['codigo'] for c in response.structured_metadata.get('codigos_h', [])]}")
        print(f"LATENCIA: {response.latency_ms:.0f}ms")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()