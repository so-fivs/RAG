"""
src/rag/generator.py

Sistema RAG + LLM Generator SIMPLIFICADO
Integra retriever + LLM + contexto todo en uno.
VERSIÓN CORREGIDA CON API ANTIGUA (ESTABLE)
"""

import ollama
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from google.genai import types
import google.genai as genai  
from config import ProjectConfig
import sys

HarmCategory = types.HarmCategory
HarmBlockThreshold = types.HarmBlockThreshold
SafetySetting = types.SafetySetting
GenerationConfig = types.GenerateContentConfig
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

@dataclass
class GeneratedResponse:
    """Respuesta generada por el sistema."""
    answer: str
    sources: List[Dict[str, Any]]
    structured_metadata: Optional[Dict[str, Any]]
    pictogramas: List[str]
    latency_ms: float
    retrieval_metrics: Dict[str, Any]


class RAGGenerator:
    """Generador de respuestas RAG con LLM (TODO EN UNO)."""
    
    def __init__(self, config, use_gemini: bool = True, temperature: float = 0.3):  
        self.config = config
        self.temperature = temperature
        self.use_gemini = use_gemini
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
        
        # INICIALIZAR RETRIEVER PRIMERO
        try:
            self.retriever = HybridRetriever(config)
            self.context = ConversationalContext()
            print("Retriever y contexto inicializados")
        except Exception as e:
            print(f"Error inicializando retriever: {e}")
            raise
        
        if use_gemini:
            try:
                print(f"Configurando Gemini API...")
                self.client = genai.Client(api_key=config.GEMINI_API_KEY)
                self.model_name = 'gemini-2.5-flash'
                self.client.models.generate_content(
                    model=self.model_name,
                    contents="Test"
                )
                print(f"✅ Gemini inicializado: {self.model_name}")
                                    
            except Exception as e:
                print(f"⚠️ Error Gemini, usando Ollama: {e}")
                self.use_gemini = False
                self.llm_model = "llama3.1:8b"
                print(f"🔄 Fallback a Ollama: {self.llm_model}")
        else:
            self.llm_model = "llama3.1:8b"
            print(f"🔄 Usando Ollama: {self.llm_model}")

    def generate_response(self, query: str, use_context: bool = True) -> GeneratedResponse:
        start_time = time.time()
        
        try:
            results, detected_product, structured_metadatas, metrics, product_info, images = self.retriever.retrieve(
                self.context.enrich_query(query) if use_context else query,
                context_product=self.context.current_product if use_context else None,
                n_candidates=30,
                n_final=5
            )
            
            if detected_product and use_context:
                self.context.set_product(detected_product)
            
            prompt = self._build_prompt(query, results, structured_metadatas, product_info)
            llm_response = self._call_llm(prompt)
            pictogramas = self._extract_pictogramas(structured_metadatas)
            sources = self._format_sources(results)
            metadata_dict = self._format_metadata(structured_metadatas)
            
            latency = (time.time() - start_time) * 1000
            all_images = pictogramas + [img['filename'] for img in images]
            all_images = list(set(all_images))
            
            return GeneratedResponse(
                answer=llm_response,
                sources=sources,
                structured_metadata=metadata_dict,
                pictogramas=all_images,
                latency_ms=latency,
                retrieval_metrics={
                    'avg_similarity': metrics.avg_similarity,
                    'top_similarity': metrics.top_similarity,
                    'total_chunks': metrics.total_chunks
                }
            )
        except Exception as e:
            print(f"Error: {e}")
            raise

    def _build_prompt(self, query: str, results: List[SearchResult], 
                      structured_metadatas: List[StructuredMetadata], 
                      product_info: Dict[str, Any]) -> str:
        prompt = f"""Eres un experto en seguridad química especializado en Fichas de Datos de Seguridad (FDS).

{'='*60}
INFORMACION DEL PRODUCTO
{'='*60}
Producto: {product_info.get('producto', 'N/A')}
Fabricante: {product_info.get('fabricante', 'N/A')}
Codigo: {product_info.get('codigo_producto', 'N/A')}
Fecha FDS: {product_info.get('fecha_fds', 'N/A')}

Codigos H identificados: {len(product_info.get('codigos_h', []))}
Codigos P identificados: {len(product_info.get('codigos_p', []))}
Componentes quimicos: {len(product_info.get('componentes_cas', []))}

PREGUNTA: {query}

"""
        
        for metadata in structured_metadatas:
            if metadata.codigos_h:
                prompt += f"\nCODIGOS DE PELIGRO:\n"
                for codigo in metadata.codigos_h:
                    prompt += f"- {codigo['codigo']}: {codigo['descripcion']}\n"
            
            if metadata.codigos_p:
                prompt += f"\nMEDIDAS DE PRECAUCION:\n"
                for codigo in metadata.codigos_p:
                    prompt += f"- {codigo['codigo']}: {codigo['descripcion']}\n"
            
            if metadata.componentes_cas:
                prompt += f"\nCOMPONENTES QUIMICOS:\n"
                for comp in metadata.componentes_cas:
                    conc = f" ({comp.get('concentracion', '')})" if comp.get('concentracion') != 'No especificada' else ""
                    prompt += f"- {comp['nombre']} - CAS: {comp['cas']}{conc}\n"
        
        prompt += "\nCONTEXTO DE LA FDS:\n"
        for i, result in enumerate(results[:5], 1):
            prompt += f"\n[Fragmento {i}]\n{result.content[:800]}\n"
        
        prompt += """

INSTRUCCIONES:
1. Proporciona una respuesta COMPLETA y DETALLADA (minimo 3 parrafos bien desarrollados)
2. Estructura: Introduccion y uso del producto -> Detalles tecnicos -> Recomendaciones practicas pero no pongas los titulos explicitamente
3. USA los codigos H/P y componentes CAS de manera implicitos en la respuesta, no digas los codigos sino solo la mencion
4. Si la informacion es insuficiente, especifica QUE datos faltan exactamente
5. Lenguaje tecnico pero accesible, sin jerga innecesaria
6. Para informacion sobre precauciones revisa controles de exposición/protección individual tambien.
7. No incuir informacion de otros productos o documentos si no encuentras directamente la respuesta.
RESPUESTA DETALLADA:
"""
        return prompt
    

    def _call_llm(self, prompt: str) -> str:
        if self.use_gemini:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=800
                    ), 
                )
                
                if response.text is None:
                    print("⚠️ ALERTA: Respuesta bloqueada por política de seguridad de Gemini (response.text es None).")
                    raise ValueError("Respuesta bloqueada, forzando fallback.")
                return response.text
            
            except Exception as e:
                print(f"⚠️ Error en Gemini, fallback a Ollama: {e}")
                self.use_gemini = False
                self.llm_model = "llama3.1:8b"
                return self._call_llm(prompt)
        else:
            try:
                print(f"  [OLLAMA] Generando con {self.llm_model}...")
                
                # ✅ Verificar que Ollama está corriendo
                import subprocess
                try:
                    subprocess.run(['ollama', 'list'], 
                                capture_output=True, 
                                timeout=5, 
                                check=True)
                except:
                    return "Error: Ollama no está corriendo. Ejecuta: ollama serve"
                
                # ✅ Generar con timeout
                response = ollama.generate(
                    model=self.llm_model,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature, 
                        'num_predict': 800,
                        'num_ctx': 4096  # ← Contexto reducido
                    }
                )
                
                # ✅ Verificar respuesta
                if not response or 'response' not in response:
                    return "Error: Ollama no devolvió respuesta válida"
                
                if len(response['response'].strip()) < 50:
                    return "Error: Respuesta de Ollama demasiado corta"
                
                print(f"  [OLLAMA] Respuesta generada: {len(response['response'])} chars")
                return response['response']
                
            except Exception as e:
                print(f"  [OLLAMA ERROR] {e}")
                return f"Error en Ollama: {str(e)}"
    
    def _extract_pictogramas(self, structured_metadatas: List[StructuredMetadata]) -> List[str]:
        """
        Extrae pictogramas de peligro basándose en códigos H.
        
        Flujo:
        1. Recibe lista de metadatas estructuradas
        2. Por cada metadata, itera sus códigos H
        3. Mapea código H → nombre de pictograma (ej: H225 → 'flame')
        4. Busca archivos en disco que contengan ese nombre
        5. Retorna lista de nombres de archivos únicos
        """
        if not structured_metadatas: 
            return []
        
        pictogramas = set()
        
        try:
            images_path = Path('/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad/data/extracted_content/images')
            
            if not images_path.exists():
                print(f"Carpeta de imagenes no encontrada: {images_path}")
                return []
            
            for metadata in structured_metadatas:
                if not metadata.codigos_h:  
                    continue
                
                for codigo in metadata.codigos_h:  
                    codigo_id = codigo.get('codigo', '')  
                    
                    if codigo_id in self.pictograma_mapping:
                        base_name = self.pictograma_mapping[codigo_id]
      
                        matching_files = list(images_path.glob(f"*{base_name}*.png")) + \
                                        list(images_path.glob(f"*{base_name}*.jpg")) + \
                                        list(images_path.glob(f"*{base_name}*.jpeg"))
                        
                        if matching_files:
                            pictogramas.add(matching_files[0].name)
            
        except Exception as e:
            print(f"Error extrayendo pictogramas: {e}")
        
        return list(pictogramas)
    
    def _format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
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
    
    def _format_metadata(self, structured_metadatas: List[StructuredMetadata]) -> Optional[Dict[str, Any]]:
        if not structured_metadatas:
            return None
        
        all_codigos_h = []
        all_codigos_p = []
        all_componentes_cas = []
        query_types = []
        total_chunks = 0
        
        for metadata in structured_metadatas:
            query_types.append(metadata.query_type)
            all_codigos_h.extend(metadata.codigos_h)
            all_codigos_p.extend(metadata.codigos_p)
            all_componentes_cas.extend(metadata.componentes_cas)
            total_chunks += metadata.extracted_from_chunks
        
        # Eliminar duplicados manteniendo orden
        unique_h = {c['codigo']: c for c in all_codigos_h}.values()
        unique_p = {c['codigo']: c for c in all_codigos_p}.values()
        unique_cas = {c['cas']: c for c in all_componentes_cas}.values()
        
        return {
            'query_types': list(set(query_types)),
            'codigos_h': list(unique_h),
            'codigos_p': list(unique_p),
            'componentes_cas': list(unique_cas),
            'extracted_from_chunks': total_chunks
        }    
    
    def reset_context(self):
        self.context.clear_product()
        print("Contexto reiniciado")


if __name__ == "__main__":
    try:
        from config import ProjectConfig
        
        config = ProjectConfig()
        generator = RAGGenerator(config, use_gemini=False)
        
        query = "Cuales son los peligros del esmalte epoxico?"
        response = generator.generate_response(query)
        
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print(f"{'='*80}")
        print(f"\nRESPUESTA:\n{response.answer}")
        print(f"\nPICTOGRAMAS: {response.pictogramas}")
        print(f"LATENCIA: {response.latency_ms:.0f}ms")
        print(f"\nFUENTES:")
        for i, source in enumerate(response.sources[:3], 1):
            print(f"  {i}. {source['producto']} (sim={source['similarity']})")
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()