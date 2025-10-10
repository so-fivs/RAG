"""
query_clarifier.py

Sistema de clarificación de queries ambiguas.
Genera contra-preguntas para guiar al usuario hacia información precisa.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass


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
        'vertido': ['derrame', 'vertido', 'fuga'],
        'manipulacion': ['manipulación', 'almacenamiento', 'precauciones'],
        'exposicion': ['exposición', 'protección', 'epi', 'ventilación'],
        'propiedades': ['propiedades', 'físicas', 'químicas', 'aspecto'],
        'estabilidad': ['estabilidad', 'reactividad', 'incompatibilidad'],
        'toxicologia': ['toxicología', 'toxicidad', 'efectos salud'],
        'ecologia': ['ecología', 'medio ambiente', 'biodegradación'],
        'eliminacion': ['eliminación', 'residuos', 'desechos'],
        'transporte': ['transporte', 'onu', 'adr', 'imdg'],
        'reglamentacion': ['reglamentación', 'normativa', 'regulación']
    }
    
    def detect_ambiguity(self, query: str, search_results: List) -> Clarification:
        """
        Detecta si la query necesita clarificación.
        
        Args:
            query: Pregunta del usuario
            search_results: Resultados de búsqueda inicial
            
        Returns:
            Objeto Clarification
        """
        query_lower = query.lower()
        
        # Verificar múltiples productos en resultados
        if search_results:
            productos = set(r.metadata.get('producto') for r in search_results if r.metadata.get('producto'))
            
            if len(productos) > 2:
                return Clarification(
                    needs_clarification=True,
                    question=f"Encontré información de {len(productos)} productos diferentes.",
                    reason="multiple_products",
                    suggestions=list(productos)[:5]
                )
        
        # Verificar términos genéricos de producto
        if any(term in query_lower for term in self.AMBIGUOUS_TERMS['producto_generico']):
            if not self._has_specific_product(query_lower):
                return Clarification(
                    needs_clarification=True,
                    question="¿A qué producto te refieres?",
                    reason="generic_product",
                    suggestions=self._get_available_products(search_results)
                )
        
        # Verificar consulta de límites ambigua
        if any(term in query_lower for term in self.AMBIGUOUS_TERMS['limites_ambiguos']):
            if not any(spec in query_lower for spec in ['dnel', 'pnec', 'explosividad', 'onu']):
                return Clarification(
                    needs_clarification=True,
                    question="¿Qué tipo de límite necesitas?",
                    reason="ambiguous_limits",
                    suggestions=[
                        "Límites de exposición ocupacional (DNEL/PNEC)",
                        "Límites de explosividad",
                        "Límites de transporte (ONU)"
                    ]
                )
        
        # Verificar información demasiado general
        if any(term in query_lower for term in self.AMBIGUOUS_TERMS['info_generica']):
            matched_section = self._match_section(query_lower)
            if not matched_section:
                return Clarification(
                    needs_clarification=True,
                    question="¿Sobre qué aspecto específico necesitas información?",
                    reason="too_general",
                    suggestions=[
                        "Composición química",
                        "Peligros y clasificación",
                        "Medidas de primeros auxilios",
                        "Manipulación y almacenamiento",
                        "Propiedades físicas"
                    ]
                )
        
        return Clarification(
            needs_clarification=False,
            question="",
            reason="specific_enough",
            suggestions=[]
        )
    
    def _has_specific_product(self, query: str) -> bool:
        """Verifica si la query menciona un producto específico."""
        specific_indicators = ['epóxico', 'epoxi', 'uretano', 'alquídico', 'texturizada']
        return any(ind in query for ind in specific_indicators)
    
    def _get_available_products(self, search_results: List) -> List[str]:
        """Extrae productos únicos de los resultados."""
        if not search_results:
            return []
        
        productos = set(r.metadata.get('producto') for r in search_results if r.metadata.get('producto'))
        return list(productos)[:5]
    
    def _match_section(self, query: str) -> Optional[str]:
        """Identifica si la query menciona una sección específica."""
        for section, keywords in self.SPECIFIC_SECTIONS.items():
            if any(kw in query for kw in keywords):
                return section
        return None
    
    def format_clarification_message(self, clarification: Clarification) -> str:
        """
        Formatea mensaje de clarificación para el usuario.
        
        Args:
            clarification: Objeto Clarification
            
        Returns:
            Mensaje formateado
        """
        if not clarification.needs_clarification:
            return ""
        
        message = clarification.question
        
        if clarification.suggestions:
            message += "\n\nOpciones:\n"
            for i, sugg in enumerate(clarification.suggestions, 1):
                message += f"{i}. {sugg}\n"
        
        return message


class ConversationalContext:
    """Mantiene contexto de conversación para queries de seguimiento."""
    
    def __init__(self):
        self.history: List[Dict] = []
        self.current_product: Optional[str] = None
        self.current_section: Optional[str] = None
    
    def add_turn(self, query: str, response: str, metadata: Dict = None):
        """Registra turno de conversación."""
        self.history.append({
            'query': query,
            'response': response,
            'metadata': metadata or {}
        })
        
        if metadata and metadata.get('producto'):
            self.current_product = metadata['producto']
    
    def get_context_for_query(self, query: str) -> str:
        """
        Enriquece query con contexto conversacional.
        
        Args:
            query: Query actual del usuario
            
        Returns:
            Query enriquecida con contexto
        """
        enriched = query
        
        if self.current_product and not any(p in query.lower() for p in ['epóxico', 'epoxi', 'uretano', 'producto']):
            enriched = f"{query} (producto: {self.current_product})"
        
        return enriched
    
    def clear(self):
        """Limpia el contexto."""
        self.history = []
        self.current_product = None
        self.current_section = None


def main():
    """Ejemplos de uso del clarificador."""
    clarifier = QueryClarifier()
    
    test_queries = [
        ("¿Qué componentes tiene?", []),
        ("Dame información sobre límites", []),
        ("¿Cuáles son los peligros del esmalte epóxico?", []),
        ("Dime todo sobre el producto", [])
    ]
    
    print("EJEMPLOS DE CLARIFICACIÓN\n")
    
    for query, results in test_queries:
        print(f"Query: {query}")
        clarification = clarifier.detect_ambiguity(query, results)
        
        if clarification.needs_clarification:
            message = clarifier.format_clarification_message(clarification)
            print(f"Necesita clarificación: {clarification.reason}")
            print(f"Respuesta:\n{message}")
        else:
            print("Query suficientemente específica")
        
        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()