import os
import re
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime
import fitz
import pdfplumber
import camelot
from camelot.core import TableList
from io import StringIO
import shutil
from dataclasses import dataclass
import hashlib

# ============================================================================
# CLASE DE CONFIGURACIÓN DEL PROYECTO
# ============================================================================

class ProjectConfig:
    """Configuración centralizada del proyecto, ahora integrada en el extractor."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        self.folders = {
            'raw_data': self.base_path / 'data' / 'raw_documents',
            'extracted_content': self.base_path / 'data' / 'extracted_content',
            'processed_chunks': self.base_path / 'data' / 'processed_chunks',
            'embeddings': self.base_path / 'data' / 'embeddings',
            'vector_db': self.base_path / 'data' / 'vector_db',
            'logs': self.base_path / 'logs',
            'outputs': self.base_path / 'outputs'
        }
        
        self.extracted_subfolders = {
            'texts': self.folders['extracted_content'] / 'texts',
            'tables': self.folders['extracted_content'] / 'tables',
            'images': self.folders['extracted_content'] / 'images',
            'metadata': self.folders['extracted_content'] / 'metadata'
        }
        
    def get_folder(self, folder_name: str) -> Path:
        """Obtiene una ruta de carpeta por su nombre."""
        if folder_name in self.folders:
            return self.folders[folder_name]
        elif folder_name in self.extracted_subfolders:
            return self.extracted_subfolders[folder_name]
        raise ValueError(f"Carpeta no encontrada: {folder_name}")

    def create_folders(self):
        """Crea todas las carpetas del proyecto si no existen."""
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        for subfolder in self.extracted_subfolders.values():
            subfolder.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CLASES DE DATOS PARA CHUNKING
# ============================================================================

@dataclass
class ChunkMetadata:
    """Metadatos de un chunk individual"""
    document_id: str
    section_number: int
    section_name: str
    chunk_sequence: int
    content_type: str  # 'text', 'table', 'mixed'
    chemical_entities: List[str]
    cas_numbers: List[str]
    hazard_codes: List[str]
    exposure_limits: List[str]
    token_count: int
    overlap_with_previous: bool = False

@dataclass
class Chunk:
    """Estructura de un chunk individual"""
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    keywords: List[str]


# ============================================================================
# PROCESADOR DE TEXTO Y ENTIDADES QUÍMICAS
# ============================================================================

class ChemicalEntityExtractor:
    """Extrae y normaliza entidades químicas del texto"""
    
    def __init__(self):
        self.cleanup_patterns = {
            # Metadatos y headers redundantes
            'headers_redundant': r'(?i)(?:sección|section)\s*\d{1,2}[:.\s-]+',
            'page_numbers': r'(?i)(?:página|page)\s*\d+\s*(?:de|of)\s*\d+',
            'dates': r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
            'version_numbers': r'(?i)versión?\s*:?\s*\d+(?:\.\d+)*',
            'document_codes': r'\b[A-Z]{2,}\d{4,}\b',
            
            # Espaciado y formato
            'extra_spaces': r'\s{2,}',
            'line_breaks': r'\n{3,}',
            'bullet_points': r'^\s*[•\-\*]\s*',
            'table_separators': r'[|\-]{3,}',
        }
        
        self.chemical_patterns = {
            # Datos químicos específicos (PRESERVAR)
            'cas_numbers': r'\b\d{1,7}-\d{2}-\d{1}\b',
            'concentrations': r'≤?≥?\s*\d+(?:\.\d+)?\s*%?(?:\s*-\s*\d+(?:\.\d+)?\s*%?)?',
            'temperatures': r'-?\d+(?:\.\d+)?\s*°[CF](?:\s*\([^)]+\))?',
            'pressures': r'\d+(?:\.\d+)?\s*(?:kPa|mmHg|atm|bar|Pa|psi)',
            'molecular_formulas': r'\b[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*\b',
            
            # Códigos regulatorios
            'h_codes': r'\bH\d{3}\b',
            'p_codes': r'\bP\d{3}(?:\+P\d{3})*\b',
            'un_numbers': r'\bUN\d{4}\b',
            'osha_pel': r'(?i)(?:OSHA\s+PEL|PEL)\s*:?\s*[^\.]+',
            'acgih_tlv': r'(?i)(?:ACGIH\s+TLV|TLV)\s*:?\s*[^\.]+',
        }
    
    def clean_text(self, text: str) -> str:
        """Limpia el texto preservando entidades químicas importantes"""
        cleaned = text
        
        # Aplicar patrones de limpieza
        for pattern_name, pattern in self.cleanup_patterns.items():
            if pattern_name == 'extra_spaces':
                cleaned = re.sub(pattern, ' ', cleaned)
            elif pattern_name == 'line_breaks':
                cleaned = re.sub(pattern, '\n\n', cleaned)
            elif pattern_name == 'bullet_points':
                cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
            else:
                cleaned = re.sub(pattern, '', cleaned)
        
        return cleaned.strip()
    
    def extract_chemical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades químicas del texto"""
        entities = {}
        
        for entity_type, pattern in self.chemical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches)) if matches else []
        
        return entities
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave relevantes del texto"""
        # Palabras clave específicas del dominio químico
        chemical_keywords = set()
        
        # Extraer entidades químicas como keywords
        entities = self.extract_chemical_entities(text)
        for entity_list in entities.values():
            chemical_keywords.update(entity_list)
        
        # Palabras clave de seguridad comunes
        safety_terms = [
            'tóxico', 'corrosivo', 'inflamable', 'explosivo', 'irritante',
            'carcinógeno', 'mutágeno', 'sensibilizante', 'peligroso',
            'respirador', 'guantes', 'ventilación', 'almacenamiento',
            'temperatura', 'presión', 'incompatible'
        ]
        
        for term in safety_terms:
            if re.search(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                chemical_keywords.add(term)
        
        return list(chemical_keywords)


# ============================================================================
# CHUNKER INTELIGENTE
# ============================================================================

class IntelligentChunker:
    """Implementa estrategias de chunking inteligente para FDS"""
    
    def __init__(self, max_tokens: int = 512, min_tokens: int = 50, overlap_percentage: float = 0.15):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_percentage = overlap_percentage
        self.entity_extractor = ChemicalEntityExtractor()
        
        # Secciones prioritarias
        self.priority_sections = {2, 3, 8, 11}  # Peligros, composición, exposición, toxicología
        
    def estimate_tokens(self, text: str) -> int:
        """Estima el número de tokens (aproximadamente 4 caracteres = 1 token)"""
        return len(text) // 4
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """Divide el texto en oraciones preservando estructura"""
        # Patrones para dividir oraciones, considerando abreviaturas químicas
        sentence_endings = r'(?<!\bCAS\.)(?<!\bNo\.)(?<!\bvol\.)(?<!\bppm\.)(?<!\bmg\.)(?<![A-Z])\.(?:\s|$)|(?<!\d)[!?](?:\s|$)'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_text_chunks(self, text: str, document_id: str, section_num: int, section_name: str) -> List[Chunk]:
        """Crea chunks de texto con overlap inteligente"""
        cleaned_text = self.entity_extractor.clean_text(text)
        
        if self.estimate_tokens(cleaned_text) <= self.max_tokens:
            # Si el texto es pequeño, crear un solo chunk
            return [self._create_single_chunk(cleaned_text, document_id, section_num, section_name, 1)]
        
        chunks = []
        sentences = self.split_text_by_sentences(cleaned_text)
        
        current_chunk = ""
        current_sentences = []
        chunk_sequence = 1
        
        for i, sentence in enumerate(sentences):
            temp_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if self.estimate_tokens(temp_chunk) > self.max_tokens:
                if current_chunk:  # Si hay contenido acumulado
                    chunk = self._create_single_chunk(
                        current_chunk, document_id, section_num, section_name, chunk_sequence
                    )
                    chunks.append(chunk)
                    chunk_sequence += 1
                    
                    # Calcular overlap
                    overlap_size = int(len(current_sentences) * self.overlap_percentage)
                    overlap_sentences = current_sentences[-overlap_size:] if overlap_size > 0 else []
                    
                    current_chunk = " ".join(overlap_sentences + [sentence])
                    current_sentences = overlap_sentences + [sentence]
                else:
                    # Oración muy larga, crear chunk solo con ella
                    chunk = self._create_single_chunk(
                        sentence, document_id, section_num, section_name, chunk_sequence
                    )
                    chunks.append(chunk)
                    chunk_sequence += 1
                    current_chunk = ""
                    current_sentences = []
            else:
                current_chunk = temp_chunk
                current_sentences.append(sentence)
        
        # Agregar último chunk si existe
        if current_chunk and self.estimate_tokens(current_chunk) >= self.min_tokens:
            chunk = self._create_single_chunk(
                current_chunk, document_id, section_num, section_name, chunk_sequence
            )
            chunks.append(chunk)
        
        return chunks
    
    def create_table_chunks(self, tables: List[pd.DataFrame], document_id: str, section_num: int) -> List[Chunk]:
        """Convierte tablas a chunks de texto estructurado"""
        chunks = []
        
        for table_idx, df in enumerate(tables):
            # Normalizar headers de tabla
            normalized_df = self._normalize_table_headers(df)
            
            # Crear chunks por filas para tablas de composición
            if self._is_composition_table(normalized_df):
                row_chunks = self._create_composition_row_chunks(
                    normalized_df, document_id, section_num, table_idx
                )
                chunks.extend(row_chunks)
            else:
                # Para otras tablas, crear un chunk general
                table_text = self._table_to_structured_text(normalized_df)
                chunk = self._create_single_chunk(
                    table_text, document_id, section_num, f"Tabla {table_idx + 1}", 1, 'table'
                )
                chunks.append(chunk)
        
        return chunks
    
    def _normalize_table_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza headers de tabla a términos estándar"""
        header_mapping = {
            'ingredient': 'ingrediente',
            'component': 'componente',
            'substance': 'sustancia',
            'cas': 'cas_number',
            'cas number': 'cas_number',
            'cas no': 'cas_number',
            'concentration': 'concentracion',
            'percentage': 'porcentaje',
            '% weight': 'porcentaje_peso',
            'wt%': 'porcentaje_peso',
            'hazard': 'peligro',
            'classification': 'clasificacion'
        }
        
        new_columns = []
        for col in df.columns:
            col_lower = str(col).lower().strip()
            mapped_col = header_mapping.get(col_lower, col)
            new_columns.append(mapped_col)
        
        df.columns = new_columns
        return df
    
    def _is_composition_table(self, df: pd.DataFrame) -> bool:
        """Determina si una tabla es de composición química"""
        composition_indicators = ['ingrediente', 'componente', 'cas_number', 'concentracion', 'porcentaje']
        columns_lower = [str(col).lower() for col in df.columns]
        
        matches = sum(1 for indicator in composition_indicators if any(indicator in col for col in columns_lower))
        return matches >= 2
    
    def _create_composition_row_chunks(self, df: pd.DataFrame, document_id: str, section_num: int, table_idx: int) -> List[Chunk]:
        """Crea un chunk por fila en tablas de composición"""
        chunks = []
        
        for row_idx, row in df.iterrows():
            row_text_parts = []
            
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    row_text_parts.append(f"{col}: {value}")
            
            if row_text_parts:
                row_text = " | ".join(row_text_parts)
                chunk_id = f"{document_id}_S{section_num:02d}_T{table_idx+1}_R{row_idx+1}"
                
                entities = self.entity_extractor.extract_chemical_entities(row_text)
                keywords = self.entity_extractor.extract_keywords(row_text)
                
                metadata = ChunkMetadata(
                    document_id=document_id,
                    section_number=section_num,
                    section_name=f"Tabla {table_idx + 1} - Fila {row_idx + 1}",
                    chunk_sequence=row_idx + 1,
                    content_type='table',
                    chemical_entities=entities.get('molecular_formulas', []),
                    cas_numbers=entities.get('cas_numbers', []),
                    hazard_codes=entities.get('h_codes', []),
                    exposure_limits=entities.get('osha_pel', []) + entities.get('acgih_tlv', []),
                    token_count=self.estimate_tokens(row_text)
                )
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    content=row_text,
                    metadata=metadata,
                    keywords=keywords
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _table_to_structured_text(self, df: pd.DataFrame) -> str:
        """Convierte tabla a texto estructurado legible"""
        text_lines = []
        
        for _, row in df.iterrows():
            row_parts = []
            for col, value in row.items():
                if pd.notna(value) and str(value).strip():
                    row_parts.append(f"{col}: {value}")
            
            if row_parts:
                text_lines.append(" | ".join(row_parts))
        
        return "\n".join(text_lines)
    
    def _create_single_chunk(self, content: str, document_id: str, section_num: int, 
                           section_name: str, sequence: int, content_type: str = 'text') -> Chunk:
        """Crea un chunk individual con todos los metadatos"""
        chunk_id = f"{document_id}_S{section_num:02d}_{sequence:03d}"
        
        entities = self.entity_extractor.extract_chemical_entities(content)
        keywords = self.entity_extractor.extract_keywords(content)
        
        metadata = ChunkMetadata(
            document_id=document_id,
            section_number=section_num,
            section_name=section_name,
            chunk_sequence=sequence,
            content_type=content_type,
            chemical_entities=entities.get('molecular_formulas', []),
            cas_numbers=entities.get('cas_numbers', []),
            hazard_codes=entities.get('h_codes', []) + entities.get('p_codes', []),
            exposure_limits=entities.get('osha_pel', []) + entities.get('acgih_tlv', []),
            token_count=self.estimate_tokens(content),
            overlap_with_previous=sequence > 1
        )
        
        return Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            keywords=keywords
        )


# ============================================================================
# CLASE PRINCIPAL DEL EXTRACTOR MEJORADA
# ============================================================================

class FDSExtractorAdvanced:
    """Extractor completo para Fichas de Datos de Seguridad con chunking inteligente"""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.chunker = IntelligentChunker()
        
        # Estadísticas de procesamiento
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_pages': 0,
            'total_tables': 0,
            'total_images': 0,
            'total_chunks': 0,
            'priority_section_chunks': 0
        }
        
        self.section_names = {
            1: "Identificación del producto y la empresa",
            2: "Identificación de peligros", 
            3: "Composición/información sobre los ingredientes",
            4: "Primeros auxilios",
            5: "Medidas de lucha contra incendios",
            6: "Medidas en caso de liberación accidental",
            7: "Manejo y almacenamiento",
            8: "Controles de exposición/protección personal",
            9: "Propiedades físicas y químicas",
            10: "Estabilidad y reactividad",
            11: "Información toxicológica",
            12: "Información ecológica",
            13: "Consideraciones de eliminación",
            14: "Información sobre transporte",
            15: "Información reglamentaria",
            16: "Otra información"
        }
        
        self.fds_sections = {
            1: [r"identificaci[oó]n(?:\s+del?\s+producto)?", r"identificacion(?:\s+del?\s+producto)?", r"identif(?:icador)?", r"nombre.*producto", r"nombre.*comercial", r"denominaci[oó]n", r"datos.*producto", r"informaci[oó]n.*producto", r"producto.*qu[ií]mico", r"sustancia.*qu[ií]mica", r"datos.*identificaci[oó]n", r"código.*producto", r"numero.*identificaci[oó]n", r"product\s+identifier", r"product\s+name", r"identification", r"product.*identification", r"product.*name", r"trade.*name", r"product.*information", r"chemical.*product", r"product.*code", r"manufacturer", r"supplier", r"fabricante", r"proveedor", r"ficha.*identificaci[oó]n", r"datos.*empresa", r"company.*information", r"emergency.*contact", r"contacto.*emergencia", r"recommended.*use", r"uso.*recomendado"],
            2: [r"identificaci[oó]n.*peligros", r"peligros", r"riesgos?", r"hazard.*identification", r"clasificaci[oó]n.*peligro", r"advertencias.*peligro", r"informaci[oó]n.*peligros", r"elementos.*etiqueta", r"pictogramas?", r"signal.*word", r"palabra.*advertencia", r"hazard.*identification", r"hazards?", r"risks?", r"dangers?", r"risk.*assessment", r"hazard.*classification", r"safety.*hazards", r"warnings?", r"caution", r"pictograms?", r"etiquetado", r"labeling", r"labelling", r"ghs", r"sga", r"frases.*[hp]", r"[hp].*phrases", r"frases.*precauci[oó]n", r"precautionary.*statements", r"signal.*word", r"indicaciones.*peligro", r"hazard.*statements", r"clasificaci[oó]n", r"classification", r"categor[ií]a.*peligro", r"hazard.*category", r"otros.*peligros", r"other.*hazards", r"hnoc", r"not.*otherwise.*classified"],
            3: [r"composici[oó]n", r"ingredientes", r"componentes", r"contenido", r"informaci[oó]n.*composici[oó]n", r"sustancias.*peligrosas", r"mezcla", r"preparado", r"formulaci[oó]n", r"constituci[oó]n", r"contenido.*qu[ií]mico", r"composition", r"ingredients", r"components", r"contents", r"mixture", r"information.*composition", r"hazardous.*ingredients", r"chemical.*composition", r"formulation", r"constitution", r"chemical.*identity", r"cas.*number", r"n[uú]mero.*cas", r"concentraci[oó]n", r"porcentaje", r"percentage", r"% peso", r"% weight", r"formula", r"f[oó]rmula", r"einecs", r"ec.*number", r"registro", r"registration", r"reach", r"wt\.?%", r"w/w", r"peso/peso", r"range", r"rango", r"limits?", r"l[ií]mites?"],
            4: [r"primeros\s+auxilios", r"medidas.*primeros.*auxilios", r"primeros.*auxilios", r"atenci[oó]n.*m[eé]dica", r"tratamiento.*urgencia", r"medidas.*emergencia", r"descripci[oó]n.*primeros.*auxilios", r"socorro", r"auxilio.*inmediato", r"first\s+aid", r"first.*aid.*measures", r"emergency.*measures", r"medical.*attention", r"emergency.*treatment", r"immediate.*care", r"emergency.*care", r"inhalaci[oó]n", r"ingesti[oó]n", r"contacto.*piel", r"contacto.*ojos", r"inhalation", r"ingestion", r"skin.*contact", r"eye.*contact", r"dermal.*contact", r"oral", r"respiratorio", r"respiratory", r"s[ií]ntomas", r"symptoms", r"efectos", r"effects", r"reacciones.*adversas", r"adverse.*reactions", r"signos", r"signs", r"manifestaciones"],
            5: [r"medidas.*incendio", r"lucha.*incendio", r"extinción", r"extincion", r"medidas.*extinción", r"fuego", r"combustión", r"combustion", r"medidas.*contra.*incendio", r"protección.*incendio", r"combate.*incendio", r"fire.*fighting", r"firefighting", r"fire.*extinguishing", r"fire.*protection", r"fire.*hazards", r"extinguishing.*measures", r"fire.*suppression", r"extintores?", r"extinguishers?", r"medios.*extinción", r"extinction.*media", r"agentes.*extintores", r"extinguishing.*agents", r"suppression.*agents", r"agua", r"water", r"espuma", r"foam", r"co2", r"polvo.*qu[ií]mico", r"dry.*chemical", r"polvo.*seco", r"arena", r"sand", r"peligros.*espec[ií]ficos", r"specific.*hazards", r"productos.*combustión", r"combustion.*products", r"humos.*tóxicos", r"toxic.*fumes"],
            6: [r"medidas.*derrame", r"liberación.*accidental", r"liberacion.*accidental", r"vertido", r"fuga", r"escape", r"derrame", r"spill", r"medidas.*vertido", r"contaminación", r"medidas.*contención", r"limpieza", r"neutralización", r"neutralizacion", r"medidas.*caso.*vertido", r"accidental.*release", r"spill", r"leak", r"spillage", r"leakage", r"environmental.*precautions", r"containment", r"cleanup", r"clean.*up", r"neutralization", r"absorption", r"release.*measures", r"absorber", r"contener", r"neutralizar", r"absorb", r"contain", r"neutralize", r"recoger", r"collect", r"sweep", r"barrer", r"aspirar", r"vacuum", r"precauciones.*ambientales", r"environmental.*precautions"],
            7: [r"manejo.*almacenamiento", r"manejo", r"almacenamiento", r"manipulaci[oó]n", r"manipulacion", r"almacén", r"almacenaje", r"conservación", r"conservacion", r"handling.*storage", r"precauciones.*manipulación", r"precauciones.*manejo", r"handling.*storage", r"handling", r"storage", r"manipulation", r"warehouse", r"conservation", r"custody", r"safe.*handling", r"safe.*storage", r"condiciones.*almacen", r"storage.*conditions", r"temperatura", r"temperature", r"humedad", r"humidity", r"incompatibles", r"incompatible", r"segregaci[oó]n", r"segregation", r"ventilaci[oó]n", r"ventilation", r"iluminaci[oó]n", r"precauciones.*segura", r"safe.*precautions", r"medidas.*seguridad", r"safety.*measures", r"buenas.*prácticas", r"good.*practices"],
            8: [r"controles.*exposici[oó]n", r"controles.*exposicion", r"protecci[oó]n.*personal", r"proteccion.*personal", r"epp", r"equipos.*protecci[oó]n", r"equipos.*proteccion", r"medidas.*protecci[oó]n", r"l[ií]mites.*exposici[oó]n", r"valores.*l[ií]mite", r"exposure.*controls", r"personal.*protection", r"ppe", r"protection.*equipment", r"exposure.*limits", r"limit.*values", r"occupational.*exposure", r"workplace.*exposure", r"personal.*protective.*equipment", r"tlv", r"pel", r"stel", r"twa", r"ceiling", r"techo", r"acgih", r"osha", r"niosh", r"límite.*permisible", r"permissible.*limit", r"respirador", r"guantes", r"gafas", r"ropa.*protecci[oó]n", r"respirator", r"gloves", r"goggles", r"protective.*clothing", r"safety.*glasses", r"face.*shield", r"pantalla.*facial", r"botas", r"boots"],
            9: [r"propiedades.*f[ií]sicas", r"propiedades.*qu[ií]micas", r"propiedades.*fisicas", r"propiedades.*quimicas", r"caracter[ií]sticas.*f[ií]sicas", r"datos.*f[ií]sicos", r"aspectos?", r"apariencia", r"color", r"olor", r"aspecto.*f[ií]sico", r"physical.*chemical.*properties", r"physical.*properties", r"chemical.*properties", r"appearance", r"color", r"colour", r"odor", r"odour", r"physical.*characteristics", r"densidad", r"viscosidad", r"solubilidad", r"ph", r"punto.*ebullici[oó]n", r"punto.*fusi[oó]n", r"density", r"viscosity", r"solubility", r"boiling.*point", r"melting.*point", r"flash.*point", r"vapor.*pressure", r"vapour.*pressure", r"punto.*inflamaci[oó]n", r"presi[oó]n.*vapor", r"estado.*f[ií]sico", r"physical.*state", r"forma", r"form", r"molecular.*weight", r"peso.*molecular"],
            10: [r"estabilidad.*reactividad", r"estabilidad", r"reactividad", r"incompatibilidad", r"descomposici[oó]n", r"descomposicion", r"polimerizaci[oó]n", r"polimerizacion", r"condiciones.*evitar", r"materiales.*evitar", r"estabilidad.*qu[ií]mica", r"stability.*reactivity", r"stability", r"reactivity", r"incompatibility", r"decomposition", r"polymerization", r"polymerisation", r"conditions.*avoid", r"materials.*avoid", r"chemical.*stability", r"thermal.*stability", r"productos.*descomposici[oó]n", r"decomposition.*products", r"reacciones.*peligrosas", r"hazardous.*reactions", r"condiciones.*inestabilidad", r"instability.*conditions", r"catalizadores", r"catalysts", r"inhibidores", r"inhibitors"],
            11: [r"informaci[oó]n.*toxicol[oó]gica", r"toxicidad", r"toxicolog[ií]a", r"efectos.*t[oó]xicos", r"peligros.*salud", r"datos.*toxicol[oó]gicos", r"toxicological.*information", r"toxicity", r"toxicology", r"toxic.*effects", r"health.*hazards", r"toxicological.*data", r"dosis.*letal", r"ld50", r"lc50", r"lethal.*dose", r"carcinogenicidad", r"mutagenicidad", r"toxicidad.*reproductiva", r"carcinogenicity", r"mutagenicity", r"reproductive.*toxicity", r"sensibilizaci[oó]n", r"sensitization", r"irritaci[oó]n", r"irritation", r"corrosividad", r"corrosivity", r"toxicidad.*aguda", r"acute.*toxicity", r"toxicidad.*cr[oó]nica", r"chronic.*toxicity", r"v[ií]as.*exposici[oó]n", r"exposure.*routes", r"s[ií]ntomas", r"symptoms", r"efectos.*inmediatos", r"immediate.*effects", r"efectos.*retardados", r"delayed.*effects"],
            12: [r"informaci[oó]n.*ecol[oó]gica", r"ecotoxicidad", r"peligros.*ambientales", r"efectos.*ambientales", r"datos.*ecol[oó]gicos", r"ecological.*information", r"ecotoxicity", r"environmental.*hazards", r"environmental.*effects", r"ecological.*data", r"toxicidad.*acu[aá]tica", r"aquatic.*toxicity", r"persistencia", r"persistence", r"biodegradabilidad", r"biodegradability", r"bioacumulaci[oó]n", r"bioaccumulation", r"movilidad.*suelo", r"soil.*mobility", r"resultados.*pbt", r"pbt.*results", r"evaluaci[oó]n.*vpvb", r"vpvb.*assessment", r"otros.*efectos.*adversos", r"other.*adverse.*effects"],
            13: [r"eliminaci[oó]n", r"eliminacion", r"disposici[oó]n", r"disposicion", r"desecho", r"residuos", r"tratamiento.*residuos", r"disposal", r"waste", r"waste.*treatment", r"disposal.*considerations", r"consideraciones.*eliminaci[oó]n", r"m[eé]todos.*eliminaci[oó]n", r"disposal.*methods", r"tratamiento.*previo", r"pretreatment", r"envases.*contaminados", r"contaminated.*packaging", r"precauciones.*eliminaci[oó]n", r"disposal.*precautions"],
            14: [r"transporte", r"informaci[oó]n.*transporte", r"transportation", r"transport.*information", r"shipping", r"n[uú]mero.*onu", r"un.*number", r"clase.*peligro", r"hazard.*class", r"grupo.*embalaje", r"packing.*group", r"riesgo.*subsidiario", r"subsidiary.*risk", r"peligros.*ambientales", r"environmental.*hazards", r"precauciones.*transporte", r"transport.*precautions", r"transporte.*mar[ií]timo", r"sea.*transport", r"transporte.*a[eé]reo", r"air.*transport", r"transporte.*terrestre", r"land.*transport", r"etiquetas.*transporte", r"transport.*labels"],
            15: [r"informaci[oó]n.*reglamentaria", r"reglamentaria", r"regulaci[oó]n", r"normativa", r"legislaci[oó]n", r"regulatory.*information", r"regulations?", r"legislation", r"normative", r"legal", r"reach", r"clp", r"osha", r"epa", r"dot", r"iata", r"imo", r"registros?", r"registrations?", r"autorizaciones?", r"authorizations?", r"restricciones?", r"restrictions?", r"evaluaci[oó]n.*seguridad", r"safety.*assessment", r"directivas", r"directives", r"reglamentos?", r"standards?", r"normas?"],
            16: [r"otra.*informaci[oó]n", r"informaci[oó]n.*adicional", r"other.*information", r"additional.*information", r"miscel[aá]nea", r"miscellaneous", r"fecha.*revisi[oó]n", r"revision.*date", r"versi[oó]n", r"version", r"preparado.*por", r"prepared.*by", r"referencias?", r"references?", r"abreviaturas?", r"abbreviations?", r"glosario", r"glossary", r"exenci[oó]n.*responsabilidad", r"disclaimer", r"fuentes.*informaci[oó]n", r"information.*sources", r"contacto.*t[eé]cnico", r"technical.*contact", r"capacitaci[oó]n", r"training", r"literatura.*t[eé]cnica", r"technical.*literature"]
        }

    def _extract_tables_camelot(self, file_path: str) -> TableList:
        """Extrae tablas usando Camelot con opciones mejoradas y sin conflictos."""
        try:
            tables = camelot.read_pdf(file_path, flavor='lattice', pages='all', line_scale=40,
                                       split_text=True, strip_text=' \n\t')
            if not tables or tables.n < 1:
                print("  - camelot: No se encontraron tablas de tipo 'lattice'. Intentando con 'stream'...")
                tables = camelot.read_pdf(file_path, flavor='stream', pages='all',
                                           split_text=True, strip_text=' \n\t')
            return tables
        except Exception as e:
            print(f"❌ Error al extraer tablas con Camelot: {e}")
            return camelot.core.TableList([])

    def _extract_tables_pdfplumber(self, doc):
        """Extrae tablas usando pdfplumber."""
        tables_df = []
        try:
            with pdfplumber.open(doc.name) as pdf:
                for page in pdf.pages:
                    plumber_tables = page.extract_tables()
                    for table_data in plumber_tables:
                        df = pd.DataFrame(table_data)
                        df = df.replace('', pd.NA).dropna(how='all')
                        if not df.empty:
                            tables_df.append(df)
            return tables_df
        except Exception as e:
            print(f"❌ Error al extraer tablas con pdfplumber: {e}")
            return []
            
    def extract_tables(self, doc) -> List[pd.DataFrame]:
        file_path = doc.name
        tables = self._extract_tables_camelot(file_path)

        if tables.n > 0:
            print(f"  - camelot: {tables.n} tablas validadas")
            camelot_tables = [table.df for table in tables]
            self.stats['total_tables'] += len(camelot_tables)
            return camelot_tables
        else:
            print("  - camelot: No se encontraron tablas. Intentando con pdfplumber...")
            plumber_tables = self._extract_tables_pdfplumber(doc)
            print(f"  - pdfplumber: {len(plumber_tables)} tablas validadas")
            self.stats['total_tables'] += len(plumber_tables)
            return plumber_tables

    def _extract_images_from_pdf(self, doc, base_name: str, output_folder: Path) -> List[str]:
        """Extrae y guarda imágenes de un documento PDF."""
        images = []
        output_folder.mkdir(parents=True, exist_ok=True)
        
        for i, page in enumerate(doc):
            pixmaps = page.get_images(full=True)
            for j, img in enumerate(pixmaps):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                image_path = output_folder / f"{base_name}_page_{i+1}_img_{j+1}.png"
                pix.save(image_path)
                images.append(str(image_path))
        return images

    def _clean_section_text(self, text: str, section_num: int) -> str:
        """Limpia el texto de una sección, eliminando artefactos y espacios extra."""
        regex_header = r"(?i)(?:secci[oó]n)?\s*" + re.escape(str(section_num)) + r"[:.\s-]+\s*"
        cleaned_text = re.sub(regex_header, '', text, count=1).strip()
        
        cleaned_text = re.sub(r'\s*\n\s*', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def _get_all_pages_text(self, doc) -> str:
        all_text = ""
        for page in doc:
            all_text += page.get_text()
        return all_text
    
    def _extract_sections_with_regex(self, full_text: str) -> Dict[int, str]:
        sections_content = {}
        found_section_numbers = []
        header_positions = []
        
        for sec_num in range(1, 17):
            section_patterns = self.fds_sections.get(sec_num, [])
            for pattern in section_patterns:
                full_pattern = r"(?i)(?:^|\n|\r)\s*(?:secci[oó]n)?\s*(\d{1,2})[:.\s-]+\s*" + pattern
                for match in re.finditer(full_pattern, full_text):
                    matched_num = int(match.group(1))
                    if matched_num == sec_num:
                        header_positions.append({'num': sec_num, 'start': match.start(), 'text': match.group(0)})
        
        header_positions.sort(key=lambda x: x['start'])

        unique_headers = []
        seen_nums = set()
        for header in header_positions:
            if header['num'] not in seen_nums:
                unique_headers.append(header)
                seen_nums.add(header['num'])

        for i, header in enumerate(unique_headers):
            start = header['start'] + len(header['text'])
            end = unique_headers[i+1]['start'] if i+1 < len(unique_headers) else len(full_text)
            section_content = full_text[start:end].strip()
            cleaned_content = self._clean_section_text(section_content, header['num'])
            sections_content[header['num']] = cleaned_content
            found_section_numbers.append(header['num'])
        
        all_sections = set(range(1, 17))
        found_sections = set(found_section_numbers)
        missing_sections = sorted(list(all_sections - found_sections))
        
        if missing_sections:
            print(f"  ⚠️ Secciones faltantes ({len(missing_sections)}): {', '.join(map(str, missing_sections))}. Buscando contenido relacionado...")
            found_map = {h['num']: h for h in unique_headers}
            for missing_num in missing_sections:
                prev_num = max([n for n in found_sections if n < missing_num], default=None)
                next_num = min([n for n in found_sections if n > missing_num], default=None)
                
                if prev_num and next_num:
                    start_content = sections_content.get(prev_num, "")
                    end_content = sections_content.get(next_num, "")
                    
                    combined_content = start_content + "\n" + end_content
                    
                    if len(combined_content) > len(start_content) + len(end_content) - 10:
                        sections_content[missing_num] = combined_content
                        print(f"    ✅ Contenido reasignado a Sección {missing_num}")
                    else:
                        print(f"    ❌ No se encontró contenido para reasignar a Sección {missing_num}")
                else:
                    print(f"    ❌ No se encontró contenido para reasignar a Sección {missing_num}")

        return sections_content

    def _save_sections(self, output_path: str, sections: Dict[int, str]):
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sorted_sections = sorted(sections.keys())
        full_content = ""
        
        for num in sorted_sections:
            title = self.section_names.get(num, f"Sección {num}")
            content = sections[num]
            if content:
                full_content += f"## Sección {num}: {title}\n\n{content.strip()}\n\n---\n\n"
        
        if full_content:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
        
    def _save_tables(self, tables: List[pd.DataFrame], base_name: str, output_folder: Path):
        output_folder.mkdir(parents=True, exist_ok=True)
        for i, table_df in enumerate(tables):
            table_path = output_folder / f"{base_name}_table_{i+1}.csv"
            table_df.to_csv(table_path, index=False, encoding='utf-8')

    # ============================================================================
    # NUEVOS MÉTODOS PARA CHUNKING INTELIGENTE
    # ============================================================================

    def generate_chunks_for_doc(self, pdf_path: str) -> List[Chunk]:
        """
        Método principal mejorado que genera múltiples chunks inteligentes por documento.
        """
        file_path_obj = Path(pdf_path)
        document_id = file_path_obj.stem
        
        print(f"🔄 Generando chunks para: {document_id}")
        
        all_chunks = []
        
        try:
            with fitz.open(pdf_path) as doc:
                # 1. Extraer texto completo y secciones
                full_text = self._get_all_pages_text(doc)
                sections = self._extract_sections_with_regex(full_text)
                
                if not sections:
                    print(f"❌ No se encontraron secciones en: {document_id}")
                    return []
                
                # 2. Procesar secciones de texto con prioridad
                for section_num in sorted(sections.keys()):
                    section_text = sections[section_num]
                    section_name = self.section_names.get(section_num, f"Sección {section_num}")
                    
                    if not section_text.strip():
                        continue
                    
                    # Crear chunks para esta sección
                    section_chunks = self.chunker.create_text_chunks(
                        section_text, document_id, section_num, section_name
                    )
                    
                    all_chunks.extend(section_chunks)
                    
                    # Contar chunks de secciones prioritarias
                    if section_num in self.chunker.priority_sections:
                        self.stats['priority_section_chunks'] += len(section_chunks)
                
                # 3. Procesar tablas
                tables = self.extract_tables(doc)
                if tables:
                    print(f"  📊 Procesando {len(tables)} tablas...")
                    
                    # Asignar tablas a secciones más probables
                    for table_idx, table_df in enumerate(tables):
                        # Determinar sección más probable basada en contenido de tabla
                        likely_section = self._determine_table_section(table_df)
                        
                        table_chunks = self.chunker.create_table_chunks(
                            [table_df], document_id, likely_section
                        )
                        all_chunks.extend(table_chunks)
                
                # 4. Actualizar estadísticas
                self.stats['total_chunks'] += len(all_chunks)
                
                print(f"  ✅ Generados {len(all_chunks)} chunks para {document_id}")
                print(f"     - Secciones procesadas: {len(sections)}")
                print(f"     - Tablas procesadas: {len(tables)}")
                print(f"     - Chunks prioritarios: {sum(1 for chunk in all_chunks if chunk.metadata.section_number in self.chunker.priority_sections)}")
                
                return all_chunks
                
        except Exception as e:
            print(f"❌ Error generando chunks para {document_id}: {e}")
            return []

    def _determine_table_section(self, df: pd.DataFrame) -> int:
        """Determina la sección más probable para una tabla basada en su contenido."""
        # Convertir toda la tabla a texto para análisis
        table_text = df.to_string().lower()
        
        # Patrones para diferentes tipos de tabla
        section_indicators = {
            3: ['ingredient', 'componente', 'cas', 'concentracion', 'porcentaje', '%'],
            8: ['pel', 'tlv', 'limite', 'exposicion', 'osha', 'acgih'],
            9: ['densidad', 'ph', 'temperatura', 'presion', 'solubilidad'],
            11: ['ld50', 'lc50', 'toxicidad', 'dosis'],
            14: ['un', 'clase', 'grupo', 'embalaje', 'transporte']
        }
        
        best_section = 3  # Por defecto, composición
        max_matches = 0
        
        for section_num, indicators in section_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in table_text)
            if matches > max_matches:
                max_matches = matches
                best_section = section_num
        
        return best_section

    def save_chunks_to_json(self, chunks: List[Chunk], output_path: str):
        """Guarda los chunks en formato JSON para uso posterior."""
        chunks_data = []
        
        for chunk in chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': {
                    'document_id': chunk.metadata.document_id,
                    'section_number': chunk.metadata.section_number,
                    'section_name': chunk.metadata.section_name,
                    'chunk_sequence': chunk.metadata.chunk_sequence,
                    'content_type': chunk.metadata.content_type,
                    'chemical_entities': chunk.metadata.chemical_entities,
                    'cas_numbers': chunk.metadata.cas_numbers,
                    'hazard_codes': chunk.metadata.hazard_codes,
                    'exposure_limits': chunk.metadata.exposure_limits,
                    'token_count': chunk.metadata.token_count,
                    'overlap_with_previous': chunk.metadata.overlap_with_previous
                },
                'keywords': chunk.keywords
            }
            chunks_data.append(chunk_dict)
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Chunks guardados en: {output_path}")

    def process_pdf_file(self, file_path: str):
        """Método actualizado que mantiene funcionalidad existente y agrega chunking."""
        self.stats['total_processed'] += 1
        start_time = datetime.now()
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        print(f"📄 Procesando: {file_name}")

        try:
            with fitz.open(file_path) as doc:
                self.stats['total_pages'] += doc.page_count

                # Extracción existente (mantener funcionalidad)
                full_text = self._get_all_pages_text(doc)
                sections = self._extract_sections_with_regex(full_text)
                
                if not sections or len(sections) < 16:
                    print(f"❌ Error procesando (no se encontraron las 16 secciones): {file_name}")
                    self.stats['failed_extractions'] += 1
                    return False
                
                print("  📊 Extrayendo tablas con múltiples métodos...")
                tables = self.extract_tables(doc)
                
                print("  🖼️ Extrayendo imágenes...")
                base_name = file_path_obj.stem
                images = self._extract_images_from_pdf(doc, base_name, self.config.extracted_subfolders['images'])
                self.stats['total_images'] += len(images)

                # === NUEVA FUNCIONALIDAD: CHUNKING INTELIGENTE ===
                print("  🧩 Generando chunks inteligentes...")
                chunks = self.generate_chunks_for_doc(file_path)
                
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                
                # Guardar chunks en archivo JSON
                chunks_path = self.config.get_folder('processed_chunks') / f"{base_name}_{timestamp}_chunks.json"
                self.save_chunks_to_json(chunks, str(chunks_path))

                # Funcionalidad existente (mantener)
                metadata_path = self.config.extracted_subfolders['metadata'] / f"{base_name}_{timestamp}_metadata.json"
                metadata = {
                    'filename': file_name,
                    'timestamp': timestamp,
                    'sections_found': sorted(sections.keys()),
                    'tables_extracted': len(tables),
                    'images_extracted': len(images),
                    'chunks_generated': len(chunks),
                    'priority_chunks': len([c for c in chunks if c.metadata.section_number in self.chunker.priority_sections])
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4, ensure_ascii=False)
                print(f"    ✅ Metadatos guardados en {metadata_path}")

                texts_path = self.config.extracted_subfolders['texts'] / f"{base_name}_sections.txt"
                self._save_sections(str(texts_path), sections)
                print(f"    ✅ Contenido de secciones guardado en {texts_path}")

                if tables:
                    tables_folder = self.config.extracted_subfolders['tables']
                    self._save_tables(tables, base_name, tables_folder)
                    print(f"    ✅ Tablas guardadas en {tables_folder}")
                
                if images:
                    print(f"    ✅ Imágenes guardadas en {self.config.extracted_subfolders['images']}")

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                self.stats['successful_extractions'] += 1
                print(f"✅ Procesado exitosamente: {file_name} ({duration:.2f}s)")
                
                return {
                    'file': file_name, 
                    'sections': sections, 
                    'tables': tables, 
                    'images': images, 
                    'chunks': chunks
                }

        except Exception as e:
            print(f"❌ Error general procesando: {file_name}. Detalles: {e}")
            self.stats['failed_extractions'] += 1
            return False


def _clean_output_folders(config: ProjectConfig):
    """Elimina todo el contenido de las carpetas de salida antes de una nueva ejecución."""
    print("🧹 Limpiando carpetas de salida...")
    folders_to_clean = [
        config.extracted_subfolders['texts'],
        config.extracted_subfolders['tables'],
        config.extracted_subfolders['images'],
        config.extracted_subfolders['metadata'],
        config.get_folder('processed_chunks')
    ]
    for folder in folders_to_clean:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    print("    ✅ Limpieza completada.")


def extract_content_with_chunking(raw_data_folder: str, base_project_path: str):
    """
    Función principal mejorada para extracción de contenido con chunking inteligente.
    """
    config = ProjectConfig(base_project_path)
    config.create_folders()
    extractor = FDSExtractorAdvanced(config)

    # Limpiar carpetas antes de cada ejecución
    _clean_output_folders(config)

    raw_data_path = Path(raw_data_folder)
    if not raw_data_path.is_dir():
        print(f"❌ La carpeta de datos crudos no existe: {raw_data_folder}")
        return []

    pdf_files = list(raw_data_path.glob('*.pdf'))
    if not pdf_files:
        print("🔍 No se encontraron archivos PDF en la carpeta. Finalizando.")
        return []

    print(f"📁 Encontrados {len(pdf_files)} archivos PDF para procesar")
    
    all_extractions = []
    all_chunks = []
    
    for pdf_file in pdf_files:
        result = extractor.process_pdf_file(str(pdf_file))
        if result:
            all_extractions.append(result)
            if 'chunks' in result:
                all_chunks.extend(result['chunks'])

    # Generar reporte consolidado
    generate_processing_report(extractor, all_chunks, config)
    
    print("\n" + "="*60)
    print("✅ RESUMEN DEL PROCESO DE EXTRACCIÓN CON CHUNKING")
    print("="*60)
    print(f"   📄 Total de archivos procesados: {extractor.stats['total_processed']}")
    print(f"   ✅ Extracciones exitosas: {extractor.stats['successful_extractions']}")
    print(f"   ❌ Extracciones fallidas: {extractor.stats['failed_extractions']}")
    print(f"   📃 Páginas totales analizadas: {extractor.stats['total_pages']}")
    print(f"   📊 Tablas totales extraídas: {extractor.stats['total_tables']}")
    print(f"   🖼️ Imágenes totales extraídas: {extractor.stats['total_images']}")
    print(f"   🧩 Chunks totales generados: {extractor.stats['total_chunks']}")
    print(f"   🎯 Chunks de secciones prioritarias: {extractor.stats['priority_section_chunks']}")
    
    # Estadísticas de chunks
    if all_chunks:
        avg_tokens = sum(chunk.metadata.token_count for chunk in all_chunks) / len(all_chunks)
        text_chunks = len([c for c in all_chunks if c.metadata.content_type == 'text'])
        table_chunks = len([c for c in all_chunks if c.metadata.content_type == 'table'])
        
        print(f"   📈 Promedio de tokens por chunk: {avg_tokens:.1f}")
        print(f"   📝 Chunks de texto: {text_chunks}")
        print(f"   📊 Chunks de tablas: {table_chunks}")
    
    return all_extractions


def generate_processing_report(extractor: FDSExtractorAdvanced, all_chunks: List[Chunk], config: ProjectConfig):
    """Genera un reporte detallado del procesamiento."""
    
    report_data = {
        'processing_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_files': extractor.stats['total_processed'],
            'successful_extractions': extractor.stats['successful_extractions'],
            'failed_extractions': extractor.stats['failed_extractions'],
            'total_pages': extractor.stats['total_pages'],
            'total_tables': extractor.stats['total_tables'],
            'total_images': extractor.stats['total_images'],
            'total_chunks': len(all_chunks)
        },
        'chunk_analytics': {},
        'chemical_entities_summary': {},
        'section_distribution': {}
    }
    
    if all_chunks:
        # Analíticas de chunks
        token_counts = [chunk.metadata.token_count for chunk in all_chunks]
        content_types = {}
        sections_distribution = {}
        
        # Entidades químicas consolidadas
        all_cas_numbers = set()
        all_hazard_codes = set()
        all_chemical_entities = set()
        
        for chunk in all_chunks:
            # Distribución por tipo de contenido
            content_type = chunk.metadata.content_type
            content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Distribución por sección
            section_num = chunk.metadata.section_number
            sections_distribution[section_num] = sections_distribution.get(section_num, 0) + 1
            
            # Entidades químicas
            all_cas_numbers.update(chunk.metadata.cas_numbers)
            all_hazard_codes.update(chunk.metadata.hazard_codes)
            all_chemical_entities.update(chunk.metadata.chemical_entities)
        
        report_data['chunk_analytics'] = {
            'total_chunks': len(all_chunks),
            'avg_token_count': sum(token_counts) / len(token_counts),
            'min_token_count': min(token_counts),
            'max_token_count': max(token_counts),
            'content_type_distribution': content_types,
            'chunks_with_overlap': sum(1 for chunk in all_chunks if chunk.metadata.overlap_with_previous)
        }
        
        report_data['chemical_entities_summary'] = {
            'unique_cas_numbers': len(all_cas_numbers),
            'unique_hazard_codes': len(all_hazard_codes),
            'unique_chemical_entities': len(all_chemical_entities),
            'cas_numbers_sample': list(all_cas_numbers)[:10],
            'hazard_codes_sample': list(all_hazard_codes)[:10]
        }
        
        report_data['section_distribution'] = sections_distribution
    
    # Guardar reporte
    report_path = config.get_folder('outputs') / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Reporte de procesamiento guardado en: {report_path}")


def extract_content_only(raw_data_folder: str, base_project_path: str):
    """
    Función mantenida para compatibilidad - extracción básica sin chunking.
    """
    config = ProjectConfig(base_project_path)
    config.create_folders()
    extractor = FDSExtractorAdvanced(config)

    # Limpia las carpetas antes de cada ejecución
    _clean_output_folders(config)

    raw_data_path = Path(raw_data_folder)
    if not raw_data_path.is_dir():
        print(f"❌ La carpeta de datos crudos no existe: {raw_data_folder}")
        return []

    pdf_files = list(raw_data_path.glob('*.pdf'))
    if not pdf_files:
        print("🔍 No se encontraron archivos PDF en la carpeta. Finalizando.")
        return []

    all_extractions = []
    for pdf_file in pdf_files:
        result = extractor.process_pdf_file(str(pdf_file))
        if result:
            all_extractions.append(result)

    print("\n✅ Resumen del proceso de extracción:")
    print(f"   • Total de archivos procesados: {extractor.stats['total_processed']}")
    print(f"   • Extracciones exitosas: {extractor.stats['successful_extractions']}")
    print(f"   • Extracciones fallidas: {extractor.stats['failed_extractions']}")
    print(f"   • Páginas totales analizadas: {extractor.stats['total_pages']}")
    print(f"   • Tablas totales extraídas: {extractor.stats['total_tables']}")
    print(f"   • Imágenes totales extraídas: {extractor.stats['total_images']}")
    return all_extractions


# ============================================================================
# FUNCIONES DE UTILIDAD PARA ANÁLISIS DE CHUNKS
# ============================================================================

def analyze_chunks_quality(chunks: List[Chunk]) -> Dict:
    """Analiza la calidad y distribución de los chunks generados."""
    
    if not chunks:
        return {"error": "No hay chunks para analizar"}
    
    analysis = {
        'total_chunks': len(chunks),
        'token_distribution': {
            'min': min(chunk.metadata.token_count for chunk in chunks),
            'max': max(chunk.metadata.token_count for chunk in chunks),
            'avg': sum(chunk.metadata.token_count for chunk in chunks) / len(chunks)
        },
        'content_type_breakdown': {},
        'section_coverage': {},
        'chemical_entities_stats': {
            'chunks_with_cas': 0,
            'chunks_with_hazard_codes': 0,
            'total_unique_cas': set(),
            'total_unique_hazards': set()
        }
    }
    
    # Analizar distribución por tipo de contenido
    for chunk in chunks:
        content_type = chunk.metadata.content_type
        analysis['content_type_breakdown'][content_type] = analysis['content_type_breakdown'].get(content_type, 0) + 1
        
        # Cobertura de secciones
        section = chunk.metadata.section_number
        analysis['section_coverage'][section] = analysis['section_coverage'].get(section, 0) + 1
        
        # Estadísticas de entidades químicas
        if chunk.metadata.cas_numbers:
            analysis['chemical_entities_stats']['chunks_with_cas'] += 1
            analysis['chemical_entities_stats']['total_unique_cas'].update(chunk.metadata.cas_numbers)
        
        if chunk.metadata.hazard_codes:
            analysis['chemical_entities_stats']['chunks_with_hazard_codes'] += 1
            analysis['chemical_entities_stats']['total_unique_hazards'].update(chunk.metadata.hazard_codes)
    
    # Convertir sets a listas para serialización JSON
    analysis['chemical_entities_stats']['total_unique_cas'] = list(analysis['chemical_entities_stats']['total_unique_cas'])
    analysis['chemical_entities_stats']['total_unique_hazards'] = list(analysis['chemical_entities_stats']['total_unique_hazards'])
    
    return analysis


def filter_chunks_by_criteria(chunks: List[Chunk], 
                             section_numbers: List[int] = None,
                             content_type: str = None,
                             min_tokens: int = None,
                             has_chemical_entities: bool = None) -> List[Chunk]:
    """Filtra chunks basado en diferentes criterios."""
    
    filtered_chunks = chunks
    
    if section_numbers:
        filtered_chunks = [c for c in filtered_chunks if c.metadata.section_number in section_numbers]
    
    if content_type:
        filtered_chunks = [c for c in filtered_chunks if c.metadata.content_type == content_type]
    
    if min_tokens:
        filtered_chunks = [c for c in filtered_chunks if c.metadata.token_count >= min_tokens]
    
    if has_chemical_entities is not None:
        if has_chemical_entities:
            filtered_chunks = [c for c in filtered_chunks 
                             if c.metadata.cas_numbers or c.metadata.chemical_entities or c.metadata.hazard_codes]
        else:
            filtered_chunks = [c for c in filtered_chunks 
                             if not (c.metadata.cas_numbers or c.metadata.chemical_entities or c.metadata.hazard_codes)]
    
    return filtered_chunks


def load_chunks_from_json(json_path: str) -> List[Chunk]:
    """Carga chunks desde un archivo JSON."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for chunk_data in chunks_data:
            metadata = ChunkMetadata(**chunk_data['metadata'])
            chunk = Chunk(
                chunk_id=chunk_data['chunk_id'],
                content=chunk_data['content'],
                metadata=metadata,
                keywords=chunk_data['keywords']
            )
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        print(f"❌ Error cargando chunks desde {json_path}: {e}")
        return []


def merge_chunks_from_multiple_files(chunks_folder: str) -> List[Chunk]:
    """Carga y combina chunks de múltiples archivos JSON."""
    chunks_path = Path(chunks_folder)
    all_chunks = []
    
    if not chunks_path.exists():
        print(f"❌ La carpeta {chunks_folder} no existe")
        return []
    
    json_files = list(chunks_path.glob('*_chunks.json'))
    if not json_files:
        print(f"❌ No se encontraron archivos de chunks en {chunks_folder}")
        return []
    
    print(f"🔄 Cargando chunks de {len(json_files)} archivos...")
    
    for json_file in json_files:
        file_chunks = load_chunks_from_json(str(json_file))
        all_chunks.extend(file_chunks)
        print(f"  ✅ Cargados {len(file_chunks)} chunks de {json_file.name}")
    
    print(f"📊 Total de chunks cargados: {len(all_chunks)}")
    return all_chunks


def export_chunks_to_csv(chunks: List[Chunk], output_path: str):
    """Exporta chunks a formato CSV para análisis externo."""
    if not chunks:
        print("❌ No hay chunks para exportar")
        return
    
    data = []
    for chunk in chunks:
        row = {
            'chunk_id': chunk.chunk_id,
            'document_id': chunk.metadata.document_id,
            'section_number': chunk.metadata.section_number,
            'section_name': chunk.metadata.section_name,
            'content_type': chunk.metadata.content_type,
            'token_count': chunk.metadata.token_count,
            'cas_numbers': '; '.join(chunk.metadata.cas_numbers),
            'hazard_codes': '; '.join(chunk.metadata.hazard_codes),
            'keywords': '; '.join(chunk.keywords),
            'content_preview': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"📊 Chunks exportados a CSV: {output_path}")


# ============================================================================
# FUNCIONES ADICIONALES PARA OPTIMIZACIÓN RAG
# ============================================================================

def create_chunk_embeddings_metadata(chunks: List[Chunk]) -> List[Dict]:
    """Prepara metadatos optimizados para sistemas de embeddings."""
    embeddings_metadata = []
    
    for chunk in chunks:
        # Crear texto enriquecido para embedding
        enriched_text = chunk.content
        
        # Añadir contexto de sección
        section_context = f"Sección {chunk.metadata.section_number}: {chunk.metadata.section_name}"
        
        # Añadir entidades químicas al contexto
        entities_context = ""
        if chunk.metadata.cas_numbers:
            entities_context += f" CAS: {', '.join(chunk.metadata.cas_numbers)}"
        if chunk.metadata.hazard_codes:
            entities_context += f" Códigos H/P: {', '.join(chunk.metadata.hazard_codes)}"
        
        # Metadatos para el vector store
        metadata = {
            'chunk_id': chunk.chunk_id,
            'document_id': chunk.metadata.document_id,
            'section_number': chunk.metadata.section_number,
            'section_name': chunk.metadata.section_name,
            'content_type': chunk.metadata.content_type,
            'token_count': chunk.metadata.token_count,
            'is_priority_section': chunk.metadata.section_number in {2, 3, 8, 11},
            'has_chemical_entities': bool(chunk.metadata.cas_numbers or chunk.metadata.chemical_entities),
            'enriched_text': f"{section_context}. {enriched_text}{entities_context}",
            'keywords': chunk.keywords,
            'cas_numbers': chunk.metadata.cas_numbers,
            'hazard_codes': chunk.metadata.hazard_codes
        }
        
        embeddings_metadata.append({
            'text': metadata['enriched_text'],
            'metadata': metadata
        })
    
    return embeddings_metadata


def create_retrieval_filters(query_type: str = "general") -> Dict:
    """Crea filtros predefinidos para diferentes tipos de consultas RAG."""
    filters = {
        'general': {},
        'safety': {
            'section_number': {'$in': [2, 4, 5, 6, 8]}  # Secciones de seguridad
        },
        'composition': {
            'section_number': 3,
            'has_chemical_entities': True
        },
        'toxicology': {
            'section_number': {'$in': [11, 12]}  # Toxicología y ecología
        },
        'handling': {
            'section_number': {'$in': [7, 8]}  # Manejo y protección
        },
        'regulatory': {
            'section_number': {'$in': [14, 15]}  # Transporte y regulatorio
        },
        'priority': {
            'is_priority_section': True
        }
    }
    
    return filters.get(query_type, {})


# ============================================================================
# FUNCIÓN PRINCIPAL PARA TESTING Y USO
# ============================================================================

if __name__ == "__main__":
    # Configuración por defecto
    RAW_DATA_FOLDER = r"/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad/data/raw_documents"
    BASE_PROJECT_PATH = r"/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad"
    
    print("🚀 Iniciando extracción con chunking inteligente...")
    print("="*60)
    
    # Ejecutar extracción con chunking
    extracted_documents = extract_content_with_chunking(RAW_DATA_FOLDER, BASE_PROJECT_PATH)
    
    print(f"\n🎉 Procesamiento completado!")
    print(f"📁 Se procesaron {len(extracted_documents)} documentos exitosamente.")
    print(f"📂 Los chunks se guardaron en: {BASE_PROJECT_PATH}/data/processed_chunks/")
    print(f"📊 El reporte detallado está en: {BASE_PROJECT_PATH}/outputs/")
    
    # Ejemplo de análisis de chunks (opcional)
    if extracted_documents:
        print("\n" + "="*40)
        print("📈 ANÁLISIS OPCIONAL DE CHUNKS")
        print("="*40)
        
        # Recopilar todos los chunks
        all_chunks = []
        for doc_result in extracted_documents:
            if 'chunks' in doc_result:
                all_chunks.extend(doc_result['chunks'])
        
        if all_chunks:
            # Análisis de calidad
            quality_analysis = analyze_chunks_quality(all_chunks)
            print(f"📊 Análisis de calidad completado:")
            print(f"   - Rango de tokens: {quality_analysis['token_distribution']['min']}-{quality_analysis['token_distribution']['max']}")
            print(f"   - Promedio de tokens: {quality_analysis['token_distribution']['avg']:.1f}")
            
            # Filtrar chunks prioritarios
            priority_chunks = filter_chunks_by_criteria(
                all_chunks, 
                section_numbers=[2, 3, 8, 11],  # Secciones prioritarias
                min_tokens=50
            )
            print(f"   - Chunks prioritarios (secciones 2,3,8,11): {len(priority_chunks)}")
            
            # Chunks con entidades químicas
            chemical_chunks = filter_chunks_by_criteria(
                all_chunks,
                has_chemical_entities=True
            )
            print(f"   - Chunks con entidades químicas: {len(chemical_chunks)}")
            
            # Exportar análisis a CSV (opcional)
            csv_path = Path(BASE_PROJECT_PATH) / "outputs" / f"chunks_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_chunks_to_csv(all_chunks, str(csv_path))
            
            # Preparar metadatos para RAG
            embeddings_metadata = create_chunk_embeddings_metadata(all_chunks)
            print(f"   - Metadatos para embeddings preparados: {len(embeddings_metadata)} elementos")
    
    print("\n✨ ¡Proceso completado exitosamente!")
    print("Los chunks están listos para ser utilizados en tu sistema RAG.")
    print("\nPróximos pasos sugeridos:")
    print("1. Revisar el reporte de procesamiento en /outputs/")
    print("2. Generar embeddings usando los chunks procesados")
    print("3. Configurar tu vector database con los metadatos enriquecidos")
    print("4. Implementar filtros de retrieval según tipo de consulta")