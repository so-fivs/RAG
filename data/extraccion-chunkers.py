import os
import re
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import shutil
from datetime import datetime
import tabula
import camelot

# PDF Processing
import fitz  # PyMuPDF
import pdfplumber

print("✅ Dependencias básicas instaladas correctamente")

# ============================================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================================

class ProjectConfig:
    """Configuración centralizada del proyecto"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Estructura de carpetas
        self.folders = {
            'raw_data': self.base_path / 'data' / 'raw_documents',
            'extracted_content': self.base_path / 'data' / 'extracted_content',
            'processed_chunks': self.base_path / 'data' / 'processed_chunks',
            'embeddings': self.base_path / 'data' / 'embeddings',
            'vector_db': self.base_path / 'data' / 'vector_db',
            'logs': self.base_path / 'logs',
            'outputs': self.base_path / 'outputs'
        }
        
        # Subcarpetas para contenido extraído
        self.extracted_subfolders = {
            'texts': self.folders['extracted_content'] / 'texts',
            'tables': self.folders['extracted_content'] / 'tables',
            'images': self.folders['extracted_content'] / 'images',
            'metadata': self.folders['extracted_content'] / 'metadata'
        }
        
        # Crear estructura de carpetas
        self.create_folder_structure()
    
    def create_folder_structure(self):
        """Crea la estructura de carpetas del proyecto"""
        print("📁 Creando estructura de carpetas...")
        
        # Carpetas principales
        for name, path in self.folders.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ {name}: {path}")
        
        # Subcarpetas de contenido extraído
        for name, path in self.extracted_subfolders.items():
            path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ extracted/{name}: {path}")
        
        print("✅ Estructura de carpetas creada")
    
    def get_path(self, folder_type: str, subfolder: str = None) -> Path:
        """Obtiene la ruta de una carpeta específica"""
        if subfolder:
            return self.extracted_subfolders.get(subfolder, self.folders[folder_type])
        return self.folders.get(folder_type, self.base_path)

# ============================================================================
# FDS EXTRACTOR MEJORADO
# ============================================================================

class FDSExtractorAdvanced:
    """Extractor completo para Fichas de Datos de Seguridad"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        
        # NOMBRES ESTÁNDAR DE SECCIONES FDS
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
        
        # PALABRAS CLAVE MEJORADAS Y AMPLIADAS PARA SECCIONES FDS
        self.fds_sections = {
            1: [
                # Español - Identificación
                r"identificaci[oó]n(?:\s+del?\s+producto)?", r"identificacion(?:\s+del?\s+producto)?", 
                r"identif(?:icador)?", r"nombre.*producto", r"nombre.*comercial", r"denominaci[oó]n",
                r"datos.*producto", r"informaci[oó]n.*producto", r"producto.*qu[ií]mico",
                r"sustancia.*qu[ií]mica", r"datos.*identificaci[oó]n", r"código.*producto",
                r"numero.*identificaci[oó]n", r"product\s+identifier", r"product\s+name",
                # Inglés - Identification
                r"identification", r"product.*identification", r"product.*name",
                r"trade.*name", r"product.*information", r"chemical.*product",
                r"product.*code", r"manufacturer", r"supplier", r"fabricante", r"proveedor",
                # Variantes adicionales
                r"ficha.*identificaci[oó]n", r"datos.*empresa", r"company.*information",
                r"emergency.*contact", r"contacto.*emergencia", r"recommended.*use", r"uso.*recomendado"
            ],
            
            2: [
                # Español - Peligros
                r"identificaci[oó]n.*peligros", r"peligros", r"riesgos?", r"hazard.*identification",
                r"clasificaci[oó]n.*peligro", r"advertencias.*peligro", r"informaci[oó]n.*peligros",
                r"elementos.*etiqueta", r"pictogramas?", r"signal.*word", r"palabra.*advertencia",
                # Inglés - Hazards
                r"hazard.*identification", r"hazards?", r"risks?", r"dangers?", r"risk.*assessment",
                r"hazard.*classification", r"safety.*hazards", r"warnings?", r"caution",
                # GHS/SGA específico
                r"pictograms?", r"etiquetado", r"labeling", r"labelling", r"ghs", r"sga",
                r"frases.*[hp]", r"[hp].*phrases", r"frases.*precauci[oó]n", r"precautionary.*statements",
                r"signal.*word", r"indicaciones.*peligro", r"hazard.*statements",
                # Clasificación específica
                r"clasificaci[oó]n", r"classification", r"categor[ií]a.*peligro", r"hazard.*category",
                r"otros.*peligros", r"other.*hazards", r"hnoc", r"not.*otherwise.*classified"
            ],
            
            3: [
                # Español - Composición
                r"composici[oó]n", r"ingredientes", r"componentes", r"contenido",
                r"informaci[oó]n.*composici[oó]n", r"sustancias.*peligrosas", r"mezcla",
                r"preparado", r"formulaci[oó]n", r"constituci[oó]n", r"contenido.*qu[ií]mico",
                # Inglés - Composition
                r"composition", r"ingredients", r"components", r"contents", r"mixture",
                r"information.*composition", r"hazardous.*ingredients", r"chemical.*composition",
                r"formulation", r"constitution", r"chemical.*identity",
                # Técnico - Datos químicos
                r"cas.*number", r"n[uú]mero.*cas", r"concentraci[oó]n", r"porcentaje",
                r"percentage", r"% peso", r"% weight", r"formula", r"f[oó]rmula",
                r"einecs", r"ec.*number", r"registro", r"registration", r"reach",
                # Rangos y límites
                r"wt\.?%", r"w/w", r"peso/peso", r"range", r"rango", r"limits?", r"l[ií]mites?"
            ],
            
            4: [
                # Español - Primeros Auxilios
                r"primeros\s+auxilios", r"medidas.*primeros.*auxilios", r"primeros.*auxilios",
                r"atenci[oó]n.*m[eé]dica", r"tratamiento.*urgencia", r"medidas.*emergencia",
                r"descripci[oó]n.*primeros.*auxilios", r"socorro", r"auxilio.*inmediato",
                # Inglés - First Aid
                r"first\s+aid", r"first.*aid.*measures", r"emergency.*measures", r"medical.*attention",
                r"emergency.*treatment", r"immediate.*care", r"emergency.*care",
                # Vías de exposición específicas
                r"inhalaci[oó]n", r"ingesti[oó]n", r"contacto.*piel", r"contacto.*ojos",
                r"inhalation", r"ingestion", r"skin.*contact", r"eye.*contact",
                r"dermal.*contact", r"oral", r"respiratorio", r"respiratory",
                # Síntomas y efectos
                r"s[ií]ntomas", r"symptoms", r"efectos", r"effects", r"reacciones.*adversas",
                r"adverse.*reactions", r"signos", r"signs", r"manifestaciones"
            ],
            
            5: [
                # Español - Incendios
                r"medidas.*incendio", r"lucha.*incendio", r"extinción", r"extincion",
                r"medidas.*extinción", r"fuego", r"combustión", r"combustion",
                r"medidas.*contra.*incendio", r"protección.*incendio", r"combate.*incendio",
                # Inglés - Fire
                r"fire.*fighting", r"firefighting", r"fire.*extinguishing", r"fire.*protection",
                r"fire.*hazards", r"extinguishing.*measures", r"fire.*suppression",
                # Medios de extinción específicos
                r"extintores?", r"extinguishers?", r"medios.*extinción", r"extinction.*media",
                r"agentes.*extintores", r"extinguishing.*agents", r"suppression.*agents",
                r"agua", r"water", r"espuma", r"foam", r"co2", r"polvo.*qu[ií]mico",
                r"dry.*chemical", r"polvo.*seco", r"arena", r"sand",
                # Peligros específicos
                r"peligros.*espec[ií]ficos", r"specific.*hazards", r"productos.*combustión",
                r"combustion.*products", r"humos.*tóxicos", r"toxic.*fumes"
            ],
            
            6: [
                # Español - Derrames/Liberación
                r"medidas.*derrame", r"liberación.*accidental", r"liberacion.*accidental",
                r"vertido", r"fuga", r"escape", r"derrame", r"spill", r"medidas.*vertido", 
                r"contaminación", r"medidas.*contención", r"limpieza", r"neutralización", 
                r"neutralizacion", r"medidas.*caso.*vertido",
                # Inglés - Spills/Release
                r"accidental.*release", r"spill", r"leak", r"spillage", r"leakage",
                r"environmental.*precautions", r"containment", r"cleanup", r"clean.*up",
                r"neutralization", r"absorption", r"release.*measures",
                # Métodos y procedimientos
                r"absorber", r"contener", r"neutralizar", r"absorb", r"contain", r"neutralize",
                r"recoger", r"collect", r"sweep", r"barrer", r"aspirar", r"vacuum",
                r"precauciones.*ambientales", r"environmental.*precautions"
            ],
            
            7: [
                # Español - Manejo y Almacenamiento
                r"manejo.*almacenamiento", r"manejo", r"almacenamiento", r"manipulaci[oó]n",
                r"manipulacion", r"almacén", r"almacenaje", r"conservación", r"conservacion",
                r"handling.*storage", r"precauciones.*manipulación", r"precauciones.*manejo",
                # Inglés - Handling and Storage
                r"handling.*storage", r"handling", r"storage", r"manipulation", r"warehouse",
                r"conservation", r"custody", r"safe.*handling", r"safe.*storage",
                # Condiciones específicas
                r"condiciones.*almacen", r"storage.*conditions", r"temperatura", r"temperature",
                r"humedad", r"humidity", r"incompatibles", r"incompatible", r"segregaci[oó]n",
                r"segregation", r"ventilaci[oó]n", r"ventilation", r"iluminaci[oó]n",
                # Precauciones
                r"precauciones.*segura", r"safe.*precautions", r"medidas.*seguridad",
                r"safety.*measures", r"buenas.*prácticas", r"good.*practices"
            ],
            
            8: [
                # Español - Protección y Exposición
                r"controles.*exposici[oó]n", r"controles.*exposicion", r"protecci[oó]n.*personal",
                r"proteccion.*personal", r"epp", r"equipos.*protecci[oó]n", r"equipos.*proteccion",
                r"medidas.*protecci[oó]n", r"l[ií]mites.*exposici[oó]n", r"valores.*l[ií]mite",
                # Inglés - Exposure Controls/Protection
                r"exposure.*controls", r"personal.*protection", r"ppe", r"protection.*equipment",
                r"exposure.*limits", r"limit.*values", r"occupational.*exposure",
                r"workplace.*exposure", r"personal.*protective.*equipment",
                # Límites específicos
                r"tlv", r"pel", r"stel", r"twa", r"ceiling", r"techo", r"acgih", r"osha",
                r"niosh", r"límite.*permisible", r"permissible.*limit",
                # EPP específico
                r"respirador", r"guantes", r"gafas", r"ropa.*protecci[oó]n", r"respirator",
                r"gloves", r"goggles", r"protective.*clothing", r"safety.*glasses",
                r"face.*shield", r"pantalla.*facial", r"botas", r"boots"
            ],
            
            9: [
                # Español - Propiedades Físicas y Químicas
                r"propiedades.*f[ií]sicas", r"propiedades.*qu[ií]micas", r"propiedades.*fisicas",
                r"propiedades.*quimicas", r"caracter[ií]sticas.*f[ií]sicas", r"datos.*f[ií]sicos",
                r"aspectos?", r"apariencia", r"color", r"olor", r"aspecto.*f[ií]sico",
                # Inglés - Physical and Chemical Properties
                r"physical.*chemical.*properties", r"physical.*properties", r"chemical.*properties",
                r"appearance", r"color", r"colour", r"odor", r"odour", r"physical.*characteristics",
                # Propiedades específicas
                r"densidad", r"viscosidad", r"solubilidad", r"ph", r"punto.*ebullici[oó]n",
                r"punto.*fusi[oó]n", r"density", r"viscosity", r"solubility", r"boiling.*point",
                r"melting.*point", r"flash.*point", r"vapor.*pressure", r"vapour.*pressure",
                r"punto.*inflamaci[oó]n", r"presi[oó]n.*vapor", r"estado.*f[ií]sico",
                r"physical.*state", r"forma", r"form", r"molecular.*weight", r"peso.*molecular"
            ],
            
            10: [
                # Español - Estabilidad y Reactividad
                r"estabilidad.*reactividad", r"estabilidad", r"reactividad", r"incompatibilidad",
                r"descomposici[oó]n", r"descomposicion", r"polimerizaci[oó]n", r"polimerizacion",
                r"condiciones.*evitar", r"materiales.*evitar", r"estabilidad.*qu[ií]mica",
                # Inglés - Stability and Reactivity
                r"stability.*reactivity", r"stability", r"reactivity", r"incompatibility",
                r"decomposition", r"polymerization", r"polymerisation", r"conditions.*avoid",
                r"materials.*avoid", r"chemical.*stability", r"thermal.*stability",
                # Reacciones específicas
                r"productos.*descomposici[oó]n", r"decomposition.*products", r"reacciones.*peligrosas",
                r"hazardous.*reactions", r"condiciones.*inestabilidad", r"instability.*conditions",
                r"catalizadores", r"catalysts", r"inhibidores", r"inhibitors",
                # Condiciones de evitar
                r"calor", r"heat", r"luz", r"light", r"humedad", r"moisture",
                r"oxidantes", r"oxidizers", r"ácidos", r"acids", r"bases", r"álcalis"
            ],
            
            11: [
                # Español - Toxicología
                r"informaci[oó]n.*toxicol[oó]gica", r"informacion.*toxicologica", r"toxicolog[ií]a",
                r"toxicologia", r"efectos.*salud", r"toxicidad", r"efectos.*adversos",
                r"carcinogenicidad", r"mutagenicidad", r"teratogenicidad", r"efectos.*tóxicos",
                # Inglés - Toxicology
                r"toxicological.*information", r"toxicology", r"health.*effects", r"toxicity",
                r"adverse.*effects", r"carcinogenicity", r"mutagenicity", r"teratogenicity",
                r"reproductive.*toxicity", r"health.*hazards", r"toxic.*effects",
                # Datos toxicológicos específicos
                r"dosis.*letal", r"dl50", r"ld50", r"cl50", r"lc50", r"dose.*letale",
                r"lethal.*dose", r"lethal.*concentration", r"concentraci[oó]n.*letal",
                r"irritaci[oó]n", r"irritacion", r"sensibilizaci[oó]n", r"sensibilizacion", 
                r"irritation", r"sensitization", r"corrosión", r"corrosion",
                # Efectos específicos
                r"effects.*overexposure", r"efectos.*sobreexposici[oó]n", r"acute.*toxicity",
                r"toxicidad.*aguda", r"chronic.*effects", r"efectos.*cr[oó]nicos",
                r"routes.*exposure", r"v[ií]as.*exposici[oó]n", r"absorption", r"absorci[oó]n"
            ],
            
            12: [
                # Español - Ecología
                r"informaci[oó]n.*ecol[oó]gica", r"informacion.*ecologica", r"ecolog[ií]a",
                r"ecologia", r"ecotoxicidad", r"efectos.*ambientales", r"medio.*ambiente",
                r"biodegradabilidad", r"bioacumulaci[oó]n", r"bioacumulacion", r"persistencia",
                r"informaci[oó]n.*ambiental", r"datos.*ambientales",
                # Inglés - Ecological Information
                r"ecological.*information", r"ecology", r"ecotoxicity", r"environmental.*effects",
                r"environment", r"biodegradability", r"bioaccumulation", r"persistence",
                r"mobility", r"environmental.*fate", r"ecotoxicological", r"environmental.*data",
                # Organismos específicos
                r"vida.*acuática", r"vida.*acuatica", r"organismos.*acu[aá]ticos", r"aquatic.*life",
                r"aquatic.*organisms", r"degradaci[oó]n", r"degradacion", r"degradation",
                r"peces", r"dafnias", r"algas", r"fish", r"daphnia", r"algae",
                # Factores ambientales
                r"factor.*m", r"m.*factor", r"pbt", r"mpb", r"pmb", r"vmpvb",
                r"bioconcentraci[oó]n", r"bioconcentration", r"movilidad.*suelo", r"soil.*mobility"
            ],
            
            13: [
                # Español - Eliminación
                r"eliminaci[oó]n", r"eliminacion", r"disposici[oó]n", r"disposicion", r"desecho",
                r"residuos", r"tratamiento.*residuos", r"gesti[oó]n.*residuos", r"gestion.*residuos",
                r"destrucci[oó]n", r"destruccion", r"incineración", r"incineracion", 
                r"consideraciones.*eliminaci[oó]n", r"m[eé]todos.*eliminaci[oó]n",
                # Inglés - Disposal
                r"disposal", r"waste.*disposal", r"waste.*treatment", r"waste.*management",
                r"destruction", r"incineration", r"landfill", r"hazardous.*waste",
                r"disposal.*considerations", r"disposal.*methods", r"waste.*handling",
                # Métodos específicos
                r"métodos.*eliminación", r"metodos.*eliminacion", r"disposal.*methods",
                r"tratamiento.*qu[ií]mico", r"tratamiento.*quimico", r"chemical.*treatment",
                r"neutralizaci[oó]n", r"neutralization", r"solidificaci[oó]n", r"solidification",
                # Regulaciones
                r"rcra", r"hazardous.*waste.*number", r"código.*residuo", r"waste.*code",
                r"regulaciones.*eliminaci[oó]n", r"disposal.*regulations"
            ],
            
            14: [
                # Español - Transporte
                r"transporte", r"informaci[oó]n.*transporte", r"informacion.*transporte",
                r"env[ií]o", r"envio", r"expedici[oó]n", r"expedicion", r"embalaje", 
                r"etiquetado.*transporte", r"datos.*transporte", r"clasificaci[oó]n.*transporte",
                # Inglés - Transport
                r"transport", r"transportation", r"shipping", r"transport.*information",
                r"shipping.*information", r"packaging", r"transport.*labeling", 
                r"transport.*labelling", r"shipping.*data",
                # Regulaciones específicas de transporte
                r"onu", r"un", r"imdg", r"iata", r"imo", r"dot", r"adr", r"rid", r"adg",
                r"dangerous.*goods", r"mercanc[ií]as.*peligrosas", r"mercancias.*peligrosas",
                r"clase.*peligro", r"hazard.*class", r"packing.*group", r"grupo.*embalaje",
                # Números y códigos
                r"un.*number", r"número.*onu", r"proper.*shipping.*name", 
                r"nombre.*apropiado.*env[ií]o", r"marine.*pollutant", r"contaminante.*marino"
            ],
            
            15: [
                # Español - Regulaciones
                r"reglamentaci[oó]n", r"reglamentacion", r"regulaci[oó]n", r"regulacion",
                r"legislaci[oó]n", r"legislacion", r"normatividad", r"marco.*legal",
                r"disposiciones.*legales", r"informaci[oó]n.*reglamentaria", 
                r"informaci[oó]n.*regulatoria", r"normativa", r"leyes",
                # Inglés - Regulatory Information
                r"regulatory", r"regulation", r"regulations", r"legislation", r"legal.*framework",
                r"regulatory.*information", r"legal.*provisions", r"compliance",
                r"regulatory.*compliance", r"legal.*requirements",
                # Regulaciones específicas
                r"reach", r"clp", r"osha", r"epa", r"dot", r"sara", r"tsca", r"einecs",
                r"inventarios", r"inventories", r"listas", r"lists", r"registro", r"registration",
                r"cercla", r"rcra", r"clean.*air.*act", r"clean.*water.*act",
                # Proposiciones y advertencias
                r"proposition.*65", r"proposici[oó]n.*65", r"california.*prop", r"prop.*65",
                r"right.*to.*know", r"derecho.*saber", r"worker.*protection"
            ],
            
            16: [
                # Español - Otra Información
                r"otra.*informaci[oó]n", r"otra.*informacion", r"informaci[oó]n.*adicional",
                r"informacion.*adicional", r"datos.*adicionales", r"observaciones", r"notas",
                r"comentarios", r"aclaraciones", r"advertencias", r"informaci[oó]n.*varia",
                r"informaci[oó]n.*complementaria", r"datos.*complementarios",
                # Inglés - Other Information
                r"other.*information", r"additional.*information", r"additional.*data",
                r"observations", r"notes", r"comments", r"clarifications", r"warnings",
                r"miscellaneous.*information", r"supplementary.*information", r"further.*information",
                # Información específica de la ficha
                r"referencias", r"bibliograf[ií]a", r"bibliografia", r"contacto", r"references",
                r"bibliography", r"contact", r"fecha.*revisi[oó]n", r"fecha.*revision",
                r"revision.*date", r"versi[oó]n", r"version", r"prepared.*by",
                r"elaborado.*por", r"fecha.*elaboraci[oó]n", r"preparation.*date",
                # Códigos y ratings
                r"hmis", r"nfpa", r"ratings", r"clasificaci[oó]n.*nfpa", r"códigos",
                r"abbreviations", r"abreviaciones", r"acronyms", r"acr[oó]nimos",
                r"legend", r"leyenda", r"disclaimer", r"exenci[oó]n.*responsabilidad"
            ]
        }
        
        # PALABRAS CLAVE ADICIONALES PARA REASIGNACIÓN AUTOMÁTICA
        self.fallback_keywords = {
            # Palabras que podrían estar en cualquier sección pero dan pistas
            "incendio|fire|flame|combustion|extinguisher": 5,
            "derrame|spill|leak|release|cleanup": 6,
            "guantes|gloves|respirator|protection|ppe|epp": 8,
            "toxicity|toxic|ld50|lc50|carcinogen": 11,
            "environment|ecological|aquatic|biodegradation": 12,
            "disposal|waste|eliminacion|residuo": 13,
            "transport|shipping|un.*number|dangerous.*goods": 14,
            "regulation|osha|epa|reach|sara|tsca": 15,
            "contact|emergency|telefono|phone|revision|version": 16
        }
        
        # Contadores para estadísticas
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_pages': 0,
            'total_images': 0,
            'total_tables': 0
        }
        self.all_sections_found = []
    
    def process_pdf_file(self, pdf_path: str) -> Dict:
        """Procesa un archivo PDF individual y guarda contenido extraído"""
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"❌ Archivo no encontrado: {pdf_path}")
            return None
        
        print(f"\n🔄 Procesando: {pdf_path.name}")
        
        # Generar ID único para el documento
        doc_id = self._generate_doc_id(pdf_path)
        
        # Extraer contenido
        extraction_result = self._extract_from_pdf(str(pdf_path), doc_id)
        
        if extraction_result:
            # Guardar contenido extraído
            self._save_extracted_content(extraction_result, doc_id)
            self.stats['successful_extractions'] += 1
            print(f"✅ Procesado exitosamente: {pdf_path.name}")
        else:
            self.stats['failed_extractions'] += 1
            print(f"❌ Error procesando: {pdf_path.name}")
        
        self.stats['total_processed'] += 1
        return extraction_result
    
    def _extract_from_pdf(self, pdf_path: str, doc_id: str) -> Dict:
        """Extracción completa de PDF"""
        
        result = {
            'document_id': doc_id,
            'source_file': pdf_path,
            'text': '',
            'sections': {},
            'tables': [],
            'images': [],
            'metadata': {},
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        try:
            # === EXTRACCIÓN DE TEXTO CON PyMuPDF ===
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extraer texto
                page_text = page.get_text()
                full_text += f"\n--- PÁGINA {page_num + 1} ---\n{page_text}"
                
                # Extraer imágenes
                self._extract_images_from_page(page, page_num, doc_id, result)
            
            result['text'] = full_text
            result['metadata']['total_pages'] = doc.page_count
            self.stats['total_pages'] += doc.page_count
            
            doc.close()
            
            # === EXTRACCIÓN DE METADATOS ===
            result['metadata'].update(self._extract_metadata(full_text, pdf_path))
            
            # === SEGMENTACIÓN MEJORADA POR SECCIONES ===
            result['sections'] = self._segment_by_sections_advanced(full_text)
            
            # === EXTRACCIÓN AVANZADA DE TABLAS ===
            result['tables'] = self._extract_tables_multi_method(pdf_path, doc_id)
            self.stats['total_tables'] += len(result['tables'])
            
            # Trackear secciones encontradas para estadísticas
            if result['sections']:
                self.all_sections_found.append(list(result['sections'].keys()))
            
            sections_found = list(result['sections'].keys())
            sections_names = [f"{num}: {self.section_names.get(num, 'Desconocida')}" for num in sections_found]
            print(f"   📊 Extraído: {len(result['sections'])} secciones, {len(result['tables'])} tablas, {len(result['images'])} imágenes")
            if sections_found:
                print(f"   📋 Secciones encontradas: {', '.join(map(str, sections_found))}")
            else:
                print(f"   📋 No se identificaron secciones estándar FDS")
                                
        except Exception as e:
            print(f"❌ Error extrayendo {pdf_path}: {e}")
            return None
        
        return result
    
    def _segment_by_sections_advanced(self, text: str) -> Dict:
        """Segmentación avanzada mejorada que garantiza encontrar todas las 16 secciones"""
        sections = {}
        unassigned_content = []
        
        # PASO 1: Segmentación estándar con patrones mejorados
        for section_num, keywords in self.fds_sections.items():
            section_found = False
            
            for keyword in keywords:
                # Patrones más flexibles y robustos
                patterns = [
                    # Formato estándar con "Sección" o "Section"
                    rf"(?:secci[oó]n|section)\s+{section_num}[.\s:-]*{keyword}",
                    
                    # Solo número seguido de punto/espacio + keyword
                    rf"^{section_num}[.\s:-]+{keyword}",
                    rf"^\s*{section_num}[.\s:-]+{keyword}",
                    
                    # Número en paréntesis o corchetes
                    rf"[(\[]{section_num}[)\]]\s*{keyword}",
                    
                    # Número con guión o separadores
                    rf"{section_num}\s*[-–—]\s*{keyword}",
                    
                    # Keywords al inicio de línea (más flexible)
                    rf"^{keyword}.*{section_num}",
                    rf"^{keyword}",
                    
                    # Formato título centrado
                    rf"^\s*{section_num}\.?\s+{keyword}\s*$"
                ]
                
                for pattern in patterns:
                    try:
                        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
                        
                        for match in matches:
                            start_pos = match.start()
                            
                            # Buscar fin de sección (siguiente sección numérica)
                            end_pos = len(text)
                            next_section_pattern = rf"(?:secci[oó]n|section)?\s*(?:\d+)[.\s:-]|^\s*(?:\d+)[.\s:-]"
                            
                            next_matches = list(re.finditer(next_section_pattern, text[start_pos + 50:], 
                                                          re.IGNORECASE | re.MULTILINE))
                            if next_matches:
                                end_pos = start_pos + 50 + next_matches[0].start()
                            
                            section_text = text[start_pos:end_pos].strip()
                            
                            # Solo guardar si tiene contenido suficiente y no existe ya
                            if len(section_text) > 30 and section_num not in sections:
                                sections[section_num] = {
                                    'title': match.group(0),
                                    'content': section_text,
                                    'length': len(section_text),
                                    'pattern_used': pattern,
                                    'keyword_matched': keyword,
                                    'confidence': 'high'
                                }
                                section_found = True
                                break
                        
                        if section_found:
                            break
                            
                    except re.error as e:
                        print(f"⚠️ Error en patrón regex para sección {section_num}: {e}")
                        continue
                
                if section_found:
                    break
        
        # PASO 2: Búsqueda de contenido no asignado para secciones faltantes
        missing_sections = [i for i in range(1, 17) if i not in sections]
        
        if missing_sections:
            print(f"   🔍 Buscando {len(missing_sections)} secciones faltantes: {missing_sections}")
            
            # Dividir texto en bloques para análisis
            text_blocks = self._split_text_into_blocks(text)
            
            for block in text_blocks:
                if not missing_sections:
                    break
                
                # Analizar cada bloque para determinar sección probable
                predicted_section = self._predict_section_from_content(block, missing_sections)
                
                if predicted_section and predicted_section in missing_sections:
                    sections[predicted_section] = {
                        'title': f"Sección {predicted_section} (auto-detectada)",
                        'content': block,
                        'length': len(block),
                        'pattern_used': 'content_analysis',
                        'keyword_matched': 'fallback_analysis',
                        'confidence': 'medium'
                    }
                    missing_sections.remove(predicted_section)
                    print(f"      ✓ Sección {predicted_section} asignada por análisis de contenido")
        
        # PASO 3: Para secciones aún faltantes, crear placeholders con contenido relevante
        final_missing = [i for i in range(1, 17) if i not in sections]
        
        if final_missing:
            print(f"   📝 Creando placeholders para {len(final_missing)} secciones: {final_missing}")
            
            for section_num in final_missing:
                # Buscar cualquier contenido que pueda ser relevante
                relevant_content = self._find_relevant_content_for_section(text, section_num)
                
                sections[section_num] = {
                    'title': f"Sección {section_num} - {self.section_names.get(section_num, 'Sin título')}",
                    'content': relevant_content or f"No se encontró información específica para la sección {section_num}.",
                    'length': len(relevant_content) if relevant_content else 0,
                    'pattern_used': 'placeholder',
                    'keyword_matched': 'not_found',
                    'confidence': 'low'
                }
        
        print(f"   ✅ Segmentación completada: {len(sections)}/16 secciones")
        return sections
    
    def _split_text_into_blocks(self, text: str) -> List[str]:
        """Divide el texto en bloques lógicos para análisis"""
        blocks = []
        
        # Dividir por dobles saltos de línea
        initial_blocks = text.split('\n\n')
        
        for block in initial_blocks:
            block = block.strip()
            if len(block) > 50:  # Solo bloques con contenido suficiente
                # Si el bloque es muy largo, dividirlo más
                if len(block) > 1500:
                    # Dividir por saltos de línea simples
                    sub_blocks = block.split('\n')
                    current_sub_block = ""
                    
                    for sub_block in sub_blocks:
                        if len(current_sub_block + sub_block) > 800:
                            if current_sub_block.strip():
                                blocks.append(current_sub_block.strip())
                            current_sub_block = sub_block
                        else:
                            current_sub_block += "\n" + sub_block
                    
                    if current_sub_block.strip():
                        blocks.append(current_sub_block.strip())
                else:
                    blocks.append(block)
        
        return blocks
    
    def _predict_section_from_content(self, content: str, missing_sections: List[int]) -> Optional[int]:
        """Predice la sección más probable basándose en el contenido"""
        content_lower = content.lower()
        
        # Puntajes para cada sección faltante
        section_scores = {}
        
        for section_num in missing_sections:
            score = 0
            
            # Verificar keywords específicas de la sección
            if section_num in self.fds_sections:
                for keyword in self.fds_sections[section_num]:
                    try:
                        # Limpiar keyword de caracteres regex especiales para búsqueda simple
                        clean_keyword = re.sub(r'[.*+?^${}()|[\]\\]', '', keyword)
                        if re.search(clean_keyword, content_lower, re.IGNORECASE):
                            score += 3
                    except:
                        continue
            
            # Verificar keywords de fallback
            for fallback_pattern, target_section in self.fallback_keywords.items():
                if target_section == section_num:
                    if re.search(fallback_pattern, content_lower, re.IGNORECASE):
                        score += 2
            
            # Puntajes adicionales basados en patrones específicos
            if section_num == 1 and any(word in content_lower for word in ['producto', 'empresa', 'fabricante', 'manufacturer', 'company']):
                score += 2
            elif section_num == 2 and any(word in content_lower for word in ['peligro', 'hazard', 'riesgo', 'risk', 'pictograma']):
                score += 2
            elif section_num == 3 and any(word in content_lower for word in ['cas', 'composición', 'composition', 'ingrediente', '%']):
                score += 2
            elif section_num == 4 and any(word in content_lower for word in ['primeros auxilios', 'first aid', 'inhalación', 'ingestion']):
                score += 2
            elif section_num == 9 and any(word in content_lower for word in ['densidad', 'density', 'ph', 'color', 'olor', 'viscosidad']):
                score += 2
            elif section_num == 11 and any(word in content_lower for word in ['ld50', 'lc50', 'toxicidad', 'toxicity', 'carcinogen']):
                score += 2
            elif section_num == 15 and any(word in content_lower for word in ['osha', 'epa', 'sara', 'reach', 'regulation']):
                score += 2
            
            if score > 0:
                section_scores[section_num] = score
        
        # Retornar la sección con mayor puntaje
        if section_scores:
            return max(section_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _find_relevant_content_for_section(self, text: str, section_num: int) -> Optional[str]:
        """Busca contenido relevante para una sección específica en todo el texto"""
        
        # Patterns específicos para cada sección
        section_patterns = {
            1: [r"producto.*(?:nombre|name)", r"empresa.*(?:fabricante|manufacturer)", r"teléfono.*(?:emergencia|emergency)"],
            2: [r"peligro.*(?:identificación|identification)", r"pictograma", r"clasificación.*peligro"],
            3: [r"cas.*\d{2,7}-\d{2}-\d", r"concentración.*\d+%", r"ingrediente.*peligroso"],
            4: [r"primeros.*auxilios", r"contacto.*ojo", r"inhalación.*fresco"],
            5: [r"extinción.*incendio", r"agua.*espuma", r"punto.*inflamación"],
            6: [r"derrame.*accidental", r"absorber.*material", r"contención"],
            7: [r"almacenamiento.*seco", r"manipulación.*segura", r"temperatura.*ambiente"],
            8: [r"límites.*exposición", r"epp.*protección", r"ventilación"],
            9: [r"densidad.*g/cm3", r"punto.*ebullición", r"ph.*\d"],
            10: [r"estabilidad.*química", r"incompatible.*materiales", r"descomposición"],
            11: [r"ld50.*mg/kg", r"toxicidad.*aguda", r"efectos.*salud"],
            12: [r"ecotoxicidad", r"biodegradabilidad", r"organismos.*acuáticos"],
            13: [r"eliminación.*residuos", r"disposición.*local", r"incineración"],
            14: [r"transporte.*un\d+", r"clase.*peligro.*\d", r"embalaje"],
            15: [r"osha.*pel", r"reach.*registro", r"sara.*título"],
            16: [r"fecha.*revisión", r"versión.*\d", r"información.*adicional"]
        }
        
        if section_num in section_patterns:
            for pattern in section_patterns[section_num]:
                match = re.search(pattern + r".*?(?=\n\n|\n[A-Z]|\Z)", text, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(0).strip()
                    if len(content) > 30:
                        return content
        
        return None
    
    def _extract_tables_multi_method(self, pdf_path: str, doc_id: str) -> List[Dict]:
        """Extracción de tablas usando múltiples métodos para máxima efectividad"""
        all_tables = []
        
        print(f"   🔍 Extrayendo tablas con múltiples métodos...")
        
        # MÉTODO 1: pdfplumber (más confiable para tablas simples)
        tables_pdfplumber = self._extract_tables_pdfplumber(pdf_path, doc_id)
        print(f"      - pdfplumber: {len(tables_pdfplumber)} tablas")
        all_tables.extend(tables_pdfplumber)
        
        # MÉTODO 2: tabula-py (mejor para tablas complejas)
        try:
            tables_tabula = self._extract_tables_tabula(pdf_path, doc_id)
            print(f"      - tabula-py: {len(tables_tabula)} tablas")
            all_tables.extend(tables_tabula)
        except Exception as e:
            print(f"      - tabula-py: Error - {e}")
        
        # MÉTODO 3: camelot (excelente para tablas bien estructuradas)
        try:
            tables_camelot = self._extract_tables_camelot(pdf_path, doc_id)
            print(f"      - camelot: {len(tables_camelot)} tablas")
            all_tables.extend(tables_camelot)
        except Exception as e:
            print(f"      - camelot: Error - {e}")
        
        # MÉTODO 4: PyMuPDF para detectar estructuras tabulares
        tables_pymupdf = self._extract_tables_pymupdf(pdf_path, doc_id)
        print(f"      - PyMuPDF: {len(tables_pymupdf)} tablas")
        all_tables.extend(tables_pymupdf)
        
        # Eliminar duplicados y fusionar tablas similares
        unique_tables = self._deduplicate_and_merge_tables(all_tables)
        
        print(f"   📊 Total después de limpieza: {len(unique_tables)} tablas únicas")
        
        return unique_tables
    
    def _extract_tables_pdfplumber(self, pdf_path: str, doc_id: str) -> List[Dict]:
        """Extracción con pdfplumber (método original mejorado)"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Configuraciones múltiples para diferentes tipos de tablas
                    table_settings = [
                        {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                        {"vertical_strategy": "text", "horizontal_strategy": "text"},
                        {"vertical_strategy": "explicit", "horizontal_strategy": "explicit"},
                        {"edge_min_length": 5, "min_words_vertical": 3, "min_words_horizontal": 1}
                    ]
                    
                    for setting_idx, settings in enumerate(table_settings):
                        try:
                            page_tables = page.extract_tables(table_settings=settings)
                            
                            for table_idx, table in enumerate(page_tables):
                                if table and len(table) > 1:
                                    processed_table = self._process_table_enhanced(
                                        table, page_num, f"{table_idx}_{setting_idx}", 
                                        doc_id, "pdfplumber"
                                    )
                                    if processed_table:
                                        tables.append(processed_table)
                                        
                        except Exception as e:
                            continue
        
        except Exception as e:
            print(f"⚠️ Error en pdfplumber: {e}")
        
        return tables
    
    def _extract_tables_tabula(self, pdf_path: str, doc_id: str) -> List[Dict]:
        """Extracción con tabula-py"""
        tables = []
        
        try:
            # Detectar tablas en todo el documento
            detected_tables = tabula.read_pdf(
                pdf_path, 
                pages='all', 
                multiple_tables=True,
                pandas_options={'header': 0}
            )
            
            for table_idx, df in enumerate(detected_tables):
                if df is not None and not df.empty and df.shape[0] > 1:
                    # Convertir DataFrame a formato estándar
                    table_data = df.fillna('').to_dict('records')
                    
                    processed_table = {
                        'table_id': f"{doc_id}_tabula_{table_idx}",
                        'filename': f"{doc_id}_tabula_{table_idx}.csv",
                        'page': 'multiple',  # tabula puede extraer de múltiples páginas
                        'shape': df.shape,
                        'headers': list(df.columns),
                        'data': table_data,
                        'source': 'tabula-py',
                        'confidence': 0.8
                    }
                    
                    # Guardar CSV
                    table_path = self.config.get_path('extracted_content', 'tables') / processed_table['filename']
                    df.to_csv(table_path, index=False, encoding='utf-8')
                    processed_table['path'] = str(table_path)
                    
                    tables.append(processed_table)
        
        except Exception as e:
            print(f"⚠️ Error en tabula: {e}")
        
        return tables
    
    def _extract_tables_camelot(self, pdf_path: str, doc_id: str) -> List[Dict]:
        """Extracción con camelot"""
        tables = []
        
        try:
            # Stream method (mejor para tablas sin bordes)
            stream_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            
            for table_idx, table in enumerate(stream_tables):
                if table.df is not None and not table.df.empty:
                    df = table.df
                    table_data = df.fillna('').to_dict('records')
                    
                    processed_table = {
                        'table_id': f"{doc_id}_camelot_stream_{table_idx}",
                        'filename': f"{doc_id}_camelot_stream_{table_idx}.csv",
                        'page': table.page,
                        'shape': df.shape,
                        'headers': list(df.columns) if df.columns is not None else [],
                        'data': table_data,
                        'source': 'camelot-stream',
                        'confidence': table.accuracy / 100 if hasattr(table, 'accuracy') else 0.7
                    }
                    
                    # Guardar CSV
                    table_path = self.config.get_path('extracted_content', 'tables') / processed_table['filename']
                    df.to_csv(table_path, index=False, encoding='utf-8')
                    processed_table['path'] = str(table_path)
                    
                    tables.append(processed_table)
            
            # Lattice method (mejor para tablas con bordes)
            try:
                lattice_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
                
                for table_idx, table in enumerate(lattice_tables):
                    if table.df is not None and not table.df.empty:
                        df = table.df
                        table_data = df.fillna('').to_dict('records')
                        
                        processed_table = {
                            'table_id': f"{doc_id}_camelot_lattice_{table_idx}",
                            'filename': f"{doc_id}_camelot_lattice_{table_idx}.csv",
                            'page': table.page,
                            'shape': df.shape,
                            'headers': list(df.columns) if df.columns is not None else [],
                            'data': table_data,
                            'source': 'camelot-lattice',
                            'confidence': table.accuracy / 100 if hasattr(table, 'accuracy') else 0.7
                        }
                        
                        # Guardar CSV
                        table_path = self.config.get_path('extracted_content', 'tables') / processed_table['filename']
                        df.to_csv(table_path, index=False, encoding='utf-8')
                        processed_table['path'] = str(table_path)
                        
                        tables.append(processed_table)
                        
            except Exception as e:
                print(f"⚠️ Error en camelot-lattice: {e}")
        
        except Exception as e:
            print(f"⚠️ Error en camelot: {e}")
        
        return tables
    
    def _extract_tables_pymupdf(self, pdf_path: str, doc_id: str) -> List[Dict]:
        """Extracción con PyMuPDF detectando patrones tabulares en texto"""
        tables = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extraer texto con posiciones
                text_dict = page.get_text("dict")
                
                # Buscar patrones tabulares
                potential_tables = self._find_tabular_patterns(text_dict, page_num, doc_id)
                tables.extend(potential_tables)
            
            doc.close()
        
        except Exception as e:
            print(f"⚠️ Error en PyMuPDF table extraction: {e}")
        
        return tables
    
    def _find_tabular_patterns(self, text_dict: dict, page_num: int, doc_id: str) -> List[Dict]:
        """Busca patrones tabulares en el texto extraído"""
        tables = []
        
        try:
            # Extraer todos los bloques de texto con sus posiciones
            blocks = text_dict.get("blocks", [])
            text_blocks = []
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                'text': span['text'].strip(),
                                'bbox': span['bbox'],
                                'x0': span['bbox'][0],
                                'y0': span['bbox'][1]
                            })
            
            # Agrupar texto por filas (misma coordenada Y aproximada)
            rows = {}
            tolerance = 5  # Tolerancia para agrupar en misma fila
            
            for block in text_blocks:
                if not block['text']:
                    continue
                    
                y = block['y0']
                row_key = None
                
                # Buscar fila existente
                for existing_y in rows.keys():
                    if abs(y - existing_y) <= tolerance:
                        row_key = existing_y
                        break
                
                if row_key is None:
                    row_key = y
                    rows[row_key] = []
                
                rows[row_key].append(block)
            
            # Ordenar filas por posición Y
            sorted_rows = sorted(rows.items(), key=lambda x: x[0])
            
            # Detectar tablas analizando patrones
            potential_table_rows = []
            current_table_rows = []
            
            for y, row_blocks in sorted_rows:
                # Ordenar bloques en la fila por posición X
                row_blocks.sort(key=lambda x: x['x0'])
                
                # Verificar si parece una fila de tabla
                if self._is_tabular_row(row_blocks):
                    current_table_rows.append(row_blocks)
                else:
                    # Si habíamos acumulado filas de tabla, procesarlas
                    if len(current_table_rows) >= 2:  # Al menos 2 filas para una tabla
                        potential_table_rows.append(current_table_rows.copy())
                    current_table_rows = []
            
            # Procesar última tabla potencial
            if len(current_table_rows) >= 2:
                potential_table_rows.append(current_table_rows)
            
            # Convertir patrones a tablas estructuradas
            for table_idx, table_rows in enumerate(potential_table_rows):
                processed_table = self._convert_pattern_to_table(
                    table_rows, page_num, table_idx, doc_id
                )
                if processed_table:
                    tables.append(processed_table)
        
        except Exception as e:
            print(f"⚠️ Error detectando patrones tabulares: {e}")
        
        return tables
    
    def _is_tabular_row(self, row_blocks: List[Dict]) -> bool:
        """Determina si una fila de texto parece parte de una tabla"""
        if len(row_blocks) < 2:
            return False
        
        # Verificar espaciado regular
        x_positions = [block['x0'] for block in row_blocks]
        
        # Buscar patrones numéricos o datos estructurados
        text_content = [block['text'] for block in row_blocks]
        
        # Indicadores de tabla
        numeric_count = sum(1 for text in text_content if re.search(r'\d', text))
        cas_numbers = sum(1 for text in text_content if re.search(r'\d{2,7}-\d{2}-\d', text))
        percentages = sum(1 for text in text_content if re.search(r'\d+\.?\d*\s*%', text))
        units = sum(1 for text in text_content if re.search(r'\d+\.?\d*\s*(mg|kg|ppm|ppb|g/l)', text))
        
        # Criterios para considerar fila tabular
        if cas_numbers > 0 or percentages > 0 or units > 0:
            return True
        
        if numeric_count >= len(text_content) * 0.5 and len(row_blocks) >= 3:
            return True
        
        return False
    
    def _convert_pattern_to_table(self, table_rows: List[List[Dict]], page_num: int, 
                                table_idx: int, doc_id: str) -> Optional[Dict]:
        """Convierte patrones detectados a estructura de tabla"""
        try:
            # Determinar número de columnas (máximo de cualquier fila)
            max_cols = max(len(row) for row in table_rows)
            
            # Construir estructura de tabla
            table_data = []
            headers = []
            
            # Primera fila como headers (si parece ser headers)
            first_row = table_rows[0]
            if self._looks_like_headers([block['text'] for block in first_row]):
                headers = [block['text'] for block in first_row]
                data_rows = table_rows[1:]
            else:
                headers = [f"Col_{i+1}" for i in range(max_cols)]
                data_rows = table_rows
            
            # Asegurar que headers tenga la longitud correcta
            while len(headers) < max_cols:
                headers.append(f"Col_{len(headers)+1}")
            
            # Procesar filas de datos
            for row in data_rows:
                row_dict = {}
                row_texts = [block['text'] for block in row]
                
                # Asegurar que la fila tenga el número correcto de columnas
                while len(row_texts) < max_cols:
                    row_texts.append("")
                
                for i, header in enumerate(headers):
                    if i < len(row_texts):
                        row_dict[header] = row_texts[i].strip()
                    else:
                        row_dict[header] = ""
                
                # Solo agregar filas con contenido
                if any(value.strip() for value in row_dict.values()):
                    table_data.append(row_dict)
            
            if not table_data:
                return None
            
            # Crear tabla procesada
            table_id = f"{doc_id}_pymupdf_p{page_num+1}_t{table_idx+1}"
            filename = f"{table_id}.csv"
            
            processed_table = {
                'table_id': table_id,
                'filename': filename,
                'page': page_num + 1,
                'shape': (len(table_data), len(headers)),
                'headers': headers,
                'data': table_data,'source': 'pymupdf-pattern',
                'confidence': 0.6
            }
            
            # Guardar CSV si tiene contenido válido
            if len(table_data) > 0:
                table_path = self.config.get_path('extracted_content', 'tables') / filename
                df = pd.DataFrame(table_data)
                df.to_csv(table_path, index=False, encoding='utf-8')
                processed_table['path'] = str(table_path)
                
                return processed_table
        
        except Exception as e:
            print(f"⚠️ Error convirtiendo patrón a tabla: {e}")
        
        return None
    
    def _looks_like_headers(self, texts: List[str]) -> bool:
        """Determina si una fila de texto parece ser headers de tabla"""
        if not texts:
            return False
        
        # Indicadores de headers
        header_indicators = [
            'nombre', 'name', 'producto', 'product', 'cas', 'concentracion', 'concentration',
            'porcentaje', 'percentage', 'limite', 'limit', 'valor', 'value', 'unidad', 'unit',
            'simbolo', 'symbol', 'formula', 'peligro', 'hazard', 'clasificacion', 'classification',
            'componente', 'component', 'sustancia', 'substance', 'ingrediente', 'ingredient'
        ]
        
        # Verificar si algún texto parece un header
        header_count = 0
        for text in texts:
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in header_indicators):
                header_count += 1
        
        return header_count >= len(texts) * 0.4
    
    def _process_table_enhanced(self, table: List[List], page_num: int, table_idx: str, 
                              doc_id: str, source: str) -> Optional[Dict]:
        """Procesamiento mejorado de tablas con validación y limpieza"""
        
        if not table or len(table) < 1:
            return None
        
        # Limpiar tabla
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cell for cell in cleaned_row):  # Solo filas con contenido
                cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < 2:  # Necesita al menos headers + 1 fila
            return None
        
        # Determinar headers
        headers = cleaned_table[0]
        data_rows = cleaned_table[1:]
        
        # Asegurar headers únicos
        unique_headers = []
        for i, header in enumerate(headers):
            if not header or header in unique_headers:
                header = f"Col_{i+1}"
            unique_headers.append(header)
        
        # Convertir a diccionarios
        table_data = []
        for row in data_rows:
            row_dict = {}
            for i, header in enumerate(unique_headers):
                if i < len(row):
                    row_dict[header] = row[i]
                else:
                    row_dict[header] = ""
            table_data.append(row_dict)
        
        # Crear estructura final
        table_id = f"{doc_id}_{source}_p{page_num+1}_t{table_idx}"
        filename = f"{table_id}.csv"
        
        processed_table = {
            'table_id': table_id,
            'filename': filename,
            'page': page_num + 1,
            'shape': (len(table_data), len(unique_headers)),
            'headers': unique_headers,
            'data': table_data,
            'source': source,
            'confidence': 0.8
        }
        
        # Guardar CSV
        try:
            table_path = self.config.get_path('extracted_content', 'tables') / filename
            df = pd.DataFrame(table_data)
            df.to_csv(table_path, index=False, encoding='utf-8')
            processed_table['path'] = str(table_path)
        except Exception as e:
            print(f"⚠️ Error guardando tabla {table_id}: {e}")
        
        return processed_table
    
    def _deduplicate_and_merge_tables(self, tables: List[Dict]) -> List[Dict]:
        """Elimina duplicados y fusiona tablas similares"""
        if not tables:
            return []
        
        unique_tables = []
        processed_signatures = set()
        
        for table in tables:
            # Crear signature de la tabla
            signature = self._create_table_signature(table)
            
            if signature not in processed_signatures:
                processed_signatures.add(signature)
                unique_tables.append(table)
        
        return unique_tables
    
    def _create_table_signature(self, table: Dict) -> str:
        """Crea una signature única para una tabla basada en su contenido"""
        try:
            shape = table.get('shape', (0, 0))
            headers = table.get('headers', [])
            
            # Tomar muestra del contenido para comparar
            sample_data = []
            for row in table.get('data', [])[:3]:  # Primeras 3 filas
                sample_data.extend(list(row.values())[:3])  # Primeros 3 valores
            
            signature_parts = [
                str(shape),
                '|'.join(headers[:5]),  # Primeros 5 headers
                '|'.join(str(x)[:20] for x in sample_data)  # Muestra de datos
            ]
            
            return '###'.join(signature_parts)
        except:
            return str(hash(str(table)))
    
    def _extract_images_from_page(self, page, page_num: int, doc_id: str, result: Dict):
        """Extrae imágenes de una página"""
        try:
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_filename = f"{doc_id}_p{page_num+1}_img{img_idx+1}.png"
                        img_path = self.config.get_path('extracted_content', 'images') / img_filename
                        
                        pix.save(str(img_path))
                        
                        result['images'].append({
                            'filename': img_filename,
                            'page': page_num + 1,
                            'path': str(img_path),
                            'size': (pix.width, pix.height)
                        })
                        
                        self.stats['total_images'] += 1
                    
                    pix = None
                    
                except Exception as e:
                    print(f"⚠️ Error extrayendo imagen {img_idx} de página {page_num+1}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error extrayendo imágenes de página {page_num+1}: {e}")
    
    def _extract_metadata(self, text: str, pdf_path: str) -> Dict:
        """Extrae metadatos del documento"""
        metadata = {
            'file_name': Path(pdf_path).name,
            'file_size': Path(pdf_path).stat().st_size,
            'extraction_method': 'PyMuPDF + pdfplumber + tabula + camelot',
            'text_length': len(text),
            'word_count': len(text.split()),
        }
        
        # Detectar idioma principal
        spanish_indicators = len(re.findall(r'\b(?:sección|información|peligros|composición)\b', text, re.IGNORECASE))
        english_indicators = len(re.findall(r'\b(?:section|information|hazards|composition)\b', text, re.IGNORECASE))
        
        metadata['primary_language'] = 'spanish' if spanish_indicators > english_indicators else 'english'
        
        # Buscar información del producto
        product_name = re.search(r'(?:nombre.*producto|product.*name)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if product_name:
            metadata['product_name'] = product_name.group(1).strip()
        
        # Buscar número CAS principal
        cas_numbers = re.findall(r'\b\d{2,7}-\d{2}-\d\b', text)
        if cas_numbers:
            metadata['cas_numbers'] = list(set(cas_numbers))
        
        return metadata
    
    def _generate_doc_id(self, pdf_path: Path) -> str:
        """Genera un ID único para el documento"""
        base_name = pdf_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    
    def _save_extracted_content(self, result: Dict, doc_id: str):
        """Guarda todo el contenido extraído"""
        
        # Guardar texto completo
        text_path = self.config.get_path('extracted_content', 'texts') / f"{doc_id}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        # Guardar metadatos completos
        metadata_path = self.config.get_path('extracted_content', 'metadata') / f"{doc_id}_metadata.json"
        
        # Preparar metadatos completos
        complete_metadata = {
            'document_id': doc_id,
            'source_file': result['source_file'],
            'extraction_timestamp': result['extraction_timestamp'],
            'metadata': result['metadata'],
            'sections_found': list(result['sections'].keys()) if result['sections'] else [],
            'tables_count': len(result['tables']),
            'images_count': len(result['images']),
            'processing_stats': {
                'total_sections': len(result['sections']),
                'high_confidence_sections': len([s for s in result['sections'].values() if s.get('confidence') == 'high']),
                'medium_confidence_sections': len([s for s in result['sections'].values() if s.get('confidence') == 'medium']),
                'low_confidence_sections': len([s for s in result['sections'].values() if s.get('confidence') == 'low'])
            },
            'tables_summary': [
                {
                    'id': table['table_id'],
                    'filename': table['filename'],
                    'source': table['source'],
                    'shape': table['shape'],
                    'confidence': table.get('confidence', 0)
                }
                for table in result['tables']
            ],
            'images_summary': result['images']
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(complete_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 Contenido guardado:")
        print(f"      📄 Texto: {text_path}")
        print(f"      📋 Metadatos: {metadata_path}")
        print(f"      📊 Tablas: {len(result['tables'])} archivos CSV")
        print(f"      🖼️ Imágenes: {len(result['images'])} archivos PNG")
    
    def print_statistics(self):
        """Imprime estadísticas del procesamiento"""
        print(f"\n📊 ESTADÍSTICAS DEL PROCESAMIENTO:")
        print(f"   📄 Documentos procesados: {self.stats['total_processed']}")
        print(f"   ✅ Extracciones exitosas: {self.stats['successful_extractions']}")
        print(f"   ❌ Extracciones fallidas: {self.stats['failed_extractions']}")
        print(f"   📃 Total páginas procesadas: {self.stats['total_pages']}")
        print(f"   📊 Total tablas extraídas: {self.stats['total_tables']}")
        print(f"   🖼️ Total imágenes extraídas: {self.stats['total_images']}")
        
        if self.all_sections_found:
            # Análisis de secciones encontradas
            section_coverage = {}
            for sections in self.all_sections_found:
                for section in sections:
                    section_coverage[section] = section_coverage.get(section, 0) + 1
            
            total_docs = len(self.all_sections_found)
            print(f"\n📋 COBERTURA DE SECCIONES FDS:")
            
            for section_num in range(1, 17):
                count = section_coverage.get(section_num, 0)
                percentage = (count / total_docs) * 100 if total_docs > 0 else 0
                section_name = self.section_names.get(section_num, f"Sección {section_num}")
                print(f"   {section_num:2d}. {section_name[:40]:40} {count:2d}/{total_docs} ({percentage:5.1f}%)")

# ============================================================================
# CHUNKER AVANZADO PARA FDS
# ============================================================================

class FDSChunkerAdvanced:
    """Chunker especializado para documentos FDS con estrategias múltiples"""
    
    def __init__(self, config: ProjectConfig, chunk_size: int = 800, overlap: int = 150):
        self.config = config
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_extracted_content(self, doc_id: str) -> List[Dict]:
        """Procesa contenido extraído y genera chunks optimizados"""
        
        print(f"🧩 Generando chunks para documento: {doc_id}")
        
        # Cargar contenido extraído
        extracted_content = self._load_extracted_content(doc_id)
        
        if not extracted_content:
            print(f"   ❌ No se pudo cargar contenido extraído para {doc_id}")
            return []
        
        # Generar chunks
        chunks = self.chunk_fds_document(extracted_content)
        
        # Guardar chunks
        if chunks:
            self._save_chunks(chunks, doc_id)
            print(f"   ✅ {len(chunks)} chunks generados")
        
        return chunks
    
    def _load_extracted_content(self, doc_id: str) -> Optional[Dict]:
        """Carga contenido extraído desde archivos"""
        
        try:
            # Cargar metadatos
            metadata_path = self.config.get_path('extracted_content', 'metadata') / f"{doc_id}_metadata.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Cargar texto
            text_path = self.config.get_path('extracted_content', 'texts') / f"{doc_id}.txt"
            text = ""
            if text_path.exists():
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Cargar tablas
            tables = []
            tables_folder = self.config.get_path('extracted_content', 'tables')
            
            for table_info in metadata.get('tables_summary', []):
                table_path = tables_folder / table_info['filename']
                if table_path.exists():
                    try:
                        df = pd.read_csv(table_path, encoding='utf-8')
                        table_data = {
                            'table_id': table_info['id'],
                            'filename': table_info['filename'],
                            'shape': table_info['shape'],
                            'data': df.to_dict('records'),
                            'source': table_info.get('source', 'unknown')
                        }
                        tables.append(table_data)
                    except Exception as e:
                        print(f"   ⚠️ Error cargando tabla {table_info['filename']}: {e}")
            
            # Reconstruir secciones desde texto (simplificado)
            sections = self._reconstruct_sections_from_text(text, metadata.get('sections_found', []))
            
            return {
                'document_id': doc_id,
                'text': text,
                'sections': sections,
                'tables': tables,
                'metadata': metadata.get('metadata', {})
            }
            
        except Exception as e:
            print(f"   ❌ Error cargando contenido para {doc_id}: {e}")
            return None
    
    def _reconstruct_sections_from_text(self, text: str, sections_found: List[int]) -> Dict:
        """Reconstruye secciones básicas desde texto completo"""
        sections = {}
        
        # Dividir texto en bloques aproximados para cada sección encontrada
        if sections_found and len(sections_found) > 1:
            text_length = len(text)
            section_size = text_length // len(sections_found)
            
            for i, section_num in enumerate(sorted(sections_found)):
                start = i * section_size
                end = (i + 1) * section_size if i < len(sections_found) - 1 else text_length
                
                section_text = text[start:end].strip()
                if section_text:
                    sections[section_num] = {
                        'content': section_text,
                        'title': f"Sección {section_num}"
                    }
        else:
            # Si no hay secciones identificadas, usar todo el texto
            sections[1] = {
                'content': text,
                'title': "Contenido completo"
            }
        
        return sections
    
    def chunk_fds_document(self, document_data: Dict) -> List[Dict]:
        """Chunking principal para documentos FDS"""
        
        chunks = []
        doc_id = document_data['document_id']
        metadata = document_data.get('metadata', {})
        
        # ESTRATEGIA 1: Chunking por secciones
        if document_data.get('sections'):
            section_chunks = self._chunk_by_sections(document_data['sections'], metadata)
            chunks.extend(section_chunks)
            print(f"   📋 {len(section_chunks)} chunks de secciones")
        
        # ESTRATEGIA 2: Chunking de tablas
        if document_data.get('tables'):
            table_chunks = self._chunk_tables(document_data['tables'], metadata)
            chunks.extend(table_chunks)
            print(f"   📊 {len(table_chunks)} chunks de tablas")
        
        # ESTRATEGIA 3: Chunking general si no hay secciones
        if not document_data.get('sections') and document_data.get('text'):
            general_chunks = self._chunk_general_text(document_data['text'], metadata)
            chunks.extend(general_chunks)
            print(f"   📄 {len(general_chunks)} chunks generales")
        
        return chunks
    
    def _chunk_by_sections(self, sections: Dict, metadata: Dict) -> List[Dict]:
        """Chunking por secciones"""
        chunks = []
        
        for section_num, section_data in sections.items():
            content = section_data['content']
            
            if len(content) <= self.chunk_size:
                chunks.append({
                    'text': content,
                    'metadata': {
                        **metadata,
                        'chunk_type': 'section',
                        'section_number': section_num,
                        'section_title': section_data['title'],
                        'chunk_size': len(content)
                    }
                })
            else:
                section_chunks = self._split_with_overlap(content)
                
                for i, chunk_text in enumerate(section_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **metadata,
                            'chunk_type': 'section',
                            'section_number': section_num,
                            'section_title': section_data['title'],
                            'chunk_index': i,
                            'chunk_size': len(chunk_text)
                        }
                    })
        
        return chunks
    
    def _chunk_tables(self, tables: List[Dict], metadata: Dict) -> List[Dict]:
        """Chunking para tablas"""
        chunks = []
        
        for table in tables:
            searchable_text = self._table_to_searchable_text(table['data'])
            
            if searchable_text.strip():
                chunks.append({
                    'text': searchable_text,
                    'metadata': {
                        **metadata,
                        'chunk_type': 'table',
                        'table_id': table['table_id'],
                        'table_shape': table['shape'],
                        'source': table.get('source', 'unknown'),
                        'chunk_size': len(searchable_text)
                    },
                    'structured_data': table['data']
                })
        
        return chunks
    
    def _chunk_general_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunking para texto general"""
        chunks = []
        
        text_chunks = self._split_with_overlap(text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_type': 'general',
                    'chunk_index': i,
                    'chunk_size': len(chunk_text)
                }
            })
        
        return chunks
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """División con overlap"""
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Buscar punto de corte natural
            cutoff = text.rfind('\n\n', start, end)
            if cutoff == -1 or cutoff <= start:
                cutoff = text.rfind('\n', start, end)
                if cutoff == -1 or cutoff <= start:
                    cutoff = text.rfind('. ', start, end)
                    if cutoff == -1 or cutoff <= start:
                        cutoff = end
            
            chunk_text = text[start:cutoff].strip()
            if chunk_text:
                chunks.append(chunk_text)
            
            start = max(cutoff - self.overlap, start + 1)
        
        return chunks
    
    def _table_to_searchable_text(self, table_data: List[Dict]) -> str:
        """Convierte tabla a texto searchable"""
        if not table_data:
            return ""
        
        text_parts = []
        
        headers = list(table_data[0].keys()) if table_data else []
        if headers:
            clean_headers = [h for h in headers if h and h.strip()]
            if clean_headers:
                text_parts.append("Columnas: " + " | ".join(clean_headers))
        
        for row in table_data:
            row_text = []
            for key, value in row.items():
                if key and value and str(value).strip() and str(value) != 'None':
                    row_text.append(f"{key}: {str(value).strip()}")
            
            if row_text:
                text_parts.append(" | ".join(row_text))
        
        return "\n".join(text_parts)
    
    def _save_chunks(self, chunks: List[Dict], doc_id: str):
        """Guarda chunks en archivo JSON"""
        output_path = self.config.get_path('processed_chunks') / f"{doc_id}_chunks.json"
        
        chunk_data = {
            'document_id': doc_id,
            'total_chunks': len(chunks),
            'chunk_types': {},
            'processing_timestamp': datetime.now().isoformat(),
            'chunks': []
        }
        
        # Estadísticas por tipo
        for chunk in chunks:
            chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
            chunk_data['chunk_types'][chunk_type] = chunk_data['chunk_types'].get(chunk_type, 0) + 1
        
        # Guardar chunks (texto completo)
        for i, chunk in enumerate(chunks):
            chunk_data['chunks'].append({
                'id': i,
                'text': chunk['text'],
                'text_length': len(chunk['text']),
                'metadata': chunk['metadata'],
                'structured_data': chunk.get('structured_data', None)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 Chunks guardados: {output_path}")

# ============================================================================
# PIPELINE PRINCIPAL COMPLETO
# ============================================================================

def main_processing_pipeline(raw_data_folder: str = None, base_project_path: str = None):
    """Pipeline principal de procesamiento con mejoras"""
    
    print("🚀 INICIANDO PIPELINE AVANZADO DE PROCESAMIENTO FDS")
    print("=" * 70)
    print("🔧 Mejoras incluidas:")
    print("   • Segmentación garantizada de 16 secciones")
    print("   • Extracción multi-método de tablas")
    print("   • Análisis de contenido inteligente")
    print("   • Detección de patrones tabulares")
    print("=" * 70)
    
    # 1. Configuración del proyecto
    config = ProjectConfig(base_project_path)
    
    # 2. Definir carpeta de datos raw
    if raw_data_folder:
        raw_folder = Path(raw_data_folder)
    else:
        raw_folder = config.get_path('raw_data')
    
    if not raw_folder.exists():
        print(f"❌ Carpeta de datos raw no encontrada: {raw_folder}")
        print(f"   Por favor coloca tus PDFs en: {raw_folder}")
        raw_folder.mkdir(parents=True, exist_ok=True)
        print(f"✅ Carpeta creada: {raw_folder}")
        return []
    
    # 3. Buscar archivos PDF
    pdf_files = list(raw_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No se encontraron archivos PDF en: {raw_folder}")
        print(f"   Coloca tus archivos PDF en: {raw_folder}")
        return []
    
    print(f"📦 Encontrados {len(pdf_files)} archivos PDF")
    
    # 4. Inicializar procesadores
    extractor = FDSExtractorAdvanced(config)
    chunker = FDSChunkerAdvanced(config, chunk_size=600, overlap=100)
    
    # 5. Procesar cada PDF
    processed_docs = []
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n{'='*70}")
        print(f"📄 PROCESANDO DOCUMENTO {i}/{len(pdf_files)}")
        print(f"{'='*70}")
        
        # PASO 1: Extraer contenido del PDF
        extraction_result = extractor.process_pdf_file(str(pdf_file))
        
        if extraction_result:
            doc_id = extraction_result['document_id']
            processed_docs.append(doc_id)
            
            # PASO 2: Generar chunks del contenido extraído
            print(f"🔄 Generando chunks para {doc_id}...")
            chunks = chunker.process_extracted_content(doc_id)
            
            if chunks:
                print(f"✅ {len(chunks)} chunks generados y guardados")
            else:
                print(f"⚠️ No se pudieron generar chunks")
        else:
            print(f"❌ Error procesando {pdf_file.name}")
    
    # 6. Estadísticas finales detalladas
    print(f"\n{'='*70}")
    print("📊 RESUMEN FINAL DEL PROCESAMIENTO")
    print(f"{'='*70}")
    extractor.print_statistics()
    
    print(f"\n📁 RUTAS DE ARCHIVOS GENERADOS:")
    print(f"   📂 Contenido extraído: {config.get_path('extracted_content')}")
    print(f"   🧩 Chunks procesados: {config.get_path('processed_chunks')}")
    print(f"   📊 Tablas CSV: {config.get_path('extracted_content', 'tables')}")
    print(f"   🖼️ Imágenes: {config.get_path('extracted_content', 'images')}")
    print(f"   📄 Textos: {config.get_path('extracted_content', 'texts')}")
    print(f"   📋 Metadatos: {config.get_path('extracted_content', 'metadata')}")
    
    print(f"\n🎯 DOCUMENTOS PROCESADOS EXITOSAMENTE:")
    for doc_id in processed_docs:
        print(f"   ✅ {doc_id}")
    
    return processed_docs

def validate_dependencies():
    """Valida que todas las dependencias estén instaladas"""
    dependencies = {
        'fitz': 'PyMuPDF',
        'pdfplumber': 'pdfplumber', 
        'tabula': 'tabula-py',
        'camelot': 'camelot-py',
        'pandas': 'pandas'
    }
    
    print("🔍 VALIDANDO DEPENDENCIAS:")
    missing = []
    
    for module, name in dependencies.items():
        try:
            if module == 'tabula':
                import tabula
            elif module == 'camelot':
                import camelot
            elif module == 'fitz':
                import fitz
            elif module == 'pdfplumber':
                import pdfplumber
            elif module == 'pandas':
                import pandas
            else:
                __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - FALTANTE")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️ DEPENDENCIAS FALTANTES:")
        print(f"   Instala con: pip install {' '.join(missing)}")
        return False
    
    print(f"\n✅ Todas las dependencias están disponibles")
    return True

# ============================================================================
# FUNCIONES DE UTILIDAD Y MANTENIMIENTO
# ============================================================================

def clean_extracted_data(base_project_path: str = None, keep_backup: bool = True):
    """Limpia datos extraídos corruptos o incompletos"""
    
    print("🧹 LIMPIANDO DATOS EXTRAÍDOS")
    print("=" * 40)
    
    config = ProjectConfig(base_project_path)
    
    if keep_backup:
        backup_folder = config.base_path / 'backup_cleaned_data'
        backup_folder.mkdir(exist_ok=True)
        print(f"📦 Backup guardado en: {backup_folder}")
    
    # Buscar archivos problemáticos
    metadata_files = list(config.get_path('extracted_content', 'metadata').glob("*_metadata.json"))
    cleaned_count = 0
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            doc_id = metadata.get('document_id', '')
            
            # Verificar integridad
            issues = []
            
            # Verificar texto
            text_file = config.get_path('extracted_content', 'texts') / f"{doc_id}.txt"
            if not text_file.exists():
                issues.append('missing_text')
            
            # Verificar secciones
            sections_found = metadata.get('sections_found', [])
            if len(sections_found) < 10:  # Muy pocas secciones
                issues.append('insufficient_sections')
            
            # Verificar tablas
            tables_summary = metadata.get('tables_summary', [])
            for table_info in tables_summary:
                table_path = config.get_path('extracted_content', 'tables') / table_info.get('filename', '')
                if not table_path.exists():
                    issues.append('missing_table')
                    break
            
            if issues:
                print(f"⚠️ Problemas en {doc_id}: {', '.join(issues)}")
                
                if keep_backup:
                    # Mover a backup
                    shutil.move(str(metadata_file), str(backup_folder / metadata_file.name))
                    if text_file.exists():
                        shutil.move(str(text_file), str(backup_folder / text_file.name))
                else:
                    # Eliminar archivos problemáticos
                    metadata_file.unlink()
                    if text_file.exists():
                        text_file.unlink()
                
                cleaned_count += 1
        
        except Exception as e:
            print(f"❌ Error procesando {metadata_file}: {e}")
    
    print(f"✅ Limpieza completada: {cleaned_count} documentos procesados")

def analyze_extraction_quality(base_project_path: str = None):
    """Analiza la calidad de las extracciones realizadas"""
    
    print("📈 ANALIZANDO CALIDAD DE EXTRACCIONES")
    print("=" * 45)
    
    config = ProjectConfig(base_project_path)
    metadata_folder = config.get_path('extracted_content', 'metadata')
    metadata_files = list(metadata_folder.glob("*_metadata.json"))
    
    if not metadata_files:
        print("❌ No se encontraron metadatos para analizar")
        return
    
    analysis = {
        'total_documents': len(metadata_files),
        'sections_analysis': {},
        'tables_analysis': {'total': 0, 'by_source': {}},
        'confidence_analysis': {'high': 0, 'medium': 0, 'low': 0}
    }
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Análisis de secciones
            sections_found = metadata.get('sections_found', [])
            for section in sections_found:
                analysis['sections_analysis'][section] = analysis['sections_analysis'].get(section, 0) + 1
            
            # Análisis de tablas
            tables_summary = metadata.get('tables_summary', [])
            analysis['tables_analysis']['total'] += len(tables_summary)
            
            for table in tables_summary:
                source = table.get('source', 'unknown')
                analysis['tables_analysis']['by_source'][source] = analysis['tables_analysis']['by_source'].get(source, 0) + 1
            
            # Análisis de confianza
            processing_stats = metadata.get('processing_stats', {})
            analysis['confidence_analysis']['high'] += processing_stats.get('high_confidence_sections', 0)
            analysis['confidence_analysis']['medium'] += processing_stats.get('medium_confidence_sections', 0)
            analysis['confidence_analysis']['low'] += processing_stats.get('low_confidence_sections', 0)
            
        except Exception as e:
            print(f"⚠️ Error analizando {metadata_file}: {e}")
    
    # Mostrar resultados
    print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
    print(f"   📄 Total documentos: {analysis['total_documents']}")
    
    print(f"\n📋 COBERTURA DE SECCIONES:")
    for section_num in range(1, 17):
        count = analysis['sections_analysis'].get(section_num, 0)
        percentage = (count / analysis['total_documents']) * 100
        section_name = f"Sección {section_num}"
        print(f"   {section_name}: {count}/{analysis['total_documents']} ({percentage:.1f}%)")
    
    print(f"\n📊 TABLAS EXTRAÍDAS:")
    print(f"   Total: {analysis['tables_analysis']['total']}")
    print(f"   Promedio por documento: {analysis['tables_analysis']['total'] / analysis['total_documents']:.1f}")
    print(f"   Por método:")
    for source, count in analysis['tables_analysis']['by_source'].items():
        print(f"     - {source}: {count}")
    
    total_sections = sum(analysis['confidence_analysis'].values())
    if total_sections > 0:
        print(f"\n🎯 CONFIANZA EN SECCIONES:")
        for level, count in analysis['confidence_analysis'].items():
            percentage = (count / total_sections) * 100
            print(f"   {level.title()}: {count} ({percentage:.1f}%)")

def export_to_structured_format(base_project_path: str = None, output_format: str = 'json'):
    """Exporta todo el contenido extraído a un formato estructurado"""
    
    print(f"📤 EXPORTANDO A FORMATO {output_format.upper()}")
    print("=" * 40)
    
    config = ProjectConfig(base_project_path)
    output_folder = config.get_path('outputs')
    
    # Recopilar todos los datos
    all_data = {
        'extraction_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_documents': 0,
            'format': output_format
        },
        'documents': []
    }
    
    metadata_files = list(config.get_path('extracted_content', 'metadata').glob("*_metadata.json"))
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            doc_id = metadata.get('document_id', '')
            
            # Cargar texto
            text_file = config.get_path('extracted_content', 'texts') / f"{doc_id}.txt"
            text_content = ""
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            
            # Cargar tablas
            tables_data = []
            tables_folder = config.get_path('extracted_content', 'tables')
            table_files = list(tables_folder.glob(f"{doc_id}_*.csv"))
            
            for table_file in table_files:
                try:
                    df = pd.read_csv(table_file, encoding='utf-8')
                    tables_data.append({
                        'filename': table_file.name,
                        'shape': df.shape,
                        'data': df.to_dict('records')
                    })
                except Exception as e:
                    print(f"⚠️ Error cargando tabla {table_file}: {e}")
            
            # Compilar documento
            doc_data = {
                'document_id': doc_id,
                'metadata': metadata.get('metadata', {}),
                'text': text_content,
                'sections_found': metadata.get('sections_found', []),
                'tables': tables_data,
                'extraction_stats': metadata.get('processing_stats', {}),
                'extraction_timestamp': metadata.get('extraction_timestamp', '')
            }
            
            all_data['documents'].append(doc_data)
            
        except Exception as e:
            print(f"❌ Error procesando {metadata_file}: {e}")
    
    all_data['extraction_summary']['total_documents'] = len(all_data['documents'])
    
    # Exportar según formato
    if output_format.lower() == 'json':
        output_file = output_folder / f"fds_extraction_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    elif output_format.lower() == 'excel':
        output_file = output_folder / f"fds_extraction_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Hoja de resumen
            summary_df = pd.DataFrame([all_data['extraction_summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Hoja por documento (limitado a primeros 10 por espacio)
            for i, doc in enumerate(all_data['documents'][:10]):
                sheet_name = f"Doc_{i+1}"[:31]  # Excel limita nombres de hojas
                doc_df = pd.DataFrame([{
                    'field': k,
                    'value': str(v)[:32000] if isinstance(v, str) else str(v)  # Excel limita contenido de celda
                } for k, v in doc.items() if k not in ['tables', 'text']])
                doc_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    else:
        print(f"❌ Formato no soportado: {output_format}")
        return
    
    print(f"✅ Exportación completada: {output_file}")
    return str(output_file)

def run_extraction_tests(test_pdf_path: str = None, base_project_path: str = None):
    """Ejecuta tests básicos del sistema de extracción"""
    
    print("🧪 EJECUTANDO TESTS DE EXTRACCIÓN")
    print("=" * 40)
    
    # Validar dependencias
    if not validate_dependencies():
        print("❌ Tests abortados: dependencias faltantes")
        return False
    
    config = ProjectConfig(base_project_path)
    
    # Si no se proporciona PDF de test, buscar uno
    if not test_pdf_path:
        raw_folder = config.get_path('raw_data')
        pdf_files = list(raw_folder.glob("*.pdf"))
        if pdf_files:
            test_pdf_path = str(pdf_files[0])
        else:
            print("❌ No se encontró PDF para testing")
            return False
    
    test_pdf_path = Path(test_pdf_path)
    if not test_pdf_path.exists():
        print(f"❌ Archivo de test no encontrado: {test_pdf_path}")
        return False
    
    print(f"🔬 Usando archivo de test: {test_pdf_path.name}")
    
    # Test 1: Extracción básica
    print("\n📋 Test 1: Extracción básica...")
    try:
        extractor = FDSExtractorAdvanced(config)
        result = extractor._extract_from_pdf(str(test_pdf_path), "test_doc")
        
        assert result is not None, "Resultado de extracción es None"
        assert 'text' in result, "Falta texto en resultado"
        assert 'sections' in result, "Falta secciones en resultado"
        assert 'tables' in result, "Falta tablas en resultado"
        assert len(result['sections']) > 0, "No se encontraron secciones"
        
        print(f"   ✅ Extracción exitosa: {len(result['sections'])} secciones, {len(result['tables'])} tablas")
        
    except Exception as e:
        print(f"   ❌ Error en test de extracción: {e}")
        return False
    
    # Test 2: Chunking
    print("\n📋 Test 2: Chunking...")
    try:
        chunker = FDSChunkerAdvanced(config)
        
        # Simular datos extraídos
        test_data = {
            'document_id': 'test_doc',
            'text': result['text'][:5000],  # Limitado para test
            'sections': {1: {'content': result['text'][:1000], 'title': 'Test'}},
            'tables': result['tables'][:1] if result['tables'] else [],
            'metadata': {'test': True}
        }
        
        chunks = chunker.chunk_fds_document(test_data)
        
        assert len(chunks) > 0, "No se generaron chunks"
        assert all('text' in chunk for chunk in chunks), "Chunks sin texto"
        assert all('metadata' in chunk for chunk in chunks), "Chunks sin metadata"
        
        print(f"   ✅ Chunking exitoso: {len(chunks)} chunks generados")
        
    except Exception as e:
        print(f"   ❌ Error en test de chunking: {e}")
        return False
    
    print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE")
    return True

def setup_project_environment(base_path: str, create_sample_config: bool = True):
    """Configura el entorno completo del proyecto"""
    
    print("🚀 CONFIGURANDO ENTORNO DEL PROYECTO")
    print("=" * 45)
    
    base_path = Path(base_path)
    
    # Crear estructura básica
    config = ProjectConfig(str(base_path))
    
    if create_sample_config:
        # Crear archivo de configuración de ejemplo
        sample_config = {
            "extraction_settings": {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "table_extraction_methods": ["pdfplumber", "tabula", "camelot", "pymupdf"],
                "image_extraction": True,
                "max_image_size_mb": 10
            },
            "section_detection": {
                "require_all_16_sections": True,
                "fallback_analysis": True,
                "confidence_threshold": 0.5
            },
            "output_settings": {
                "save_csv_tables": True,
                "save_images": True,
                "create_chunks": True,
                "detailed_metadata": True
            }
        }
        
        config_file = base_path / 'config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, ensure_ascii=False, indent=2)
        
        print(f"📋 Configuración de ejemplo creada: {config_file}")
    
    # Crear archivo README
    readme_content = """# Extractor Avanzado de Fichas de Datos de Seguridad (FDS)

## Estructura del Proyecto

```
proyecto/
├── data/
│   ├── raw_documents/          # PDFs originales aquí
│   ├── extracted_content/      # Contenido extraído
│   │   ├── texts/             # Textos completos
│   │   ├── tables/            # Tablas en CSV
│   │   ├── images/            # Imágenes extraídas
│   │   └── metadata/          # Metadatos de extracción
│   ├── processed_chunks/       # Chunks procesados
│   └── vector_db/             # Base de datos vectorial
├── logs/                      # Logs de procesamiento
├── outputs/                   # Exportaciones
└── config.json               # Configuración

## Uso Básico

1. Coloca tus PDFs en `data/raw_documents/`
2. Ejecuta el pipeline principal
3. Revisa los resultados en las carpetas correspondientes

## Características

- ✅ Garantiza extracción de las 16 secciones FDS
- ✅ Múltiples métodos de extracción de tablas
- ✅ Análisis inteligente de contenido
- ✅ Detección de patrones tabulares
- ✅ Estadísticas detalladas de calidad
"""
    
    readme_file = base_path / 'README.md'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 README creado: {readme_file}")
    print("✅ Entorno configurado correctamente")
    return str(base_path)

# ============================================================================
# FUNCIONES DE ACCESO RÁPIDO
# ============================================================================

def extract_content_only(raw_data_folder: str, base_project_path: str = None):
    """Solo extrae contenido SIN generar chunks"""
    
    print("🔍 EXTRAYENDO CONTENIDO DE PDFs")
    print("=" * 35)
    
    config = ProjectConfig(base_project_path)
    extractor = FDSExtractorAdvanced(config)
    
    raw_folder = Path(raw_data_folder)
    pdf_files = list(raw_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No se encontraron PDFs en: {raw_folder}")
        return []
    
    processed_docs = []
    for pdf_file in pdf_files:
        result = extractor.process_pdf_file(str(pdf_file))
        if result:
            processed_docs.append(result['document_id'])
    
    extractor.print_statistics()
    return processed_docs

def generate_chunks_only(doc_ids: List[str] = None, base_project_path: str = None):
    """Solo genera chunks de contenido ya extraído"""
    
    print("🧩 GENERANDO CHUNKS DE CONTENIDO EXTRAÍDO")
    print("=" * 45)
    
    config = ProjectConfig(base_project_path)
    chunker = FDSChunkerAdvanced(config)
    
    # Si no se especifican doc_ids, buscar todos los metadatos
    if not doc_ids:
        metadata_folder = config.get_path('extracted_content', 'metadata')
        metadata_files = list(metadata_folder.glob("*_metadata.json"))
        doc_ids = [f.stem.replace('_metadata', '') for f in metadata_files]
    
    if not doc_ids:
        print("❌ No se encontraron documentos procesados")
        return
    
    print(f"📦 Procesando chunks para {len(doc_ids)} documentos")
    
    total_chunks = 0
    for doc_id in doc_ids:
        print(f"\n🔄 Generando chunks para: {doc_id}")
        chunks = chunker.process_extracted_content(doc_id)
        total_chunks += len(chunks)
    
    print(f"\n✅ Total de chunks generados: {total_chunks}")

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Validar dependencias primero
    if not validate_dependencies():
        print("❌ Por favor instala las dependencias faltantes antes de continuar")
        exit(1)
    
    # CONFIGURACIÓN - AJUSTA ESTAS RUTAS
    RAW_DATA_FOLDER = r"/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad/data/raw_documents"
    BASE_PROJECT_PATH = r"/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad"
    
    # OPCIÓN 1: Proceso completo (extracción + chunks)
    #print("Iniciando proceso completo...")
    processed = main_processing_pipeline(RAW_DATA_FOLDER, BASE_PROJECT_PATH)
    
    # OPCIÓN 2: Solo extracción
    #processed = extract_content_only(RAW_DATA_FOLDER, BASE_PROJECT_PATH)
    
    # OPCIÓN 3: Solo chunks (si ya tienes contenido extraído)
    # generate_chunks_only(base_project_path=BASE_PROJECT_PATH)
    
    # OPCIÓN 4: Análisis de calidad
    # analyze_extraction_quality(BASE_PROJECT_PATH)
    
    # OPCIÓN 5: Exportar datos
    # export_to_structured_format(BASE_PROJECT_PATH, 'json')
    
    # OPCIÓN 6: Tests
    # run_extraction_tests(base_project_path=BASE_PROJECT_PATH)

print("\n✅ Extractor FDS Avanzado completado")
print("🎯 Funciones disponibles:")
print("   • main_processing_pipeline() - Proceso completo")
print("   • extract_content_only() - Solo extracción")
print("   • generate_chunks_only() - Solo chunking")
print("   • analyze_extraction_quality() - Análisis de calidad")
print("   • export_to_structured_format() - Exportar datos")
print("   • run_extraction_tests() - Ejecutar tests")
print("   • setup_project_environment() - Configurar proyecto")
print("   • clean_extracted_data() - Limpiar datos")
print("   • validate_dependencies() - Validar dependencias")