import os
import re
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
import fitz
import pdfplumber
import camelot
from camelot.core import TableList
from io import StringIO
import shutil

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
# CLASE PRINCIPAL DEL EXTRACTOR
# ============================================================================

class FDSExtractorAdvanced:
    """Extractor completo para Fichas de Datos de Seguridad"""

    def __init__(self, config: ProjectConfig):
        self.config = config
        
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
            10: [r"estabilidad.*reactividad", r"estabilidad", r"reactividad", r"incompatibilidad", r"descomposici[oó]n", r"descomposicion", r"polimerizaci[oó]n", r"polimerizacion", r"condiciones.*evitar", r"materiales.*evitar", r"estabilidad.*qu[ií]mica", r"stability.*reactivity", r"stability", r"reactivity", r"incompatibility", r"decomposition", r"polymerization", r"polymerisation", r"conditions.*avoid", r"materials.*avoid", r"chemical.*stability", r"thermal.*stability", r"productos.*descomposici[oó]n", r"decomposition.*products", r"reacciones.*peligrosas", r"hazardous.*reactions", r"condiciones.*inestabilidad", r"instability.*conditions", r"catalizadores", r"catalysts", r"inhibidores", r"inhibitors", r"calor", r"heat", r"luz", r"light", r"humedad", r"moisture", r"oxidantes", r"oxidizers", r"ácidos", r"acids", r"bases", r"álcalis"],
            11: [r"informaci[oó]n.*toxicol[oó]gica", r"informacion.*toxicologica", r"toxicolog[ií]a", r"toxicologia", r"efectos.*salud", r"toxicidad", r"efectos.*adversos", r"carcinogenicidad", r"mutagenicidad", r"teratogenicidad", r"efectos.*tóxicos", r"toxicological.*information", r"toxicology", r"health.*effects", r"toxicity", r"adverse.*effects", r"carcinogenicity", r"mutagenicity", r"teratogenicity", r"reproductive.*toxicity", r"health.*hazards", r"toxic.*effects", r"dosis.*letal", r"dl50", r"ld50", r"cl50", r"lc50", r"dose.*letale", r"lethal.*dose", r"lethal.*concentration", r"concentraci[oó]n.*letal", r"irritaci[oó]n", r"irritacion", r"sensibilizaci[oó]n", r"sensibilizacion", r"irritation", r"sensitization", r"corrosión", r"corrosion", r"effects.*overexposure", r"efectos.*sobreexposici[oó]n", r"acute.*toxicity", r"toxicidad.*aguda", r"chronic.*effects", r"efectos.*cr[oó]nicos", r"routes.*exposure", r"v[ií]as.*exposici[oó]n", r"absorption", r"absorci[oó]n"],
            12: [r"informaci[oó]n.*ecol[oó]gica", r"informacion.*ecologica", r"ecolog[ií]a", r"ecologia", r"ecotoxicidad", r"ecotoxicity", r"efectos.*ambientales", r"medio.*ambiente", r"biodegradabilidad", r"bioacumulaci[oó]n", r"bioacumulacion", r"persistencia", r"informaci[oó]n.*ambiental", r"datos.*ambientales", r"ecological.*information", r"ecology", r"ecotoxicity", r"environmental.*effects", r"environment", r"biodegradability", r"bioaccumulation", r"persistence", r"mobility", r"environmental.*fate", r"ecotoxicological", r"environmental.*data", r"vida.*acuática", r"vida.*acuatica", r"organismos.*acu[aá]ticos", r"aquatic.*life", r"aquatic.*organisms", r"degradaci[oó]n", r"degradacion", r"degradation", r"peces", r"dafnias", r"algas", r"fish", r"daphnia", r"algae", r"factor.*m", r"m.*factor", r"pbt", r"mpb", r"pmb", r"vmpvb", r"bioconcentraci[oó]n", r"bioconcentration", r"movilidad.*suelo", r"soil.*mobility"],
            13: [r"eliminaci[oó]n", r"eliminacion", r"disposici[oó]n", r"disposicion", r"desecho", r"residuos", r"tratamiento.*residuos", r"gesti[oó]n.*residuos", r"gestion.*residuos", r"destrucci[oó]n", r"destruccion", r"incineración", r"incineracion", r"consideraciones.*eliminaci[oó]n", r"m[eé]todos.*eliminaci[oó]n", r"disposal", r"waste.*disposal", r"waste.*treatment", r"waste.*management", r"destruction", r"incineration", r"landfill", r"hazardous.*waste", r"disposal.*considerations", r"disposal.*methods", r"waste.*handling", r"métodos.*eliminación", r"metodos.*eliminacion", r"disposal.*methods", r"tratamiento.*qu[ií]mico", r"tratamiento.*biol[oó]gico", r"chemical.*treatment", r"biological.*treatment", r"reciclaje", r"recicling", r"reciclado", r"recycled", r"reciclar", r"recycle", r"reuso", r"reutilizar", r"reutilizacion", r"reuse", r"reutilization", r"envases", r"containers", r"embalajes", r"packagings"],
            14: [r"informaci[oó]n.*transporte", r"informacion.*transporte", r"transporte", r"transportation", r"regulaciones.*transporte", r"clasificaci[oó]n.*transporte", r"categor[ií]a.*transporte", r"n[uú]mero.*onu", r"n[uú]mero.*un", r"clase.*peligro", r"grupo.*embalaje", r"etiquetas.*transporte", r"s[ií]mbolos.*transporte", r"transporte.*mar[ií]timo", r"transporte.*a[eé]reo", r"transporte.*terrestre", r"transport.*information", r"transportation", r"transport.*regulations", r"transport.*classification", r"transport.*category", r"un.*number", r"un.*no", r"imdg", r"iata", r"dot", r"dangerous.*goods", r"mercanc[ií]as.*peligrosas", r"clase.*onu", r"un.*class", r"packing.*group", r"marine.*pollutant", r"contaminante.*marino", r"marpol", r"adr", r"rid"],
            15: [r"informaci[oó]n.*reglamentaria", r"informacion.*reglamentaria", r"regulaciones", r"reglamentos", r"legislaci[oó]n", r"legislacion", r"normas.*nacionales", r"normas.*internacionales", r"leyes", r"informaci[oó]n.*seguridad", r"informacion.*salud", r"informaci[oó]n.*ambiental", r"regulatory.*information", r"regulations", r"legislation", r"national.*norms", r"international.*norms", r"laws", r"safety.*information", r"health.*information", r"environmental.*information", r"osha", r"epca", r"sara", r"ts", r"sara.*title", r"prop.*65", r"california.*proposition", r"inventario.*qu[ií]mico", r"chemical.*inventory", r"aics", r"dsl", r"ndsl", r"iel", r"keci", r"piccs", r"tsr", r"iecsc", r"ecl"],
            16: [r"otra.*informaci[oó]n", r"otra.*informacion", r"otros.*datos", r"other.*information", r"revisión", r"revision", r"fecha.*revisión", r"fecha.*emisión", r"fecha.*preparaci[oó]n", r"fecha.*actualizaci[oó]n", r"otras.*consideraciones", r"revision.*date", r"issue.*date", r"preparation.*date", r"update.*date", r"other.*considerations", r"abreviaturas", r"acr[oó]nimos", r"abreviations", r"acronyms", r"exenc[ioó]n.*responsabilidad", r"disclaimer", r"fuentes.*datos", r"data.*sources", r"referencias", r"references", r"entrenamiento", r"training", r"capacitaci[oó]n", r"training", r"usos.*recomendados", r"recommended.*uses"]
        }
        
        self.stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_pages': 0,
            'total_tables': 0,
            'total_images': 0,
        }

    def _get_page_content_by_lib(self, doc, lib_name: str, p_idx: int):
        if lib_name == 'pdfplumber':
            with pdfplumber.open(doc.name) as pdf:
                page = pdf.pages[p_idx]
                return page.extract_text()
        return ""

    def _extract_tables_camelot(self, file_path: str) -> TableList:
        """Extrae tablas usando Camelot con opciones mejoradas y sin conflictos."""
        try:
            # Intentar primero con el flavor 'lattice' para tablas con líneas
            tables = camelot.read_pdf(file_path, flavor='lattice', pages='all', line_scale=40,
                                       split_text=True, strip_text=' \n\t')
            if not tables or tables.n < 1:
                # Si no encuentra tablas, intentar con 'stream' para tablas sin líneas
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
        # Eliminar el prefijo "Sección X" repetido si aparece al inicio
        regex_header = r"(?i)(?:secci[oó]n)?\s*" + re.escape(str(section_num)) + r"[:.\s-]+\s*"
        cleaned_text = re.sub(regex_header, '', text, count=1).strip()
        
        # Unificar múltiples saltos de línea y espacios
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
                        print(f"    ✓ Contenido reasignado a Sección {missing_num}")
                    else:
                        print(f"    ❌ No se encontró contenido para reasignar a Sección {missing_num}")
                else:
                    print(f"    ❌ No se encontró contenido para reasignar a Sección {missing_num}")

        return sections_content
    def _clean_section_text(self, text: str, section_num: int) -> str:
        """Limpia el texto de una sección, eliminando artefactos y espacios extra."""
        # Eliminar el prefijo "Sección X" repetido si aparece al inicio
        regex_header = r"(?i)(?:secci[oó]n)?\s*" + re.escape(str(section_num)) + r"[:.\s-]+\s*"
        cleaned_text = re.sub(regex_header, '', text, count=1).strip()

        # Unificar múltiples saltos de línea y espacios
        cleaned_text = re.sub(r'\s*\n\s*', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

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
    
    def process_pdf_file(self, file_path: str):
        self.stats['total_processed'] += 1
        start_time = datetime.now()
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        print(f"🔄 Procesando: {file_name}")

        try:
            with fitz.open(file_path) as doc:
                self.stats['total_pages'] += doc.page_count

                full_text = self._get_all_pages_text(doc)
                sections = self._extract_sections_with_regex(full_text)
                
                if not sections or len(sections) < 16:
                    print(f"❌ Error procesando (no se encontraron las 16 secciones): {file_name}")
                    self.stats['failed_extractions'] += 1
                    return False
                
                print("  🔍 Extrayendo tablas con múltiples métodos...")
                tables = self.extract_tables(doc)
                
                print("  🖼️ Extrayendo imágenes...")
                base_name = file_path_obj.stem
                images = self._extract_images_from_pdf(doc, base_name, self.config.extracted_subfolders['images'])
                self.stats['total_images'] += len(images)

                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                
                extracted_content_path = self.config.get_folder('extracted_content')
                
                metadata_path = self.config.extracted_subfolders['metadata'] / f"{base_name}_{timestamp}_metadata.json"
                metadata = {
                    'filename': file_name,
                    'timestamp': timestamp,
                    'sections_found': sorted(sections.keys()),
                    'tables_extracted': len(tables),
                    'images_extracted': len(images)
                }
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4)
                print(f"    ✓ Metadatos guardados en {metadata_path}")

                texts_path = self.config.extracted_subfolders['texts'] / f"{base_name}_sections.txt"
                self._save_sections(str(texts_path), sections)
                print(f"    ✓ Contenido de secciones guardado en {texts_path}")

                if tables:
                    tables_folder = self.config.extracted_subfolders['tables']
                    self._save_tables(tables, base_name, tables_folder)
                    print(f"    ✓ Tablas guardadas en {tables_folder}")
                
                if images:
                    print(f"    ✓ Imágenes guardadas en {self.config.extracted_subfolders['images']}")

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                self.stats['successful_extractions'] += 1
                print(f"✅ Procesado exitosamente: {file_name}")
                return {'file': file_name, 'sections': sections, 'tables': tables, 'images': images}

        except Exception as e:
            print(f"❌ Error general procesando: {file_name}. Detalles: {e}")
            self.stats['failed_extractions'] += 1
            return False

def _clean_output_folders(config: ProjectConfig):
    """Elimina todo el contenido de las carpetas de salida antes de una nueva ejecución."""
    print("🧹 Limpiando carpetas de salida...")
    folders_to_clean = [
        config.get_folder('texts'),
        config.get_folder('tables'),
        config.get_folder('images'),
        config.get_folder('metadata')
    ]
    for folder in folders_to_clean:
        if folder.exists():
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
    print("    ✅ Limpieza completada.")


def extract_content_only(raw_data_folder: str, base_project_path: str):
    """
    Función principal para la extracción de contenido de los PDFs.
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

def generate_chunk_for_doc(self, pdf_path: str) -> dict:
        """
        Procesa un PDF completo y devuelve un único chunk (la ficha completa).
        """
        full_text = self.extract_text_from_pdf(pdf_path)
        sections = self._extract_sections_with_regex(full_text)

        if not sections:
            return {
                "doc_id": os.path.basename(pdf_path),
                "text": ""
            }

        # Concatenar secciones en orden
        ordered_sections = [sections[num] for num in sorted(sections.keys())]
        full_doc_text = "\n\n".join(ordered_sections)

        return {
            "doc_id": os.path.basename(pdf_path),
            "text": full_doc_text
        }
    
if __name__ == "__main__":
    RAW_DATA_FOLDER = r"/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad/data/raw_documents"
    BASE_PROJECT_PATH = r"/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad"
    
    print("Iniciando solo la extracción...")
    extracted_documents = extract_content_only(RAW_DATA_FOLDER, BASE_PROJECT_PATH)
    
    print(f"\nExtracción completada. Se procesaron {len(extracted_documents)} documentos exitosamente.")

    # 🚀 Generar un chunk por documento
    extractor = FDSExtractorAdvanced(config=config.py)
    chunks = []

    for doc_path in extracted_documents:
        try:
            chunk = extractor.generate_chunk_for_doc(doc_path)
            chunks.append(chunk)
            print(f"✅ Chunk creado para {chunk['doc_id']}")
        except Exception as e:
            print(f"❌ Error creando chunk para {doc_path}: {e}")

    print(f"\nSe generaron {len(chunks)} chunks (uno por ficha).")


