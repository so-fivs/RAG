import json
import shutil
from pathlib import Path
from typing import Dict, List
import fitz
import camelot
import re
import pandas as pd
import pdfplumber
from pathlib import Path
from typing import List
import hashlib
import config as ProjectConfig



# ============================================================================
# CONFIGURACIÓN Y LIMPIEZA
# ============================================================================

def limpiar_carpetas_salida(config):
    """Elimina todo el contenido de las carpetas de salida."""
    carpetas = ['tables', 'texts', 'images', 'metadata']
    for carpeta in carpetas:
        folder_path = config.get_folder(carpeta)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

   
# ============================================================================
# EXTRACCIÓN DE TABLAS
# ============================================================================

def extraer_y_limpiar_tablas(pdf_path: str, config) -> List[pd.DataFrame]:
    """Extrae tablas con validación y deduplicación mejorada usando múltiples métodos."""
    nombre_base = Path(pdf_path).stem
    carpeta_tablas = config.get_folder('tables')
    tablas_validas = []
    hashes_vistos = set()
    
    def hash_tabla(df: pd.DataFrame) -> str:
        """Genera hash único de tabla para detectar duplicados."""
        contenido = df.fillna('').astype(str).values.tobytes()
        return hashlib.md5(contenido).hexdigest()
    
    def limpiar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza agresiva de tabla."""
        # Eliminar filas/columnas completamente vacías
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Limpiar strings y saltos de línea internos (CRÍTICO)
        df = df.map(lambda x: ' '.join(str(x).split()) if pd.notna(x) else x)
        df = df.replace(['', '-', 'nan', 'None', 'null'], None)
        
        # Eliminar filas que son títulos de sección (ej: "SECCIÓN 14: ...")
        df = df[~df.iloc[:, 0].astype(str).str.contains('SECCI[ÓO]N \d+', case=False, na=False)]
        
        # Eliminar filas con solo números (0,1,2,3)
        primera_col = df.iloc[:, 0].astype(str)
        df = df[~primera_col.str.match(r'^\d+$', na=False)]
        
        # Eliminar columnas que son >90% vacías
        umbral = len(df) * 0.1
        df = df.dropna(thresh=umbral, axis=1)
        
        # Detectar y usar primera fila como header si tiene sentido
        if len(df) > 1:
            primera = df.iloc[0]
            # Si la primera fila tiene >60% valores no nulos, usarla como header
            if primera.notna().sum() >= len(df.columns) * 0.6:
                # Verificar que no sea una fila de datos numéricos puros
                if not all(str(x).replace('.','').isdigit() for x in primera.dropna()):
                    df.columns = [str(c).strip() if pd.notna(c) else f"Col_{i}" 
                                 for i, c in enumerate(primera)]
                    df = df.iloc[1:].reset_index(drop=True)
        
        # Renombrar primera columna si es genérica
        if df.columns[0] in ['0', 'Col_0', '']:
            df.columns = ['Concepto'] + list(df.columns[1:])
        
        # Renombrar columnas duplicadas
        cols = []
        seen = {}
        for col in df.columns:
            col_str = str(col)
            if col_str in seen:
                seen[col_str] += 1
                cols.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                cols.append(col_str)
        df.columns = cols
        
        return df.reset_index(drop=True)
    
    def validar_tabla(df: pd.DataFrame) -> bool:
        """Valida si tabla es útil."""
        if df.shape[0] < 2 or df.shape[1] < 2:
            return False
        total = df.shape[0] * df.shape[1]
        vacias = df.isna().sum().sum()
        return (vacias / total) < 0.75
    
    # MÉTODO 1: Camelot Lattice (tablas con bordes)
    try:
        for tabla in camelot.read_pdf(pdf_path, pages='all', flavor='lattice', 
                                      line_scale=40, shift_text=['l', 't']):
            df = limpiar_dataframe(tabla.df)
            if validar_tabla(df):
                h = hash_tabla(df)
                if h not in hashes_vistos:
                    hashes_vistos.add(h)
                    tablas_validas.append(df)
    except:
        pass
    
    # MÉTODO 2: PDFPlumber (mejor para tablas sin bordes claros)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for tabla in page.extract_tables():
                    if not tabla or len(tabla) < 2:
                        continue
                    df = pd.DataFrame(tabla[1:], columns=tabla[0])
                    df = limpiar_dataframe(df)
                    if validar_tabla(df):
                        h = hash_tabla(df)
                        if h not in hashes_vistos:
                            hashes_vistos.add(h)
                            tablas_validas.append(df)
    except:
        pass
    
    # MÉTODO 3: Camelot Stream (fallback para tablas complejas)
    if not tablas_validas:
        try:
            for tabla in camelot.read_pdf(pdf_path, pages='all', flavor='stream',
                                         edge_tol=50, row_tol=10):
                df = limpiar_dataframe(tabla.df)
                if validar_tabla(df):
                    h = hash_tabla(df)
                    if h not in hashes_vistos:
                        hashes_vistos.add(h)
                        tablas_validas.append(df)
        except:
            pass
    
    # Guardar CSVs
    for i, df in enumerate(tablas_validas):
        output_csv = carpeta_tablas / f"{nombre_base}_tabla_{i}.csv"
        df.to_csv(output_csv, index=False, encoding='utf-8')
    
    return tablas_validas
# ============================================================================
# EXTRACCIÓN Y PROCESAMIENTO DE TEXTO
# ============================================================================

def extraer_metadatos_y_secciones(texto: str) -> Dict:
    """Extrae metadata completa y secciones del documento."""
    
    # Mapeo de número a nombre de sección
    NOMBRES_SECCIONES = {
        "1": "identificacion_producto",
        "2": "identificacion_peligros",
        "3": "composicion_componentes",
        "4": "primeros_auxilios",
        "5": "medidas_contra_incendios",
        "6": "medidas_vertido_accidental",
        "7": "manipulacion_almacenamiento",
        "8": "controles_exposicion_proteccion",
        "9": "propiedades_fisicas_quimicas",
        "10": "estabilidad_reactividad",
        "11": "informacion_toxicologica",
        "12": "informacion_ecologica",
        "13": "consideraciones_eliminacion",
        "14": "informacion_transporte",
        "15": "informacion_reglamentaria",
        "16": "otra_informacion"
    }
    
    SECCIONES_PATRONES = {
        "1": [r"identificaci[oó]n(?:\s+del?\s+producto)?", r"product\s+identifier", r"manufacturer", r"supplier", r"fabricante", r"proveedor"],
        "2": [r"identificaci[oó]n.*peligros", r"hazard.*identification", r"clasificaci[oó]n.*peligro", r"pictogramas?", r"signal.*word"],
        "3": [r"composici[oó]n", r"ingredientes", r"componentes", r"composition", r"ingredients", r"components"],
        "4": [r"primeros\s+auxilios", r"first\s+aid", r"first.*aid.*measures"],
        "5": [r"medidas.*incendio", r"lucha.*incendio", r"fire.*fighting", r"firefighting"],
        "6": [r"medidas.*derrame", r"liberación.*accidental", r"vertido", r"accidental.*release", r"spill"],
        "7": [r"manejo.*almacenamiento", r"manipulaci[oó]n.*almacenamiento", r"handling.*storage"],
        "8": [r"controles.*exposici[oó]n", r"protecci[oó]n.*personal", r"exposure.*controls", r"personal.*protection"],
        "9": [r"propiedades.*f[ií]sicas.*qu[ií]micas", r"physical.*chemical.*properties"],
        "10": [r"estabilidad.*reactividad", r"stability.*reactivity"],
        "11": [r"informaci[oó]n.*toxicol[oó]gica", r"toxicological.*information"],
        "12": [r"informaci[oó]n.*ecol[oó]gica", r"ecological.*information"],
        "13": [r"eliminaci[oó]n", r"disposal", r"consideraciones.*eliminaci[oó]n"],
        "14": [r"transporte", r"informaci[oó]n.*transporte", r"transportation"],
        "15": [r"informaci[oó]n.*reglamentaria", r"regulatory.*information"],
        "16": [r"otra.*informaci[oó]n", r"other.*information"]
    }
    
    
    
    metadata = {
        "nombre_producto": None,
        "proveedor": None,
        "codigo": None,
        "fecha_emision": None,
        "secciones": {}
    }
    
    lineas = texto.split('\n')[:50]
    
    for i, linea in enumerate(lineas):
        if re.search(r'nombre\s+del\s+producto|product\s+name', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1].strip()
                if 3 < len(candidato) < 200:
                    metadata["nombre_producto"] = candidato
                    break
    
    for i, linea in enumerate(lineas):
        if re.search(r'fabricante|distribuidor|manufacturer|supplier', linea, re.IGNORECASE):
            if i + 1 < len(lineas):
                candidato = lineas[i + 1].strip()
                if 3 < len(candidato) < 200:
                    metadata["proveedor"] = candidato
                    break
    
    match_codigo = re.search(r'c[óo]digo[:\s]+(\S+)', texto[:2000], re.IGNORECASE)
    if match_codigo:
        metadata["codigo"] = match_codigo.group(1).strip()
    
    match_fecha = re.search(r'fecha\s+de\s+emisi[óo]n[:\s]+([\d./-]+)|revision\s+date[:\s]+([\d./-]+)', texto[:2000], re.IGNORECASE)
    if match_fecha:
        metadata["fecha_emision"] = (match_fecha.group(1) or match_fecha.group(2)).strip()    
    # Uso del producto
    match_uso = re.search(r'uso\s+del\s+producto[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)', texto[:3000], re.IGNORECASE | re.DOTALL)
    if match_uso:
        metadata["uso_producto"] = match_uso.group(1).strip()
    
     # Frases H y P
    metadata["indicaciones_peligro"] = list(set(re.findall(r'H\d{3}\b', texto)))
    metadata["frases_precaucion"] = list(set(re.findall(r'P\d{3}(?:\s*\+\s*P\d{3})*', texto)))
    
    # AGREGAR ESTO:
    # Extraer descripciones completas de H y P
    metadata["indicaciones_peligro_detalle"] = []
    for codigo_h in metadata["indicaciones_peligro"]:
        match = re.search(rf'{codigo_h}\s+(.*?)(?:\n|H\d{{3}}|$)', texto, re.IGNORECASE)
        if match:
            metadata["indicaciones_peligro_detalle"].append({
                "codigo": codigo_h,
                "descripcion": match.group(1).strip()
            })
    
    metadata["frases_precaucion_detalle"] = []
    for codigo_p in metadata["frases_precaucion"]:
        match = re.search(rf'{codigo_p}\s+(.*?)(?:\n|P\d{{3}}|$)', texto, re.IGNORECASE)
        if match:
            metadata["frases_precaucion_detalle"].append({
                "codigo": codigo_p,
                "descripcion": match.group(1).strip()
            })
    
    # Componentes químicos
    metadata["componentes_quimicos"] = []
    lineas_cas = re.findall(r'(.*?)\s+CAS:\s*(\d+-\d+-\d+)(?:\s+([\d,.-]+%?\s*-?\s*[\d,.-]*%?))?', texto)
    for nombre, cas, concentracion in lineas_cas:
        comp = {"nombre": nombre.strip(), "cas": cas.strip()}
        if concentracion:
            comp["concentracion"] = concentracion.strip()
        metadata["componentes_quimicos"].append(comp)
    
    # Propiedades físico-químicas
    metadata["propiedades_fisicoquimicas"] = {}
    props = {
        "estado_fisico": r'Estado\s+f[íi]sico[:\s]+(.*?)(?:\n|$)',
        "color": r'Color[:\s]+(.*?)(?:\n|$)',
        "ph": r'pH[:\s]+([\d.,\s±]+)',
        "punto_inflamacion": r'Punto\s+de\s+inflamaci[óo]n[:\s]+(.*?)(?:\n|$)',
        "densidad": r'Densidad(?:\s+aparente)?[:\s]+([\d.,\s±]+\s*g/cm3)'
    }
    for key, patron in props.items():
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            metadata["propiedades_fisicoquimicas"][key] = match.group(1).strip()
    
    # Continúa con la extracción de secciones...
    patron_seccion = r'(?:^|\n)\s*(?:SECCI[ÓO]N|SECTION)\s+(\d{1,2})\s*:'
    matches = list(re.finditer(patron_seccion, texto, re.IGNORECASE | re.MULTILINE))
    
    print("\n  Secciones detectadas:")
    
    if matches:
        for i, match in enumerate(matches):
            num_seccion = match.group(1).strip()
            inicio = match.end()
            fin = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
            
            contenido = texto[inicio:fin].strip()
            
            nombre_seccion = NOMBRES_SECCIONES.get(num_seccion, f"seccion_{num_seccion}")
            
            if nombre_seccion not in metadata["secciones"] and len(contenido) > 50:
                metadata["secciones"][nombre_seccion] = contenido
                print(f"    {nombre_seccion}: {len(contenido)} caracteres")
    
    if not metadata["secciones"]:
        print("    Usando detección por palabras clave...")
        
        for num_seccion, patrones in SECCIONES_PATRONES.items():
            for patron in patrones:
                matches_patron = list(re.finditer(rf'(?:^|\n)\s*{patron}', texto, re.IGNORECASE))
                
                if matches_patron:
                    inicio = matches_patron[0].start()
                    
                    fin = len(texto)
                    for otra_num in SECCIONES_PATRONES.keys():
                        if otra_num != num_seccion:
                            for otro_patron in SECCIONES_PATRONES[otra_num]:
                                siguiente = re.search(rf'(?:^|\n)\s*{otro_patron}', texto[inicio + 200:], re.IGNORECASE)
                                if siguiente:
                                    posible_fin = inicio + 200 + siguiente.start()
                                    if posible_fin < fin:
                                        fin = posible_fin
                    
                    contenido = texto[inicio:fin].strip()
                    
                    nombre_seccion = NOMBRES_SECCIONES.get(num_seccion, f"seccion_{num_seccion}")
                    
                    if nombre_seccion not in metadata["secciones"] and len(contenido) > 100:
                        metadata["secciones"][nombre_seccion] = contenido
                        print(f"    {nombre_seccion}: {len(contenido)} caracteres")
                        break
    
    if not metadata["secciones"]:
        print("    No se identificaron secciones estructuradas")
    
    return metadata

def procesar_texto(pdf_path: str, config, nombre_base: str) -> Dict:
    """Procesa texto y genera estructura JSON con metadata."""
    doc = fitz.open(pdf_path)
    texto_completo = "\n\n".join([pagina.get_text() for pagina in doc])
    doc.close()
    
    texto_limpio = re.sub(r'[ \t]+', ' ', texto_completo)
    texto_limpio = re.sub(r'\n{3,}', '\n\n', texto_limpio)
    texto_limpio = re.sub(r'^\s+', '', texto_limpio, flags=re.MULTILINE).strip()
    
    metadata = extraer_metadatos_y_secciones(texto_limpio)
    
    resultado = {
        "contenido_texto": texto_limpio,
        "metadata": metadata
    }
    
    carpeta_textos = config.get_folder('texts')
    carpeta_metadata = config.get_folder('metadata')
    
    output_txt = carpeta_textos / f"{nombre_base}_texto_limpio.txt"
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(texto_limpio)
    
    output_json = carpeta_metadata / f"{nombre_base}_metadata.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    return resultado


# ============================================================================
# EXTRACCIÓN DE IMÁGENES
# ============================================================================

def extraer_imagenes(pdf_path: str, config, nombre_base: str) -> int:
    """Extrae todas las imágenes del PDF."""
    doc = fitz.open(pdf_path)
    carpeta_imgs = config.get_folder('images')
    
    contador = 0
    for num_pag, pagina in enumerate(doc):
        for idx_img, img in enumerate(pagina.get_images()):
            xref = img[0]
            base_img = doc.extract_image(xref)
            output_path = carpeta_imgs / f"{nombre_base}_p{num_pag}_img{idx_img}.{base_img['ext']}"
            with open(output_path, 'wb') as f:
                f.write(base_img["image"])
            contador += 1
    
    doc.close()
    return contador


# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def pipeline_completo(pdf_path: str, config: ProjectConfig, primera_vez: bool = False):
    """Ejecuta pipeline completo de extracción."""
    nombre_base = Path(pdf_path).stem
    if primera_vez:
        config.create_folders()
        limpiar_carpetas_salida(config)
    
    resultado_texto = procesar_texto(pdf_path, config, nombre_base)
    tablas_limpias = extraer_y_limpiar_tablas(pdf_path, config)
    num_imgs = extraer_imagenes(pdf_path, config, nombre_base)
    
    return {
        'texto': resultado_texto,
        'num_tablas': len(tablas_limpias),
        'num_imagenes': num_imgs
    }


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    carpeta_pdfs = Path("raw_documents")
    archivos_pdf = list(carpeta_pdfs.glob("*.pdf"))
    config = ProjectConfig()
    carpeta_pdfs = config.get_folder('raw_documents')
    archivos_pdf = list(carpeta_pdfs.glob("*.pdf"))
    
    if not archivos_pdf:
        print("No se encontraron PDFs")
    else:
        print(f"Procesando {len(archivos_pdf)} archivos PDF\n")
        
        for i, pdf_path in enumerate(archivos_pdf, 1):
            print(f"[{i}/{len(archivos_pdf)}] {pdf_path.name}")
            
            try:
                resultados = pipeline_completo(
                    pdf_path=str(pdf_path),
                    ruta_config="/Users/sofiavelandiasierra/Documents/rag-fichas-seguridad/config.py",
                    primera_vez=(i == 1)
                )
                
                print(f"  Texto: {len(resultados['texto']['contenido_texto'])} chars")
                print(f"  Tablas: {resultados['num_tablas']}")
                print(f"  Imagenes: {resultados['num_imagenes']}\n")
                
            except Exception as e:
                print(f"  Error: {e}\n")
                continue
        
        print("Procesamiento completado")