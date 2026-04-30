import json
import shutil
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List
from docling.document_converter import DocumentConverter
import fitz
import sys

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    from src.config import ProjectConfig
except ImportError:
    from config import ProjectConfig


def limpiar_carpetas_salida(config):
    """Elimina y recrea las carpetas gold/ para evitar datos duplicados en re-ejecuciones."""
    carpetas = ['texts', 'tables', 'images', 'metadata']
    for carpeta in carpetas:
        folder_path = config.get_folder(carpeta)
        if folder_path.exists():
            shutil.rmtree(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
    print("  Carpetas gold/ limpiadas y recreadas.")


def procesar_con_docling(pdf_path: str, config):
    nombre_base = Path(pdf_path).stem
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    texto_md = result.document.export_to_markdown()

    tablas_dfs = []
    carpeta_tablas = config.get_folder('tables')
    carpeta_tablas.mkdir(parents=True, exist_ok=True)

    for i, table in enumerate(result.document.tables):
        df = table.export_to_dataframe(doc=result.document)

        # Descartar tablas con headers numéricos (tablas mal parseadas por Docling)
        headers_son_numericos = all(str(c).strip().isdigit() for c in df.columns)
        if headers_son_numericos:
            continue

        if df.shape[0] >= 2 and df.shape[1] >= 2:
            # Limpiar ':' sobrantes del layout de dos columnas del PDF
            df = df.apply(lambda col: col.map(
                lambda x: str(x).rstrip(' :').strip() if isinstance(x, str) else x
            ))
            df.to_csv(carpeta_tablas / f"{nombre_base}_tabla_{i}.csv", index=False)
            tablas_dfs.append(df)

    return texto_md, tablas_dfs


def extraer_metadata_fds(texto_md: str, nombre_base: str) -> Dict:
    NOMBRES_SECCIONES = {
        "1":  "identificacion_producto",
        "2":  "identificacion_peligros",
        "3":  "composicion_componentes",
        "4":  "primeros_auxilios",
        "5":  "medidas_contra_incendios",
        "6":  "medidas_vertido_accidental",
        "7":  "manipulacion_almacenamiento",
        "8":  "controles_exposicion_proteccion",
        "9":  "propiedades_fisicas_quimicas",
        "10": "estabilidad_reactividad",
        "11": "informacion_toxicologica",
        "12": "informacion_ecologica",
        "13": "consideraciones_eliminacion",
        "14": "informacion_transporte",
        "15": "informacion_reglamentaria",
        "16": "otra_informacion"
    }

    PIGTO_PATRONES = {
        "Inflamable":    r"l[íi]quidos?\s+inflamables?|inflamable",
        "Corrosivo":     r"corrosivo|corrosion\s+cut",
        "Toxicidad":     r"toxicidad\s+aguda",
        "Irritante":     r"irritaci[oó]n\s+cut[aá]nea|irritaci[oó]n\s+ocular",
        "Peligro Salud": r"carcinogenicidad|mutagenicidad|sensibilizaci[oó]n",
        "Gas Presión":   r"gas\s+a\s+presi[oó]n",
    }

    metadata = {
        "nombre_producto": nombre_base,
        "pictogramas": [],
        "h_codes": [],
        "p_codes": [],
        "secciones": {}
    }

    for nombre, patron in PIGTO_PATRONES.items():
        if re.search(patron, texto_md, re.IGNORECASE):
            metadata["pictogramas"].append(nombre)

    # Procesamiento línea a línea: evita que el regex H consuma líneas P
    patron_h = re.compile(r'^(H\d{3})\s*[:\-]?\s*(.+)$', re.IGNORECASE)
    patron_p = re.compile(r'^(P\d{3}(?:\s*\+\s*P\d{3})*)\s*[:\-]?\s*(.+)$', re.IGNORECASE)

    for linea in texto_md.splitlines():
        linea = linea.strip()
        m = patron_h.match(linea)
        if m and m.group(2).strip():
            metadata["h_codes"].append({
                "codigo": m.group(1).upper(),
                "descripcion": m.group(2).strip()
            })
            continue
        m = patron_p.match(linea)
        if m and m.group(2).strip():
            metadata["p_codes"].append({
                "codigo": m.group(1).upper().replace(' ', ''),
                "descripcion": m.group(2).strip()
            })

    patron_seccion = r'(?:^|\n)#*\s*(?:SECCI[ÓO]N|SECTION)\s+(\d{1,2})\s*[:\.]?'
    matches = list(re.finditer(patron_seccion, texto_md, re.IGNORECASE))

    for i, match in enumerate(matches):
        num = match.group(1)
        inicio = match.end()
        fin = matches[i + 1].start() if i + 1 < len(matches) else len(texto_md)
        contenido = texto_md[inicio:fin].strip()
        metadata["secciones"][NOMBRES_SECCIONES.get(num, f"seccion_{num}")] = contenido

    return metadata


def extraer_imagenes_soporte(pdf_path: str, config, nombre_base: str):
    doc = fitz.open(pdf_path)
    carpeta = config.get_folder('images')
    carpeta.mkdir(parents=True, exist_ok=True)

    for i, pag in enumerate(doc):
        for j, img in enumerate(pag.get_images()):
            try:
                pix = fitz.Pixmap(doc, img[0])
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                elif pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
                pix.save(carpeta / f"{nombre_base}_p{i}_img{j}.png")
            except Exception as e:
                print(f"  [WARN] Imagen {nombre_base}_p{i}_img{j} omitida: {e}")
    doc.close()


def pipeline_fds_docling(pdf_path: str, config, primera_ejecucion: bool = False):
    nombre_base = Path(pdf_path).stem

    # Limpia en la primera ejecución del lote para no duplicar datos
    if primera_ejecucion:
        limpiar_carpetas_salida(config)

    texto_md, tablas = procesar_con_docling(pdf_path, config)
    metadata = extraer_metadata_fds(texto_md, nombre_base)

    carpeta_texts = config.get_folder('texts')
    carpeta_texts.mkdir(parents=True, exist_ok=True)
    (carpeta_texts / f"{nombre_base}.md").write_text(texto_md, encoding='utf-8')

    carpeta_meta = config.get_folder('metadata')
    carpeta_meta.mkdir(parents=True, exist_ok=True)
    with open(carpeta_meta / f"{nombre_base}.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    extraer_imagenes_soporte(pdf_path, config, nombre_base)

    return {
        "tablas":  len(tablas),
        "h_codes": len(metadata["h_codes"]),
        "p_codes": len(metadata["p_codes"]),
    }


if __name__ == "__main__":
    from config import ProjectConfig
    config = ProjectConfig()
    archivos = list(config.get_folder('bronze').glob("*.pdf"))

    if not archivos:
        print("No se encontraron PDFs en data/bronze/")
    else:
        print(f"Procesando {len(archivos)} PDFs...\n")
        for i, path in enumerate(archivos):
            print(f"[{i+1}/{len(archivos)}] {path.name}")
            try:
                res = pipeline_fds_docling(str(path), config, primera_ejecucion=(i == 0))
                print(f"  Listo: {res['tablas']} tablas | {res['h_codes']} H codes | {res['p_codes']} P codes\n")
            except Exception as e:
                print(f"  ERROR: {e}\n")