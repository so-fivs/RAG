from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import json
import re
import pandas as pd
import shutil
import sys

root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

try:
    from s3_sync import sync_vector_db
    S3_DISPONIBLE = True
except ImportError:
    S3_DISPONIBLE = False

from config import ProjectConfig

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
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
    "16": "otra_informacion",
}

PATRON_SECCION = re.compile(r'(?:^|\n)#{1,3}\s*(?:[Ss]ecci[oó]n\s+|SECCI[ÓO]N\s+)?(\d{1,2})\s*[\.:]', re.MULTILINE)

CHUNK_MAX   = 800
CHUNK_MIN   = 200
OVERLAP_PAR = 1  # párrafos de solapamiento al subdividir


# ─────────────────────────────────────────────
# EXTRACCIÓN DE IDENTIDAD
# ─────────────────────────────────────────────
def extraer_identidad(texto_md: str, seccion1: str) -> dict:
    """Extrae nombre comercial, fabricante y CAS principal."""
    identidad = {"nombre_comercial": "", "fabricante": "", "cas_principal": ""}

    # Nombre comercial: primeras 6 líneas no vacías del markdown
    lineas = [l.strip() for l in texto_md.splitlines() if l.strip()]
    for linea in lineas[:6]:
        if not re.search(r'^#|ficha|seguridad|sds|fds', linea, re.IGNORECASE):
            identidad["nombre_comercial"] = linea
            break

    # Fabricante: buscar en sección 1 patrón "fabricante/proveedor : Nombre"
    m = re.search(
        r'(?:fabricante|proveedor)[^:\n]{0,30}[:\-]\s*([A-ZÁÉÍÓÚÑ][^\n]{4,60})',
        seccion1, re.IGNORECASE
    )
    if m:
        identidad["fabricante"] = m.group(1).strip()

    # CAS principal: primer número CAS en sección 3
    seccion3 = ""
    matches = list(PATRON_SECCION.finditer(texto_md))
    for i, match in enumerate(matches):
        if match.group(1) == "3":
            fin = matches[i+1].start() if i+1 < len(matches) else len(texto_md)
            seccion3 = texto_md[match.end():fin]
            break

    cas = re.search(r'\b(\d{2,7}-\d{2}-\d)\b', seccion3)
    if cas:
        identidad["cas_principal"] = cas.group(1)

    return identidad


# ─────────────────────────────────────────────
# METADATA ENRIQUECIDA
# ─────────────────────────────────────────────
def construir_metadata_enriquecida(meta_fds: dict, pdf_name: str, texto_md: str) -> dict:
    """Construye metadata completa desde JSON gold/metadata/ y el markdown."""
    seccion1 = meta_fds.get("secciones", {}).get("identificacion_producto", "")
    identidad = extraer_identidad(texto_md, seccion1)

    # H codes: "H226: Líquido y vapores inflamables | H315: Provoca irritación"
    h_str = " | ".join(
        f"{h.get('codigo','')}: {h.get('descripcion','')}"
        for h in meta_fds.get("h_codes", [])
    )

    # P codes: igual formato
    p_str = " | ".join(
        f"{p.get('codigo','')}: {p.get('descripcion','')}"
        for p in meta_fds.get("p_codes", [])
    )

    # Pictogramas: string limpio
    pictogramas_str = ", ".join(meta_fds.get("pictogramas", []))

    return {
        "producto":        meta_fds.get("nombre_producto", pdf_name),
        "nombre_comercial": identidad["nombre_comercial"],
        "fabricante":      identidad["fabricante"],
        "cas_principal":   identidad["cas_principal"],
        "archivo":         pdf_name,
        "pictogramas":     pictogramas_str,
        "h_codes":         h_str,
        "p_codes":         p_str,
    }


# ─────────────────────────────────────────────
# CHUNKING SEMÁNTICO POR SECCIÓN
# ─────────────────────────────────────────────
def subdividir_por_parrafos(texto: str, nombre_seccion: str,
                             metadata_base: dict) -> list[Document]:
    """
    Si un bloque de sección supera CHUNK_MAX, lo divide por párrafos completos
    con solapamiento de OVERLAP_PAR párrafos. Nunca corta dentro de un párrafo.
    """
    parrafos = []
    bloque_actual = []
    for linea in texto.splitlines():
        if linea.strip():
            bloque_actual.append(linea.strip())
        else:
            if bloque_actual:
                parrafos.append("\n".join(bloque_actual))
                bloque_actual = []
    if bloque_actual:
        parrafos.append("\n".join(bloque_actual))

    if not parrafos:
        return []

    # Si cabe completo, devolver como un solo chunk
    if len(texto) <= CHUNK_MAX:
        if len(texto) >= CHUNK_MIN:
            meta = {**metadata_base, "seccion_fds": nombre_seccion, "tipo_contenido": "texto"}
            return [Document(page_content=texto.strip(), metadata=meta)]
        return []

    # Subdividir acumulando párrafos hasta llenar CHUNK_MAX
    chunks = []
    buffer = []
    buffer_len = 0

    for par in parrafos:
        if buffer_len + len(par) > CHUNK_MAX and buffer:
            contenido = "\n\n".join(buffer)
            if len(contenido) >= CHUNK_MIN:
                meta = {**metadata_base, "seccion_fds": nombre_seccion, "tipo_contenido": "texto"}
                chunks.append(Document(page_content=contenido, metadata=meta))
            # Solapamiento: mantener últimos OVERLAP_PAR párrafos
            buffer = buffer[-OVERLAP_PAR:] if OVERLAP_PAR else []
            buffer_len = sum(len(p) for p in buffer)

        buffer.append(par)
        buffer_len += len(par)

    # Último buffer
    if buffer:
        contenido = "\n\n".join(buffer)
        if len(contenido) >= CHUNK_MIN:
            meta = {**metadata_base, "seccion_fds": nombre_seccion, "tipo_contenido": "texto"}
            chunks.append(Document(page_content=contenido, metadata=meta))

    return chunks


def chunking_semantico_por_secciones(texto_md: str,
                                      metadata_base: dict) -> list[Document]:
    """
    Divide el markdown exactamente en las 16 secciones FDS.
    Cada sección es un bloque semántico completo.
    Si supera CHUNK_MAX se subdivide por párrafos completos.
    Descarta fragmentos menores a CHUNK_MIN (ruido residual).
    """
    matches = list(PATRON_SECCION.finditer(texto_md))
    chunks = []

    # Texto antes de la primera sección (encabezado del documento)
    if matches and matches[0].start() > 0:
        preambulo = texto_md[:matches[0].start()].strip()
        if len(preambulo) >= CHUNK_MIN:
            meta = {**metadata_base, "seccion_fds": "encabezado", "tipo_contenido": "texto"}
            chunks.append(Document(page_content=preambulo, metadata=meta))

    for i, match in enumerate(matches):
        num = match.group(1)
        nombre_seccion = NOMBRES_SECCIONES.get(num, f"seccion_{num}")

        inicio = match.end()
        fin = matches[i+1].start() if i+1 < len(matches) else len(texto_md)
        contenido = texto_md[inicio:fin].strip()

        if not contenido:
            continue

        sub_chunks = subdividir_por_parrafos(contenido, nombre_seccion, metadata_base)
        chunks.extend(sub_chunks)

    return chunks


# ─────────────────────────────────────────────
# PERSISTENCIA SILVER
# ─────────────────────────────────────────────
def guardar_silver(chunks: list[Document], config, nombre_base: str):
    silver_path = config.get_folder('processed_chunks') / f"{nombre_base}_chunks.json"
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    silver_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


# ─────────────────────────────────────────────
# TABLAS
# ─────────────────────────────────────────────
def procesar_tablas_como_docs(config, nombre_base: str,
                               metadata_base: dict) -> list[Document]:
    carpeta_tablas = config.get_folder('tables')
    archivos_tabla = list(carpeta_tablas.glob(f"{nombre_base}_tabla_*.csv"))

    docs_tablas = []
    for path_csv in archivos_tabla:
        try:
            df = pd.read_csv(path_csv)
            tabla_texto = df.to_markdown(index=False)
            if not tabla_texto or len(tabla_texto) < CHUNK_MIN:
                continue
            meta = {**metadata_base, "tipo_contenido": "tabla", "archivo_csv": path_csv.name}
            docs_tablas.append(Document(page_content=tabla_texto, metadata=meta))
        except Exception as e:
            print(f"    [WARN] Tabla {path_csv.name} omitida: {e}")

    return docs_tablas


# ─────────────────────────────────────────────
# LIMPIEZA Y VECTOR DB
# ─────────────────────────────────────────────
def limpiar_para_reingesta(config):
    """Elimina vector_db y silver/chunked para evitar duplicados. Preserva gold/."""
    db_path = Path(str(config.get_folder('vector_db')))
    silver_path = Path(str(config.get_folder('processed_chunks')))

    for path in [db_path, silver_path]:
        if path.exists():
            shutil.rmtree(path)
            print(f"  Limpiado: {path}")
        path.mkdir(parents=True, exist_ok=True)

    print("  gold/ preservado intacto.\n")


def ingestar_por_tipo(chunks: list[Document], config):
    print("  Conectando a Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    db_path = str(config.get_folder('vector_db'))

    # 1. Separar por tipo de contenido primero
    chunks_texto = [c for c in chunks if c.metadata.get("tipo_contenido") == "texto"]
    chunks_tabla = [c for c in chunks if c.metadata.get("tipo_contenido") == "tabla"]

    # 2. Truncar contenido (Función interna para mantener limpieza)
    def truncar_chunks(lista_chunks, max_chars=1500):
        for c in lista_chunks:
            if len(c.page_content) > max_chars:
                c.page_content = c.page_content[:max_chars]
        return lista_chunks

    # Aplicar truncado
    chunks_texto = truncar_chunks(chunks_texto)
    chunks_tabla = truncar_chunks(chunks_tabla)

    # 3. Ingestión en Chroma
    if chunks_texto:
        print(f"  -> {len(chunks_texto)} chunks de texto a fds_textos...")
        Chroma.from_documents(
            documents=chunks_texto,
            embedding=embeddings,
            collection_name="fds_textos",
            persist_directory=db_path
        )

    if chunks_tabla:
        print(f"  -> {len(chunks_tabla)} tablas a fds_tablas...")
        Chroma.from_documents(
            documents=chunks_tabla,
            embedding=embeddings,
            collection_name="fds_tablas",
            persist_directory=db_path
        )

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def pipeline_chunking_e_ingesta(config):
    """
    Lee gold/ ya procesado por Docling.
    Chunking semántico por sección FDS + ingesta a ChromaDB.
    Elimina vector_db y silver/chunked antes de correr (evita duplicados).
    NO toca gold/.
    """
    pdfs = list(config.get_folder('bronze').glob("*.pdf"))

    if not pdfs:
        print("No hay PDFs en data/bronze/")
        return

    print(f"Procesando {len(pdfs)} PDFs (sin re-extracción)...\n")

    # Limpiar duplicados antes de empezar
    limpiar_para_reingesta(config)

    todos_los_chunks = []

    for i, pdf_path in enumerate(pdfs):
        nombre_base = pdf_path.stem
        print(f"[{i+1}/{len(pdfs)}] {pdf_path.name}")

        meta_path = config.get_folder('metadata') / f"{nombre_base}.json"
        if not meta_path.exists():
            print(f"  Sin metadata, saltando...")
            continue

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_fds = json.load(f)
        except Exception as e:
            print(f"  Error leyendo metadata: {e}")
            continue

        md_path = config.get_folder('texts') / f"{nombre_base}.md"
        if not md_path.exists():
            print(f"  Sin markdown en gold/texts/, saltando...")
            continue

        texto_md = md_path.read_text(encoding='utf-8')

        # Metadata enriquecida
        metadata_base = construir_metadata_enriquecida(meta_fds, pdf_path.name, texto_md)

        # Chunking semántico por las 16 secciones
        try:
            chunks_txt = chunking_semantico_por_secciones(texto_md, metadata_base)
            guardar_silver(chunks_txt, config, nombre_base)
            todos_los_chunks.extend(chunks_txt)
            print(f"  {len(chunks_txt)} chunks de texto")
        except Exception as e:
            print(f"  Error chunking: {e}")

        # Tablas
        try:
            chunks_tab = procesar_tablas_como_docs(config, nombre_base, metadata_base)
            if chunks_tab:
                todos_los_chunks.extend(chunks_tab)
                print(f"  {len(chunks_tab)} tablas")
        except Exception as e:
            print(f"  Error tablas: {e}")

    if not todos_los_chunks:
        print("\nNo hay chunks. Verifica que gold/ tenga datos.")
        return

    print(f"\nTotal: {len(todos_los_chunks)} chunks para ingesta.\n")
    ingestar_por_tipo(todos_los_chunks, config)
    print("\nProceso terminado. Colecciones fds_textos y fds_tablas actualizadas.")
    if S3_DISPONIBLE:
        print("\n☁️  Subiendo vector_db/ a S3...")
        sync_vector_db(config, direccion="push")



if __name__ == "__main__":
    config = ProjectConfig()
    config.verify_structure()
    pipeline_chunking_e_ingesta(config)