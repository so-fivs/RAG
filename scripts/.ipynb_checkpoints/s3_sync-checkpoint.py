"""
s3_sync.py — Sincroniza datos del proyecto RAG con S3.
Las rutas locales se leen desde config.py — es la unica fuente de verdad.

Cualquier cuenta con IAM role:
    python3 s3_sync.py pull       <- descarga vector_db/ desde S3
    python3 s3_sync.py pull-all   <- descarga todo desde S3
"""

import sys
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

# Agregar project_root al path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
BUCKET_NAME = "rag-fds-data"
AWS_REGION  = "us-east-1"
BUCKET_URL  = f"https://{BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com"

# Mapeo nombre_config -> prefijo S3
# Las rutas locales se resuelven via config.get_folder(nombre_config)
SYNC_MAP = {
    'bronze':           'bronze/',
    'texts':            'gold/texts/',
    'tables':           'gold/tables/',
    'images':           'gold/images/',
    'metadata':         'gold/metadata/',
    'processed_chunks': 'silver/chunked/',
    'vector_db':        'vector_db/',
}

# Lo minimo para correr el RAG sin reprocesar
PULL_DEFAULT = ['vector_db/']


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE BOTO3
# ─────────────────────────────────────────────────────────────────────────────
def get_client():
    import boto3
    return boto3.client("s3", region_name=AWS_REGION)


# ─────────────────────────────────────────────────────────────────────────────
# SETUP — owner, primera vez
# ─────────────────────────────────────────────────────────────────────────────
def setup_bucket():
    from botocore.exceptions import ClientError
    s3 = get_client()

    print(f"\n{'='*60}")
    print(f"SETUP S3 — bucket: {BUCKET_NAME} | region: {AWS_REGION}")
    print(f"{'='*60}\n")

    try:
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )
        print(f"  Bucket creado: s3://{BUCKET_NAME}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  Bucket ya existe: s3://{BUCKET_NAME}")
        else:
            print(f"  Error: {e}")
            sys.exit(1)

    for prefijo in sorted(set(SYNC_MAP.values())):
        s3.put_object(Bucket=BUCKET_NAME, Key=prefijo)
        print(f"  Prefijo creado: {prefijo}")

    print(f"\n  Bucket listo: s3://{BUCKET_NAME}")
    print(f"  Recuerda abrir permisos de lectura en AWS Console → S3 → Permissions.\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUSH — owner
# ─────────────────────────────────────────────────────────────────────────────
def push(carpetas: list = None):
    from botocore.exceptions import ClientError
    s3     = get_client()
    config = ProjectConfig()
    mapa   = {k: v for k, v in SYNC_MAP.items()
              if carpetas is None or v in carpetas}

    print(f"\n{'='*60}")
    print("PUSH -> S3")
    print(f"{'='*60}\n")

    total_subidos = 0
    total_bytes   = 0

    for nombre_config, prefijo_s3 in mapa.items():
        # Ruta local viene de config — unica fuente de verdad
        ruta_local = config.get_folder(nombre_config)

        if not ruta_local.exists():
            print(f"  No existe localmente: {ruta_local}")
            continue

        archivos = [f for f in ruta_local.rglob("*") if f.is_file()]
        if not archivos:
            print(f"  Vacio: {nombre_config}")
            continue

        print(f"  {ruta_local} -> s3://{BUCKET_NAME}/{prefijo_s3}")
        for archivo in archivos:
            key = prefijo_s3 + str(archivo.relative_to(ruta_local))
            try:
                s3.upload_file(str(archivo), BUCKET_NAME, key)
                total_bytes   += archivo.stat().st_size
                total_subidos += 1
            except ClientError as e:
                print(f"    Error {archivo.name}: {e}")
        print(f"    {len(archivos)} archivos subidos")

    print(f"\n  Push completo — {total_subidos} archivos | {total_bytes / 1_048_576:.1f} MB\n")


# ─────────────────────────────────────────────────────────────────────────────
# PULL — cualquier cuenta con IAM role
# ─────────────────────────────────────────────────────────────────────────────
def _listar_objetos(prefijo: str) -> list:
    """
    Lista objetos bajo un prefijo S3.
    Usa boto3 con IAM role — funciona en cualquier cuenta
    siempre que el bucket permita acceso de lectura.
    """
    from botocore.exceptions import ClientError
    s3    = get_client()
    keys  = []

    try:
        pag = s3.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=BUCKET_NAME, Prefix=prefijo):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
                    keys.append(key)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "AccessDenied":
            print(f"  Error 403 en {prefijo} — verifica permisos del bucket.")
        else:
            print(f"  Error listando {prefijo}: {e}")

    return keys


def pull(prefijos: list = None):
    """
    Descarga archivos desde S3 usando IAM role.
    Recrea exactamente la jerarquia de carpetas definida en config.py.
    """
    config           = ProjectConfig()
    mapa_inverso     = {v: k for k, v in SYNC_MAP.items()}
    prefijos_a_bajar = prefijos if prefijos else PULL_DEFAULT

    print(f"\n{'='*60}")
    print(f"PULL <- S3")
    print(f"Bucket: s3://{BUCKET_NAME}")
    print(f"{'='*60}\n")

    s3    = get_client()
    total = 0

    for prefijo in prefijos_a_bajar:
        nombre_config = mapa_inverso.get(prefijo)
        if not nombre_config:
            print(f"  Prefijo desconocido: {prefijo}")
            continue

        # Ruta local desde config — respeta cualquier cambio en config.py
        carpeta_local = config.get_folder(nombre_config)
        carpeta_local.mkdir(parents=True, exist_ok=True)

        print(f"  {prefijo} -> {carpeta_local}")

        keys = _listar_objetos(prefijo)
        if not keys:
            print(f"    Sin archivos")
            continue

        count = 0
        for key in keys:
            relativo = key[len(prefijo):]
            destino  = carpeta_local / relativo
            destino.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.download_file(BUCKET_NAME, key, str(destino))
                count += 1
                total += 1
            except Exception as e:
                print(f"    Error {key}: {e}")

        print(f"    {count} archivos descargados")

    print(f"\n  Pull completo — {total} archivos\n")


def pull_all():
    pull(list(set(SYNC_MAP.values())))


# ─────────────────────────────────────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────────────────────────────────────
def status():
    from botocore.exceptions import ClientError
    s3     = get_client()
    config = ProjectConfig()

    print(f"\n{'='*60}")
    print(f"STATUS — s3://{BUCKET_NAME}")
    print(f"{'='*60}\n")

    for nombre_config, prefijo_s3 in SYNC_MAP.items():
        ruta_local     = config.get_folder(nombre_config)
        archivos_local = set()
        if ruta_local.exists():
            archivos_local = {
                str(f.relative_to(ruta_local))
                for f in ruta_local.rglob("*") if f.is_file()
            }

        archivos_s3 = set()
        try:
            pag = s3.get_paginator("list_objects_v2")
            for page in pag.paginate(Bucket=BUCKET_NAME, Prefix=prefijo_s3):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith("/"):
                        archivos_s3.add(key[len(prefijo_s3):])
        except ClientError:
            pass

        solo_local = archivos_local - archivos_s3
        solo_s3    = archivos_s3    - archivos_local
        en_ambos   = archivos_local & archivos_s3

        print(f"  {nombre_config}/ -> {ruta_local}")
        print(f"    Local:{len(archivos_local)} | S3:{len(archivos_s3)} | Sync:{len(en_ambos)}")
        if solo_local:
            print(f"    Pendiente push: {len(solo_local)}")
        if solo_s3:
            print(f"    Pendiente pull: {len(solo_s3)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES PARA OTROS MODULOS
# extrac_pdf_docling.py, chunker_langchain.py y main.py las importan
# ─────────────────────────────────────────────────────────────────────────────
def upload_bronze(config):
    s3       = get_client()
    carpeta  = config.get_folder('bronze')
    prefijo  = SYNC_MAP['bronze']
    archivos = list(carpeta.glob("*.pdf"))
    for archivo in archivos:
        key = prefijo + archivo.name
        s3.upload_file(str(archivo), BUCKET_NAME, key)
        print(f"  {archivo.name} -> s3://{BUCKET_NAME}/{key}")
    print(f"  {len(archivos)} PDFs subidos a bronze/")


def download_bronze(config):
    s3      = get_client()
    carpeta = config.get_folder('bronze')
    carpeta.mkdir(parents=True, exist_ok=True)
    prefijo = SYNC_MAP['bronze']
    count   = 0
    pag     = s3.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=BUCKET_NAME, Prefix=prefijo):
        for obj in page.get("Contents", []):
            key    = obj["Key"]
            nombre = Path(key).name
            if not nombre.endswith(".pdf"):
                continue
            s3.download_file(BUCKET_NAME, key, str(carpeta / nombre))
            count += 1
            print(f"  {nombre}")
    print(f"  {count} PDFs descargados a bronze/")


def sync_vector_db(config, direccion: str = "push"):
    s3      = get_client()
    carpeta = config.get_folder('vector_db')    # ruta viene de config
    prefijo = SYNC_MAP['vector_db']             # prefijo S3 desde SYNC_MAP

    if direccion == "push":
        archivos = [f for f in carpeta.rglob("*") if f.is_file()]
        for archivo in archivos:
            key = prefijo + str(archivo.relative_to(carpeta))
            s3.upload_file(str(archivo), BUCKET_NAME, key)
        print(f"  vector_db/ subido ({len(archivos)} archivos)")

    elif direccion == "pull":
        carpeta.mkdir(parents=True, exist_ok=True)
        count = 0
        pag   = s3.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=BUCKET_NAME, Prefix=prefijo):
            for obj in page.get("Contents", []):
                key      = obj["Key"]
                relativo = key[len(prefijo):]
                if not relativo:
                    continue
                destino  = carpeta / relativo
                destino.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(BUCKET_NAME, key, str(destino))
                count += 1
        print(f"  vector_db/ descargado ({count} archivos)")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    comandos = {
        "setup":    setup_bucket,
        "push":     push,
        "pull":     pull,
        "pull-all": pull_all,
        "status":   status,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in comandos:
        print(__doc__)
        print(f"Comandos disponibles: {', '.join(comandos.keys())}")
        sys.exit(1)

    comandos[sys.argv[1]]()