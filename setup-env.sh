#!/bin/bash
# setup-env.sh — Configura el entorno RAG FDS completo
# Uso: bash setup-env.sh
set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_PATH="$PROJECT_ROOT/rag_env"

echo "========================================================"
echo "SETUP RAG FDS"
echo "========================================================"
cd "$PROJECT_ROOT"

# ── 1. Entorno virtual ────────────────────────────────────────────────────────
if [ ! -d "$ENV_PATH" ]; then
    echo "[1/6] Creando entorno virtual..."
    python3 -m venv "$ENV_PATH"
else
    echo "[1/6] Entorno virtual ya existe, omitiendo creacion"
fi

source "$ENV_PATH/bin/activate" || { echo "ERROR: No se pudo activar el entorno"; exit 1; }

# ── 2. Dependencias pip ───────────────────────────────────────────────────────
echo "[2/6] Instalando dependencias pip..."
pip install --upgrade pip --quiet

if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    pip install -r "$PROJECT_ROOT/requirements.txt" --prefer-binary --quiet
else
    echo "ERROR: No existe requirements.txt"
    exit 1
fi

# ── 3. Docker ─────────────────────────────────────────────────────────────────
echo "[3/6] Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo "  Docker no encontrado, instalando..."
    sudo yum install docker -y -q
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
    newgrp docker
else
    if ! sudo systemctl is-active --quiet docker; then
        echo "  Docker instalado pero detenido, levantando..."
        sudo systemctl start docker
    else
        echo "  Docker ya esta corriendo"
    fi
fi

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
echo "[4/6] Verificando contenedor Ollama..."

# Caso 1: contenedor corriendo
if docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
    echo "  Contenedor ollama ya esta corriendo"

# Caso 2: contenedor existe pero detenido
elif docker ps -a --format '{{.Names}}' | grep -q "^ollama$"; then
    echo "  Contenedor ollama existe pero esta detenido, reiniciando..."
    docker start ollama
    sleep 5

# Caso 3: contenedor no existe
else
    echo "  Creando contenedor ollama..."
    docker run -d \
        --name ollama \
        -p 11434:11434 \
        -v /home/ec2-user/SageMaker/ollama_data:/root/.ollama \
        --restart unless-stopped \
        ollama/ollama
    sleep 10
fi

# ── 5. Modelos Ollama ─────────────────────────────────────────────────────────
echo "[5/6] Verificando modelos Ollama..."

MODELOS_INSTALADOS=$(docker exec ollama ollama list 2>/dev/null | awk 'NR>1 {print $1}')

if echo "$MODELOS_INSTALADOS" | grep -q "nomic-embed-text"; then
    echo "  nomic-embed-text ya instalado"
else
    echo "  Descargando nomic-embed-text..."
    docker exec ollama ollama pull nomic-embed-text
fi

if echo "$MODELOS_INSTALADOS" | grep -q "qwen2.5:1.5b"; then
    echo "  qwen2.5:1.5b ya instalado"
else
    echo "  Descargando qwen2.5:1.5b..."
    docker exec ollama ollama pull qwen2.5:1.5b
fi

# ── 6. Estructura local + S3 pull ─────────────────────────────────────────────
echo "[6/6] Carpetas locales y sincronizacion S3..."

python3 -c "
import sys
sys.path.insert(0, '.')
from config import ProjectConfig
ProjectConfig().create_folders()
"

# Pull de vector_db desde S3 (si boto3 disponible y vector_db vacio)
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')

try:
    from config import ProjectConfig
    from scripts.s3_sync import sync_vector_db

    config   = ProjectConfig()
    db_path  = config.get_folder('vector_db')
    db_vacia = not db_path.exists() or not any(db_path.rglob('*.sqlite3'))

    if db_vacia:
        print("  vector_db/ vacio, descargando desde S3...")
        sync_vector_db(config, direccion="pull")
    else:
        print("  vector_db/ ya existe localmente, omitiendo pull")

except ImportError as e:
    print(f"  S3 no disponible ({e}), omitiendo pull")
except Exception as e:
    print(f"  Error en S3 pull: {e}")
PYEOF

# ── Aliases utiles ────────────────────────────────────────────────────────────
grep -qxF "alias ollama-logs='docker logs -f ollama'" ~/.bashrc || \
    echo "alias ollama-logs='docker logs -f ollama'" >> ~/.bashrc
grep -qxF "alias ollama-restart='docker restart ollama'" ~/.bashrc || \
    echo "alias ollama-restart='docker restart ollama'" >> ~/.bashrc
grep -qxF "alias ollama-models='docker exec ollama ollama list'" ~/.bashrc || \
    echo "alias ollama-models='docker exec ollama ollama list'" >> ~/.bashrc
source ~/.bashrc 2>/dev/null || true

# ── Verificacion final ────────────────────────────────────────────────────────
echo ""
echo "--------------------------------------------------------"
echo "VERIFICACION"
echo "--------------------------------------------------------"
python3 -c "
import numpy, cv2, chromadb, ollama, fastapi, boto3
print(f'  numpy      {numpy.__version__}')
print(f'  opencv     {cv2.__version__}')
print(f'  chromadb   {chromadb.__version__}')
print(f'  fastapi    {fastapi.__version__}')
print(f'  boto3      {boto3.__version__}')
print(f'  ollama     OK')
"

echo "--------------------------------------------------------"
echo "  Ollama:  http://localhost:11434"
echo "  API URL: https://seurag.notebook.us-east-1.sagemaker.aws/proxy/8000"
echo "--------------------------------------------------------"
echo ""
echo "  Para lanzar la API:"
echo "    source rag_env/bin/activate"
echo "    python3 src/api/main.py"
echo ""
echo "  Para reprocesar PDFs (si hay nuevas FDS en bronze/):"
echo "    python3 src/ingest/extrac_pdf_docling.py"
echo "    python3 src/ingest/chunker_langchain.py"
echo "--------------------------------------------------------"