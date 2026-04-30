#!/bin/bash
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_PATH="$PROJECT_ROOT/rag_env"

echo "🚀 Iniciando despliegue RAG FDS..."
cd $PROJECT_ROOT

# 1. Crear entorno virtual si no existe
if [ ! -d "$ENV_PATH" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate" || { echo "❌ No se pudo activar el entorno"; exit 1; }

# 2. Dependencias base
echo "📦 Instalando dependencias..."
pip install --upgrade pip --quiet

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --prefer-binary --quiet
else
    echo "❌ No existe requirements.txt"
    exit 1
fi

# 3. Dependencias adicionales del sistema RAG
echo "📦 Instalando dependencias RAG..."
pip install \
    docling \
    PyMuPDF \
    "transformers>=4.45.0" \
    chromadb \
    langchain-community \
    langchain-ollama \
    langchain-text-splitters \
    ollama \
    fastapi \
    "uvicorn[standard]" \
    mlflow \
    jinja2 \
    python-multipart \
    sentence-transformers \
    tabulate \
    --prefer-binary --quiet

# 4. Docker + Ollama
echo "🐳 Verificando Docker y Ollama..."
if ! command -v docker &> /dev/null; then
    echo "  Instalando Docker..."
    sudo yum update -y -q
    sudo yum install docker -y -q
    sudo systemctl start docker
    sudo usermod -aG docker ec2-user
fi

if ! docker ps | grep -q ollama; then
    echo "  Levantando contenedor Ollama..."
    docker run -d \
        --name ollama \
        -p 11434:11434 \
        -v "$PROJECT_ROOT/ollama_data:/root/.ollama" \
        --restart unless-stopped \
        ollama/ollama
    sleep 8
fi

# 5. Modelos Ollama
echo "🤖 Descargando modelos Ollama..."
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull qwen2.5:1.5b

# 6. Carpetas del proyecto
echo "📁 Creando estructura de carpetas..."
python3 -c "
import sys; sys.path.insert(0, '.')
from config import ProjectConfig
ProjectConfig().create_folders()
"

# 7. Verificación final
echo "-------------------------------------------------------"
echo "🔍 VERIFICACIÓN"
echo "-------------------------------------------------------"
python3 << EOF
import numpy as np
import cv2
import chromadb
import ollama
import fastapi
print(f"✅ Numpy:     {np.__version__}")
print(f"✅ OpenCV:    {cv2.__version__}")
print(f"✅ ChromaDB:  {chromadb.__version__}")
print(f"✅ FastAPI:   {fastapi.__version__}")
print(f"✅ Ollama:    OK")
print("\n✨ Entorno listo!")
EOF

echo "-------------------------------------------------------"
echo "✅ SETUP COMPLETO"
echo "-------------------------------------------------------"
echo ""
echo "▶️  Para procesar PDFs (solo primera vez):"
echo "    source rag_env/bin/activate"
echo "    python3 src/ingest/extrac_pdf_docling.py"
echo "    python3 src/ingest/chunker_langchain.py"
echo ""
echo "▶️  Para lanzar la API:"
echo "    source rag_env/bin/activate"
echo "    python3 src/api/main.py"
echo "-------------------------------------------------------"