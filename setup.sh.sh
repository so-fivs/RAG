#!/bin/bash

# ============================================================================
# SCRIPT DE INSTALACIÓN AUTOMATIZADA - Sistema RAG FDS
# ============================================================================
# Uso: bash setup.sh
# ============================================================================

set -e  # Exit on error

echo "🚀 Iniciando configuración del Sistema RAG para Fichas de Seguridad..."
echo ""

# ────────────────────────────────────────────────────────────────────────────
# 1. VERIFICAR PYTHON
# ────────────────────────────────────────────────────────────────────────────
echo "📋 Paso 1: Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Instálalo desde python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python detectado: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.9" | bc) -eq 1 ]]; then
    echo "⚠️  Advertencia: Se recomienda Python >= 3.9"
fi

# ────────────────────────────────────────────────────────────────────────────
# 2. CREAR ENTORNO VIRTUAL
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "📦 Paso 2: Creando entorno virtual..."

if [ -d "venv" ]; then
    echo "⚠️  El directorio 'venv' ya existe."
    read -p "¿Deseas eliminarlo y recrearlo? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✅ Entorno virtual recreado"
    else
        echo "ℹ️  Usando entorno existente"
    fi
else
    python3 -m venv venv
    echo "✅ Entorno virtual creado"
fi

# Activar entorno virtual
source venv/bin/activate

# ────────────────────────────────────────────────────────────────────────────
# 3. ACTUALIZAR PIP
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "⬆️  Paso 3: Actualizando pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✅ Pip actualizado"

# ────────────────────────────────────────────────────────────────────────────
# 4. INSTALAR DEPENDENCIAS
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "📚 Paso 4: Instalando dependencias de Python..."
echo "   (Esto puede tomar varios minutos...)"

if [ ! -f "requirements.txt" ]; then
    echo "❌ Archivo requirements.txt no encontrado"
    exit 1
fi

pip install -r requirements.txt

echo "✅ Dependencias instaladas"

# ────────────────────────────────────────────────────────────────────────────
# 5. VERIFICAR OLLAMA
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "🤖 Paso 5: Verificando Ollama..."

if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama no está instalado"
    echo ""
    echo "📥 Instala Ollama desde: https://ollama.com/download"
    echo ""
    echo "Comandos de instalación:"
    echo "  macOS:   brew install ollama"
    echo "  Linux:   curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Windows: Descarga desde https://ollama.com/download"
    exit 1
else
    echo "✅ Ollama detectado: $(ollama --version)"
fi

# Verificar si Ollama está corriendo
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "⚠️  Ollama no está corriendo"
    echo "   Inicia el servidor con: ollama serve"
    echo ""
    read -p "¿Deseas iniciar Ollama ahora? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "🔄 Iniciando Ollama en background..."
        ollama serve &
        sleep 3
        echo "✅ Ollama iniciado"
    fi
else
    echo "✅ Ollama está corriendo"
fi

# ────────────────────────────────────────────────────────────────────────────
# 6. DESCARGAR MODELOS
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "📥 Paso 6: Verificando modelos de Ollama..."

# Verificar nomic-embed-text
if ollama list | grep -q "nomic-embed-text"; then
    echo "✅ nomic-embed-text ya está instalado"
else
    echo "📥 Descargando nomic-embed-text (esto puede tardar)..."
    ollama pull nomic-embed-text
fi

# Verificar qwen2.5:1.5b
if ollama list | grep -q "qwen2.5:1.5b"; then
    echo "✅ qwen2.5:1.5b ya está instalado"
else
    echo "📥 Descargando qwen2.5:1.5b (esto puede tardar)..."
    ollama pull qwen2.5:1.5b
fi

echo "✅ Modelos verificados"

# ────────────────────────────────────────────────────────────────────────────
# 7. VERIFICAR TESSERACT (OPCIONAL)
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "🔍 Paso 7: Verificando Tesseract OCR..."

if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract OCR detectado: $(tesseract --version | head -n1)"
else
    echo "⚠️  Tesseract OCR no está instalado (opcional para OCR de imágenes)"
    echo "   Instálalo con:"
    echo "     macOS:   brew install tesseract"
    echo "     Ubuntu:  sudo apt-get install tesseract-ocr"
    echo "     Windows: https://github.com/UB-Mannheim/tesseract/wiki"
fi

# ────────────────────────────────────────────────────────────────────────────
# 8. CREAR ESTRUCTURA DE CARPETAS
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "📁 Paso 8: Verificando estructura de carpetas..."

mkdir -p templates
mkdir -p data/vector_db
mkdir -p data/extracted_content/images
mkdir -p data/processed_chunks
mkdir -p data/metadata

echo "✅ Estructura de carpetas verificada"

# ────────────────────────────────────────────────────────────────────────────
# 9. VERIFICAR CHROMADB
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "🗄️  Paso 9: Verificando ChromaDB..."

python3 << EOF
import chromadb
from pathlib import Path

db_path = Path('./data/vector_db')
try:
    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    
    if len(collections) == 0:
        print("⚠️  ChromaDB está vacío. Ejecuta chromadb_ingestor.py para cargar datos.")
    else:
        print(f"✅ ChromaDB tiene {len(collections)} colecciones:")
        for coll in collections:
            count = coll.count()
            print(f"   - {coll.name}: {count} chunks")
except Exception as e:
    print(f"⚠️  Error verificando ChromaDB: {e}")
EOF

# ────────────────────────────────────────────────────────────────────────────
# 10. RESUMEN FINAL
# ────────────────────────────────────────────────────────────────────────────
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ INSTALACIÓN COMPLETADA                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Siguiente paso: Iniciar el sistema"
echo ""
echo "🔧 Comandos para iniciar:"
echo ""
echo "   Terminal 1 (Ollama):"
echo "   $ ollama serve"
echo ""
echo "   Terminal 2 (Backend):"
echo "   $ source venv/bin/activate"
echo "   $ cd src/api"
echo "   $ python main.py"
echo ""
echo "   Navegador:"
echo "   Abre: http://localhost:8000"
echo ""
echo "📚 Documentación completa en: README.md"
echo ""
echo "⚠️  Notas importantes:"
echo "   - Si ChromaDB está vacío, ejecuta: python src/data_processing/chromadb_ingestor.py"
echo "   - Si falta index.html, cópialo a templates/index.html"
echo "   - Si falta generator.py optimizado, actualízalo en src/rag/"
echo ""

# Desactivar entorno virtual
deactivate