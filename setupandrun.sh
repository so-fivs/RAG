#!/bin/bash

# ============================================================================
# SETUP AND RUN - Sistema RAG FDS con Google Drive
# ============================================================================
# Automatiza: Sync Google Drive → Detectar nuevos → Procesar → Ingestar → Run
# ============================================================================

set -e  # Exit on error

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   🚀 SISTEMA RAG FDS - SETUP AUTOMÁTICO CON GOOGLE DRIVE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GDRIVE_PATH="$HOME/Google Drive/FDS_PDFs_Shared"
RAW_DOCS_PATH="$PROJECT_ROOT/raw_documents"
STATE_FILE="$PROJECT_ROOT/.rag_state.json"

cd "$PROJECT_ROOT"

# ────────────────────────────────────────────────────────────────────────────
# 1. VERIFICAR PYTHON Y VENV
# ────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/10]${NC} Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 no instalado${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $(python3 --version | cut -d' ' -f2)${NC}"

# Activar venv
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creando entorno virtual...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✓ Entorno virtual activado${NC}"

# ────────────────────────────────────────────────────────────────────────────
# 2. INSTALAR/ACTUALIZAR DEPENDENCIAS
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/10]${NC} Verificando dependencias..."
pip install -q --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencias actualizadas${NC}"
fi

# ────────────────────────────────────────────────────────────────────────────
# 3. VERIFICAR GOOGLE DRIVE
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/10]${NC} Verificando Google Drive..."

if [ ! -d "$GDRIVE_PATH" ]; then
    echo -e "${RED}❌ Google Drive no sincronizado${NC}"
    echo -e "${YELLOW}Por favor:${NC}"
    echo "  1. Instala Google Drive Desktop"
    echo "  2. Sincroniza la carpeta: FDS_PDFs_Shared"
    echo "  3. Vuelve a ejecutar este script"
    exit 1
fi

GDRIVE_PDFS=$(find "$GDRIVE_PATH" -name "*.pdf" | wc -l)
echo -e "${GREEN}✓ Google Drive encontrado: $GDRIVE_PDFS PDFs${NC}"

# Crear/verificar symlink
if [ ! -e "$RAW_DOCS_PATH" ]; then
    ln -s "$GDRIVE_PATH" "$RAW_DOCS_PATH"
    echo -e "${GREEN}✓ Symlink creado: raw_documents → Google Drive${NC}"
elif [ -L "$RAW_DOCS_PATH" ]; then
    echo -e "${GREEN}✓ Symlink ya existe${NC}"
fi

# ────────────────────────────────────────────────────────────────────────────
# 4. DETECTAR PDFS NUEVOS/MODIFICADOS
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[4/10]${NC} Detectando cambios en PDFs..."

python3 << 'EOF'
import json
import hashlib
from pathlib import Path

STATE_FILE = Path('.rag_state.json')
RAW_DOCS = Path('raw_documents')

# Cargar estado anterior
if STATE_FILE.exists():
    with open(STATE_FILE, 'r') as f:
        state = json.load(f)
else:
    state = {'documents': {}, 'last_sync': None}

# Calcular hashes actuales
current_pdfs = {}
for pdf in RAW_DOCS.glob('*.pdf'):
    pdf_hash = hashlib.md5(pdf.read_bytes()).hexdigest()
    current_pdfs[pdf.name] = {
        'hash': pdf_hash,
        'size': pdf.stat().st_size
    }

# Detectar cambios
nuevos = []
modificados = []
eliminados = []

for pdf_name, pdf_info in current_pdfs.items():
    if pdf_name not in state['documents']:
        nuevos.append(pdf_name)
    elif state['documents'][pdf_name]['hash'] != pdf_info['hash']:
        modificados.append(pdf_name)

for pdf_name in state['documents']:
    if pdf_name not in current_pdfs:
        eliminados.append(pdf_name)

# Resultados
print(f"📊 Estado:")
print(f"  Total PDFs en Drive: {len(current_pdfs)}")
print(f"  Nuevos: {len(nuevos)}")
print(f"  Modificados: {len(modificados)}")
print(f"  Eliminados: {len(eliminados)}")

# Guardar lista de PDFs a procesar
to_process = nuevos + modificados
if to_process:
    with open('.pdfs_to_process.txt', 'w') as f:
        f.write('\n'.join(to_process))
    print(f"\n⚠️  Requiere procesamiento:")
    for pdf in to_process:
        status = "[NUEVO]" if pdf in nuevos else "[MODIFICADO]"
        print(f"    {status} {pdf}")
else:
    print("\n✅ Todos los PDFs ya están procesados")

# Guardar estado actualizado
state['documents'] = current_pdfs
state['to_process'] = to_process
with open(STATE_FILE, 'w') as f:
    json.dump(state, f, indent=2)

exit(0 if not to_process else 1)
EOF

NEEDS_PROCESSING=$?

# ────────────────────────────────────────────────────────────────────────────
# 5. VERIFICAR OLLAMA
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[5/10]${NC} Verificando Ollama..."

if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama no instalado${NC}"
    echo "Instala desde: https://ollama.com/download"
    exit 1
fi

if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Iniciando Ollama...${NC}"
    ollama serve &
    sleep 3
fi

echo -e "${GREEN}✓ Ollama corriendo${NC}"

# Verificar modelos
if ! ollama list | grep -q "nomic-embed-text"; then
    echo -e "${YELLOW}📥 Descargando nomic-embed-text...${NC}"
    ollama pull nomic-embed-text
fi

if ! ollama list | grep -q "qwen2.5:1.5b"; then
    echo -e "${YELLOW}📥 Descargando qwen2.5:1.5b...${NC}"
    ollama pull qwen2.5:1.5b
fi

echo -e "${GREEN}✓ Modelos listos${NC}"

# ────────────────────────────────────────────────────────────────────────────
# 6. PROCESAR PDFS NUEVOS/MODIFICADOS
# ────────────────────────────────────────────────────────────────────────────
if [ $NEEDS_PROCESSING -eq 1 ]; then
    echo -e "\n${YELLOW}[6/10]${NC} Procesando PDFs nuevos/modificados..."
    
    # Leer lista de PDFs a procesar
    if [ -f ".pdfs_to_process.txt" ]; then
        TOTAL_TO_PROCESS=$(wc -l < .pdfs_to_process.txt)
        echo -e "${BLUE}Procesando $TOTAL_TO_PROCESS PDFs...${NC}"
        
        # PASO 6.1: EXTRACTOR
        echo -e "\n${BLUE}  → Extrayendo contenido de PDFs...${NC}"
        python3 extractor_pdf.py
        
        # PASO 6.2: CHUNKER + EMBEDDINGS
        echo -e "\n${BLUE}  → Generando chunks y embeddings...${NC}"
        python3 chunkers_embedder.py
        
        # PASO 6.3: INGESTOR (con skip_existing=True automático)
        echo -e "\n${BLUE}  → Ingiriendo en ChromaDB...${NC}"
        python3 chromadb_ingestor.py
        
        # Limpiar archivo temporal
        rm -f .pdfs_to_process.txt
        
        echo -e "${GREEN}✓ Procesamiento completado${NC}"
    fi
else
    echo -e "\n${YELLOW}[6/10]${NC} ${GREEN}✓ No hay PDFs nuevos para procesar${NC}"
fi

# ────────────────────────────────────────────────────────────────────────────
# 7. ANÁLISIS DE BASE DE DATOS (DIAGNÓSTICO)
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[7/10]${NC} Analizando base de datos..."

python3 << 'EOF'
import chromadb
from pathlib import Path

db_path = Path('./data/vector_db')
try:
    client = chromadb.PersistentClient(path=str(db_path))
    collections = client.list_collections()
    
    if not collections:
        print("⚠️  ChromaDB vacía (se procesará en próxima ejecución)")
    else:
        total_chunks = 0
        for coll in collections:
            count = coll.count()
            total_chunks += count
            print(f"  ✓ {coll.name}: {count} chunks")
        print(f"\nTotal: {total_chunks} chunks en ChromaDB")
except Exception as e:
    print(f"⚠️  Error: {e}")
EOF

# ────────────────────────────────────────────────────────────────────────────
# 8. VERIFICAR ESTRUCTURA DE CARPETAS
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[8/10]${NC} Verificando estructura..."

python3 -c "from config import ProjectConfig; c = ProjectConfig(); c.create_folders(); c.verify_structure()"

# ────────────────────────────────────────────────────────────────────────────
# 9. INICIAR BACKEND
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[9/10]${NC} Iniciando backend FastAPI..."

# Verificar que main.py existe
if [ ! -f "src/api/main.py" ]; then
    echo -e "${RED}❌ src/api/main.py no encontrado${NC}"
    exit 1
fi

# Matar procesos anteriores de uvicorn si existen
pkill -f "uvicorn.*main:app" 2>/dev/null || true

# Iniciar FastAPI en background
cd src/api
python3 main.py &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Esperar a que inicie
sleep 5

if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ Backend corriendo en http://localhost:8000 (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}❌ Error iniciando backend${NC}"
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────
# 10. ABRIR NAVEGADOR
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[10/10]${NC} Abriendo navegador..."

# Detectar OS y abrir navegador
if [[ "$OSTYPE" == "darwin"* ]]; then
    open http://localhost:8000
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open http://localhost:8000 2>/dev/null || true
fi

echo -e "${GREEN}✓ Navegador abierto${NC}"

# ────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ────────────────────────────────────────────────────────────────────────────
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}   ✅ SISTEMA LISTO${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}🌐 Frontend:${NC} http://localhost:8000"
echo -e "${GREEN}📊 Backend PID:${NC} $BACKEND_PID"
echo -e "${GREEN}💾 Estado guardado en:${NC} .rag_state.json"
echo ""
echo -e "${YELLOW}📋 Comandos útiles:${NC}"
echo "  • Ver logs backend: tail -f logs/backend.log"
echo "  • Detener backend: kill $BACKEND_PID"
echo "  • Analizar DB: python analyze_db.py"
echo "  • Limpiar y reiniciar: ./cleanup_and_reset.sh"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Mantener script corriendo y mostrar logs
echo -e "${YELLOW}Presiona Ctrl+C para detener el servidor${NC}"
trap "echo -e '\n${RED}Deteniendo servidor...${NC}'; kill $BACKEND_PID 2>/dev/null; exit" INT

# Seguir logs del backend
tail -f src/api/*.log 2>/dev/null || wait $BACKEND_PID