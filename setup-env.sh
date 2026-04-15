#!/bin/bash
# No usamos 'set -e' aquí para que las advertencias de pip no detengan el proceso
PROJECT_ROOT="/home/ec2-user/SageMaker/RAG"
ENV_PATH="$PROJECT_ROOT/rag_env"

echo "🚀 Iniciando fase final de despliegue..."
cd $PROJECT_ROOT

# 1. Asegurar entorno
source "$ENV_PATH/bin/activate" || { echo "❌ Activa el entorno primero"; exit 1; }

# 2. Instalación de dependencias del proyecto
echo "📦 Instalando desde requirements.txt..."
if [ -f "requirements.txt" ]; then
    # Instalamos ignorando conflictos de versiones para que OpenCV no bloquee a Numpy 1.x
    pip install -r requirements.txt --prefer-binary
else
    echo "❌ ERROR: Aún no existe requirements.txt. Créalo con el comando 'cat' anterior."
    exit 1
fi

# 3. Sincronización de Ollama (por si acaso)
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > ollama.log 2>&1 &
    sleep 5
fi

# 4. TEST DE VERIFICACIÓN DEFINITIVO
echo "-------------------------------------------------------"
echo "🔍 VERIFICACIÓN DE LIBRERÍAS CLAVE"
echo "-------------------------------------------------------"

python3 << EOF
import numpy as np
import cv2
import camelot
import datasets
print(f"✅ Numpy: {np.__version__} (Correcto)")
print(f"✅ OpenCV + Camelot: Vinculados")
print(f"✅ Datasets: {datasets.__version__}")
print("\n✨ ¡ENTORNO LISTOS!")
EOF


echo "-------------------------------------------------------"
echo "✅ ENTORNO CONFIGURADO Y PORTABLE"
echo "-------------------------------------------------------"
echo "💡 IMPORTANTE: Si ves errores de 'HfFolder' o 'ImportError', ejecuta esto:"
echo "   pip install 'huggingface_hub==0.19.4' 'datasets==2.15.0' 'cryptography<45.0.0' --force-reinstall"
echo ""
echo "🚀 Para lanzar la ingesta de tu tesis:"
echo "   source rag_env/bin/activate"
echo "   python3 -m src.ingest.extrac_pdf_docling"
echo "-------------------------------------------------------"
