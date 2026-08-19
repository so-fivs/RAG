#!/bin/bash
set -e
exec > >(tee -a /var/log/rag-bootstrap.log) 2>&1

echo "===== RAG FDS bootstrap: $(date) ====="

APP_USER="ubuntu"
APP_DIR="/home/$APP_USER/RAG"

# ---------------------------------------------------------------------------
# 1. Paquetes del sistema
# ---------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.11 python3.11-venv python3-pip git curl unzip \
    poppler-utils ghostscript tesseract-ocr libgl1 libglib2.0-0

# AWS CLI v2 (usa el rol de la instancia, sin credenciales manuales)
if ! command -v aws &> /dev/null; then
    curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install
fi

# ---------------------------------------------------------------------------
# 2. Ollama + modelos que se usan en tiempo de consulta (embeddings + LLM)
# ---------------------------------------------------------------------------
curl -fsSL https://ollama.com/install.sh | sh
sleep 5
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b

# ---------------------------------------------------------------------------
# 3. Clonar repo
# ---------------------------------------------------------------------------
sudo -u $APP_USER git clone --branch ${repo_branch} ${repo_url} "$APP_DIR"
cd "$APP_DIR"

# ---------------------------------------------------------------------------
# 4. Entorno Python
# ---------------------------------------------------------------------------
sudo -u $APP_USER python3.11 -m venv venv
sudo -u $APP_USER ./venv/bin/pip install --upgrade pip
sudo -u $APP_USER ./venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------------------
# 5. PDFs desde el bucket S3 (siempre, los sirve /api/download_pdf)
# ---------------------------------------------------------------------------
sudo -u $APP_USER mkdir -p data/bronze
sudo -u $APP_USER aws s3 sync "s3://${bucket_name}/fds/" data/bronze/

# ---------------------------------------------------------------------------
# 6. Vector DB: reusar dump si existe, si no, correr el pipeline completo
#    (extracción + chunking + descripciones con llava:7b) y subir el dump
#    para que la próxima máquina no tenga que reprocesar nada.
# ---------------------------------------------------------------------------
DUMP_KEY="dump/vector_db.tar.gz"

if aws s3api head-object --bucket "${bucket_name}" --key "$DUMP_KEY" > /dev/null 2>&1; then
    echo "Dump de vector_db encontrado en s3://${bucket_name}/$DUMP_KEY — restaurando (sin reprocesar PDFs)..."
    aws s3 cp "s3://${bucket_name}/$DUMP_KEY" /tmp/vector_db.tar.gz
    sudo -u $APP_USER mkdir -p data
    sudo -u $APP_USER tar xzf /tmp/vector_db.tar.gz -C data/
    chown -R $APP_USER:$APP_USER data/vector_db
else
    echo "No hay dump previo. Corriendo pipeline completo de ingesta (usa llava:7b, puede tardar varios minutos)..."
    ollama pull llava:7b

    sudo -u $APP_USER ./venv/bin/python src/ingest/extrac_pdf.py
    sudo -u $APP_USER ./venv/bin/python src/ingest/chunkers-embedder.py
    sudo -u $APP_USER ./venv/bin/python src/ingest/image_vision_procesor.py
    sudo -u $APP_USER ./venv/bin/python src/ingest/chromadb_ingestor.py

    echo "Generando dump de vector_db y subiéndolo a s3://${bucket_name}/$DUMP_KEY para futuras instancias..."
    sudo -u $APP_USER tar czf /tmp/vector_db.tar.gz -C data vector_db
    aws s3 cp /tmp/vector_db.tar.gz "s3://${bucket_name}/$DUMP_KEY"
fi

# ---------------------------------------------------------------------------
# 7. Servicio systemd para el API
# ---------------------------------------------------------------------------
cat > /etc/systemd/system/rag-fds.service <<EOF
[Unit]
Description=RAG FDS FastAPI backend
After=network.target ollama.service

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR/src/api
ExecStart=$APP_DIR/venv/bin/python $APP_DIR/src/api/main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rag-fds.service
systemctl start rag-fds.service

echo "===== RAG FDS bootstrap terminado: $(date) ====="
