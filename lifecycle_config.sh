#!/bin/bash
# lifecycle_config.sh — SageMaker Lifecycle Configuration (tipo: Start notebook)
# Este script corre como root cada vez que el notebook instance arranca.


set -e

OLLAMA_DATA="/home/ec2-user/SageMaker/ollama_data"

echo "========================================================"
echo "LIFECYCLE: Iniciando servicios RAG"
echo "========================================================"

# ── Docker ────────────────────────────────────────────────────────────────────
if ! systemctl is-active --quiet docker; then
    echo "Levantando Docker..."
    systemctl start docker
fi

# Dar permisos a ec2-user si no los tiene
if ! groups ec2-user | grep -q docker; then
    usermod -aG docker ec2-user
fi

# ── Contenedor Ollama ─────────────────────────────────────────────────────────
mkdir -p "$OLLAMA_DATA"
chown ec2-user:ec2-user "$OLLAMA_DATA"

if docker ps --format '{{.Names}}' | grep -q "^ollama$"; then
    echo "Contenedor ollama ya corriendo"

elif docker ps -a --format '{{.Names}}' | grep -q "^ollama$"; then
    echo "Reiniciando contenedor ollama existente..."
    docker start ollama

else
    echo "Creando contenedor ollama..."
    docker run -d \
        --name ollama \
        -p 11434:11434 \
        -v "$OLLAMA_DATA:/root/.ollama" \
        --restart unless-stopped \
        ollama/ollama
fi

# Esperar a que Ollama responda
echo "Esperando a Ollama..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:11434 > /dev/null 2>&1; then
        echo "Ollama listo en intento $i"
        break
    fi
    sleep 2
done

# ── Modelos (solo si no existen) ──────────────────────────────────────────────
MODELOS=$(docker exec ollama ollama list 2>/dev/null | awk 'NR>1 {print $1}' || echo "")

if ! echo "$MODELOS" | grep -q "nomic-embed-text"; then
    echo "Descargando nomic-embed-text..."
    docker exec ollama ollama pull nomic-embed-text
fi

if ! echo "$MODELOS" | grep -q "qwen2.5:1.5b"; then
    echo "Descargando qwen2.5:1.5b..."
    docker exec ollama ollama pull qwen2.5:1.5b
fi

echo "========================================================"
echo "LIFECYCLE COMPLETO"
echo "Ollama: $(curl -s http://localhost:11434)"
echo "Modelos disponibles:"
docker exec ollama ollama list
echo "========================================================"