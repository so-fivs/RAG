# RAG FDS — Sistema de Recuperación Aumentada para Fichas de Datos de Seguridad

Sistema RAG (**Retrieval-Augmented Generation**) para consultar Fichas de Datos de Seguridad (FDS) de productos químicos. Procesa PDFs, vectoriza su contenido en ChromaDB y expone una API conversacional con recuperación semántica híbrida y generación de lenguaje natural local mediante Ollama.

---

## Arquitectura General

```mermaid
graph TB
    subgraph INPUT["📥 Ingesta (Bronze → Silver → Gold)"]
        A[PDFs FDS<br/>data/bronze/] -->|extrac_pdf.py| B[Textos + Tablas + Imágenes<br/>data/gold/]
        B -->|chunkers_embedder.py| C[Chunks + Embeddings 768D<br/>data/silver/chunked/]
        B -->|image_vision_processor.py| D[Descripciones visuales<br/>llava:7b]
        C --> E[ChromaDB<br/>data/vector_db/]
        D --> E
    end

    subgraph RAG["🔍 Sistema RAG"]
        F[Query usuario] --> G[HybridRetriever<br/>src/backend/retriever.py]
        G -->|similitud coseno| E
        G --> H[Chunks relevantes<br/>+ metadata H/P/CAS]
        H --> I[RAGGenerator<br/>src/backend/generator.py]
        I -->|qwen2.5:1.5b| J[Respuesta natural]
    end

    subgraph API["🌐 API FastAPI"]
        K[Frontend<br/>src/frontend/] --> L[POST /api/chat]
        L --> F
        J --> L
        L --> K
    end

    subgraph EVAL["📊 Evaluación"]
        M[RAGAS<br/>evaluation/] -->|métricas| N[faithfulness<br/>relevancy<br/>precision]
        O[MLflow<br/>mlruns/] -->|tracking| P[latencia<br/>similitud<br/>chunks]
    end
```

---

## Pipeline de Procesamiento (Arquitectura Medallón)

```mermaid
flowchart LR
    subgraph BRONZE["🥉 Bronze — Raw"]
        A[PDF FDS original]
    end

    subgraph GOLD["🥇 Gold — Enriched"]
        B[texts/]
        C[tables/]
        D[images/]
        E[metadata/]
    end

    subgraph SILVER["🥈 Silver — Chunked"]
        F[chunked/\n*_chunks.json\nembeddings 768D]
    end

    subgraph VECTOR["🗄️ Vector DB"]
        G[(ChromaDB\nfds_textos: 80\nfds_tablas: 10\nfds_imagens: 30)]
    end

    A -->|extrac_pdf.py| B & C & D & E
    D -->|image_vision_processor.py\nllava:7b| D
    B & C & D & E -->|chunkers_embedder.py\nnomic-embed-text| F
    F -->|chromadb_ingestor.py| G
```

---

## Flujo RAG: Retriever y Generator

```mermaid
sequenceDiagram
    participant U as Usuario
    participant API as FastAPI
    participant R as HybridRetriever
    participant DB as ChromaDB
    participant LLM as qwen2.5:1.5b

    U->>API: POST /api/chat {query, session_id}
    API->>R: retrieve(query, n_candidates=30, n_final=15)
    R->>R: detectar_tipo_query()<br/>(peligros/composicion/primeros_auxilios...)
    R->>R: fuzzy_match_producto()
    R->>DB: query fds_textos + fds_tablas + fds_imagens
    DB-->>R: chunks + similitudes coseno
    R->>R: boost_imágenes + rerank
    R-->>API: top chunks + metadata H/P/CAS
    API->>LLM: prompt + contexto (2048 tokens)
    LLM-->>API: respuesta natural
    API->>API: MLflow.log_metrics()
    API-->>U: {answer, sources, codigos_h, codigos_p}
```

---

## Evaluación con RAGAS

```mermaid
flowchart TD
    A[Dataset de prueba\nevaluation/ragas_eval_dataset.json] --> B[RAGAS Evaluator\nevaluation/ragas_eval_ollama.py]
    
    B --> C{Métricas}
    C --> D[faithfulness\n¿respuesta basada en docs?]
    C --> E[answer_relevancy\n¿respuesta relevante?]
    C --> F[context_precision\n¿chunks correctos recuperados?]
    C --> G[context_recall\n¿todos los chunks necesarios?]
    
    D & E & F & G --> H[ragas_eval_summary.json]
    H --> I[MLflow Dashboard\nhttp://localhost:5000]
```

---

## Modelos Utilizados

| Modelo | Propósito | Dimensión | Comando |
|--------|-----------|-----------|---------|
| `nomic-embed-text` | Embeddings vectoriales (texto, tablas, imágenes) | 768D | `ollama pull nomic-embed-text` |
| `qwen2.5:1.5b` | Generación de respuestas — temp: 0.3, tokens: 500, ctx: 2048 | 1.5B params | `ollama pull qwen2.5:1.5b` |
| `llava:7b` | Descripción multimodal de imágenes (preprocesamiento offline) | 7B params | `ollama pull llava:7b` |

---

## Estadísticas del Dataset

| Colección ChromaDB | Chunks | Tipo de contenido |
|-------------------|--------|-------------------|
| `fds_textos` | 80 | Texto por secciones FDS |
| `fds_tablas` | 10 | Tablas convertidas a texto |
| `fds_imagens` | 30 | Descripciones visuales |
| **Total** | **120** | **5 documentos FDS** |

**Parámetros de chunking:**
- Tamaño máximo: 800 tokens por chunk (~3.200 caracteres)
- Overlap: 150 tokens entre chunks consecutivos
- Método: división por párrafos respetando límite de tokens
- Contexto: nombre del producto + sección prepended en cada chunk

---

## Estructura del Repositorio

```
RAG/
├── config.py                          # Configuración centralizada de rutas
├── requirements.txt
├── setupandrun.sh                     # Setup automático con Google Drive
│
├── data/
│   ├── bronze/                        # 🥉 PDFs originales
│   ├── silver/
│   │   └── chunked/                   # 🥈 Chunks + embeddings JSON
│   ├── gold/                          # 🥇 Contenido extraído
│   │   ├── texts/
│   │   ├── tables/
│   │   ├── images/
│   │   └── metadata/
│   └── vector_db/                     # ChromaDB (chroma.sqlite3)
│
├── evaluation/
│   ├── ragas_eval_dataset.json        # Dataset de preguntas/respuestas
│   ├── ragas_eval_ollama.py           # Evaluación con modelos locales
│   └── ragas_eval_summary.json        # Resultados de métricas
│
├── scripts/
│   ├── analyze_db.py
│   ├── chunkers_embedder.py
│   └── cleanup_chromadb.py
│
├── src/
│   ├── api/
│   │   └── main.py                    # API FastAPI
│   ├── backend/
│   │   ├── generator.py               # RAGGenerator
│   │   ├── retriever.py               # HybridRetriever
│   │   └── vectorstore.py             # Wrapper ChromaDB
│   ├── frontend/
│   │   └── index.html                 # Interfaz web
│   └── ingest/
│       ├── chromadb_diagnostic.py
│       ├── chromadb_ingestor.py
│       ├── extrac_pdf.py
│       └── image_vision_procesor.py
│
└── mlruns/                            # Tracking MLflow
```

---

## Configuración y Ejecución

```bash
# Prerequisitos
brew install ollama && ollama serve
ollama pull nomic-embed-text && ollama pull qwen2.5:1.5b

# Instalación
git clone https://github.com/so-fivs/RAG.git && cd RAG
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Pipeline de procesamiento (una vez por lote de PDFs)
python src/ingest/extrac_pdf.py
python src/ingest/chunkers_embedder.py
python src/ingest/image_vision_procesor.py   # requiere llava:7b
python src/ingest/chromadb_ingestor.py
python src/ingest/chromadb_diagnostic.py     # verificar ingesta

# Levantar API
cd src/api && python main.py               # http://127.0.0.1:8000

# Ver métricas MLflow
mlflow ui --backend-store-uri file:./mlruns  # http://127.0.0.1:5000

# Evaluación RAGAS
python evaluation/ragas_eval_ollama.py
```

---

## Endpoints de la API

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/chat` | Consulta RAG con sesión conversacional |
| `GET` | `/api/pictograma/{filename}` | Sirve imágenes GHS |
| `GET` | `/api/download_pdf/{nombre_o_codigo}` | Descarga PDF original |
| `POST` | `/api/reset_context/{session_id}` | Reinicia contexto conversacional |
| `GET` | `/health` | Estado del sistema |
| `GET` | `/` | Interfaz web |

---

## Ramas del Repositorio

| Rama | Descripción |
|------|-------------|
| `main` | Sistema base estable. Pipeline completo + API funcional |
| `experimento-1` | Prototipo inicial. 3 colecciones ChromaDB y chunking básico |
| `experimento-2` | Metadata enriquecida: códigos H/P y CAS |
| `experimento-3` | Retriever híbrido con fuzzy matching y detección de tipo de query |
| `experimento-4` | Integración llava:7b para descripciones semánticas de imágenes |
| `apifinal` | API FastAPI completa: sesiones, MLflow, endpoints de descarga |
| `apigoogledrive` | Integración Google Drive para ingesta automática (en desarrollo) |

---

## Notas Técnicas

- ChromaDB usa similitud coseno (`hnsw:space: cosine`). No soporta `$contains` — los filtros por producto se realizan en memoria.
- Los metadatos de códigos H/P se almacenan como strings serializados en ChromaDB y se parsean en tiempo de consulta.
- El contexto conversacional vive en memoria por sesión y se pierde al reiniciar el servidor.
- La latencia con `qwen2.5:1.5b` en CPU es de 30–240 segundos por consulta.
- Parche aplicado en `local_persistent_hnsw.py` para compatibilidad de `hnswlib==0.8.0` en macOS (`file_handle_count` + `persistence_location`).


