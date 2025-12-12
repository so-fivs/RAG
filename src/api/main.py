from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from urllib.parse import unquote
import sys
import logging
import time
import traceback
from pathlib import Path
import mlflow
import warnings
warnings.filterwarnings("ignore", message="Add of existing embedding ID")
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from config import ProjectConfig
from src.rag.generator import RAGGenerator
config = ProjectConfig()
db_path = config.get_folder('vector_db')
images_path = config.get_folder('images')
pdf_path = config.get_folder('raw_documents')

# -----------------------------------------------------------------------------
# MODELOS
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    use_context: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    structured_metadata: Optional[Dict[str, Any]] = None
    pictogramas: List[str] = []
    metrics: Dict[str, float] = {}
    fds_reference: Optional[Dict[str, Any]] = None

# -----------------------------------------------------------------------------
# FASTAPI INITIAL SETUP
# -----------------------------------------------------------------------------
app = FastAPI(
    title="RAG FDS System",
    description="Sistema RAG para Fichas de Datos de Seguridad",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent.parent

# -----------------------------------------------------------------------------
# INICIALIZACIÓN GLOBAL
# -----------------------------------------------------------------------------
config = ProjectConfig()
rag_generator = None
sessions: Dict[str, Dict[str, Any]] = {}

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("RAG_FDS_System")

MAX_TIMEOUT = 600 

# -----------------------------------------------------------------------------
# (OPCIONAL) Middleware de logging de peticiones / respuestas (útil para debug)
# -----------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n➡️ Incoming: {request.method} {request.url}")
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000.0
    print(f"⬅️ Response status={response.status_code} completed_in={duration:.0f}ms")
    return response

# -----------------------------------------------------------------------------
# STARTUP
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global rag_generator
    sessions.clear()
    logging.getLogger('chromadb').setLevel(logging.ERROR)

    print("\n" + "="*80)
    print("Iniciando RAG System...")
    print("="*80)

    try:
        print("\n[1/3] Inicializando RAG Generator...")
        rag_generator = RAGGenerator(config, use_gemini=False)

        print("[2/3] Verificando Ollama...")
        import ollama
        models = ollama.list()
        print(f"Ollama OK - {len(models.get('models', []))} modelos disponibles")

        print("[3/3] Verificando ChromaDB...")
        if not db_path.exists():
            raise Exception(f"ChromaDB no encontrado en {db_path}")

        print("\n" + "="*80)
        print("Sistema RAG listo.")
        print("="*80)

    except Exception as e:
        print("\nERROR EN STARTUP:")
        traceback.print_exc()
        raise

# -----------------------------------------------------------------------------
# Lógica principal que genera la respuesta (devuelve un dict serializable)
# -----------------------------------------------------------------------------
async def run_chat_logic(request: ChatRequest) -> Dict[str, Any]:
    """
    Ejecuta la lógica del RAG y retorna un diccionario (resp_dict) listo para JSON/Pydantic.
    Esta función devuelve siempre un dict (no una tupla).
    """
    print("\n" + "="*80)
    print(f"Nueva consulta de sesión: {request.session_id}")
    print(f"Query: {request.query}")
    print("="*80)

    if not rag_generator:
        raise HTTPException(status_code=503, detail="RAG no inicializado")

    # Crear contexto de sesión liviano si no existe
    if request.session_id not in sessions:
        print(f"Creando nueva sesión: {request.session_id}")
        try:
            session_ctx = rag_generator.context.__class__()  # ConversationalContext()
        except Exception:
            session_ctx = rag_generator.context
        sessions[request.session_id] = {
            "context": session_ctx,
            "created_at": time.time()
        }

    # Reusar generador global y asignar contexto de sesión
    generator = rag_generator
    session_ctx = sessions[request.session_id]["context"]
    generator.context = session_ctx
    run_name = f"query_{int(time.time())}"

    try:
        start_ts = time.time()
        # generate_response es función síncrona en tu generator; se llama aquí
        response = generator.generate_response(request.query, use_context=request.use_context)
        total_elapsed_ms = (time.time() - start_ts) * 1000.0
        # Normalizar sources
        normalized_sources: List[Dict[str, Any]] = []
        for s in (response.sources or []):
            try:
                normalized_sources.append({
                    "producto": str(s.get('producto', 'N/A')),
                    "seccion": str(s.get('seccion', 'N/A')),
                    "similarity": float(s.get('similarity', 0.0) or 0.0),
                    "tipo": str(s.get('tipo', 'N/A')),
                    "chunk_id": str(s.get('chunk_id', '')),
                    "content_preview": str(s.get('content_preview', '') or "")[:1000]
                })
            except Exception:
                # en caso de estructura inesperada, agregar fallback mínimo
                normalized_sources.append({
                    "producto": "N/A",
                    "seccion": "N/A",
                    "similarity": 0.0,
                    "tipo": "N/A",
                    "chunk_id": "",
                    "content_preview": ""
                })

        # Normalizar structured_metadata: solo dict o None
        structured_metadata = response.structured_metadata if isinstance(response.structured_metadata, dict) else None

        # Normalizar pictogramas siempre lista
        pictogramas = list(response.pictogramas) if response.pictogramas else []

        # Normalizar metrics garantizando tipos numéricos
        retrieval_metrics = getattr(response, "retrieval_metrics", {}) or {}
        metrics = {
            "latency_ms": float(getattr(response, "latency_ms", 0.0) or 0.0),
            "avg_similarity": float(retrieval_metrics.get("avg_similarity", 0.0) or 0.0),
            "top_similarity": float(retrieval_metrics.get("top_similarity", 0.0) or 0.0),
            "total_chunks": int(retrieval_metrics.get("total_chunks", 0) or 0)
        }
        
        fds_reference = response.fds_reference if hasattr(response, 'fds_reference') else None
        if fds_reference and isinstance(fds_reference, dict):
            fds_reference = {
                "producto": str(fds_reference.get('producto', 'N/A')),
                "fabricante": str(fds_reference.get('fabricante', 'N/A')),
                "codigo": str(fds_reference.get('codigo', 'N/A')),
                "fecha": str(fds_reference.get('fecha', 'N/A'))
            }
        else:
            fds_reference = None

        resp_dict: Dict[str, Any] = {
            "answer": str(response.answer or ""),
            "sources": normalized_sources,
            "structured_metadata": structured_metadata,
            "pictogramas": pictogramas,
            "fds_reference": fds_reference,
            "metrics": metrics,
        }

        # Print de depuración: muestra exactamente lo que enviamos al frontend
        print("\n===== RESPUESTA FINAL ENVIADA AL FRONTEND =====")
        print(resp_dict)
        print("================================================\n")
        print(f"[TIMING] generator_latency_ms={metrics['latency_ms']:.0f}  total_request_ms={total_elapsed_ms:.0f}")

        # MLflow: en bloque try/except para no afectar respuesta final
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("query", request.query[:100])
                mlflow.log_param("session_id", request.session_id)
                mlflow.log_metric("latency_ms", metrics["latency_ms"])
                mlflow.log_metric("avg_similarity", metrics["avg_similarity"])
                mlflow.log_metric("top_similarity", metrics["top_similarity"])
                mlflow.log_metric("total_chunks", metrics["total_chunks"])
                if structured_metadata:
                    if "codigos_h" in structured_metadata:
                        mlflow.log_metric("codigos_h_count", len(structured_metadata.get("codigos_h", [])))
                    if "codigos_p" in structured_metadata:
                        mlflow.log_metric("codigos_p_count", len(structured_metadata.get("codigos_p", [])))
        except Exception as ml_e:
            print("MLflow warning:", ml_e)

        return resp_dict

    except Exception as e:
        print("\nERROR CRÍTICO EN run_chat_logic:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})
    

# -----------------------------------------------------------------------------
# ENDPOINT /api/chat UNIFICADO + TIMEOUT
# -----------------------------------------------------------------------------
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal: ejecuta run_chat_logic con timeout y retorna ChatResponse.
    """
    try:
        resp_dict = await asyncio.wait_for(run_chat_logic(request), timeout=MAX_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout: la consulta tardó demasiado.")
    except HTTPException:
        raise
    except Exception as e:
        print("ERROR en chat_endpoint:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    # Asegurar que resp_dict es un mapping (no tupla)
    if isinstance(resp_dict, tuple):
        if len(resp_dict) == 1 and isinstance(resp_dict[0], dict):
            resp_dict = resp_dict[0]
        else:
            raise HTTPException(status_code=500, detail="Respuesta inesperada (tupla) del procesador")

    # ✅ FIX: Extraer fds_reference del resp_dict
    fds_reference = resp_dict.get("fds_reference", None)
    if fds_reference and isinstance(fds_reference, dict):
        fds_reference = {
            "producto": str(fds_reference.get('producto', 'N/A')),
            "fabricante": str(fds_reference.get('fabricante', 'N/A')),
            "codigo": str(fds_reference.get('codigo', 'N/A')),
            "fecha": str(fds_reference.get('fecha', 'N/A'))
        }
    else:
        fds_reference = None

    # Validación/normalización final antes del retorno
    resp_dict_final = {
        "answer": str(resp_dict.get("answer", "") or ""),
        "sources": resp_dict.get("sources", []) or [],
        "structured_metadata": resp_dict.get("structured_metadata", None),
        "pictogramas": resp_dict.get("pictogramas", []) or [],
        "fds_reference": fds_reference,  # ✅ Ahora está definida correctamente
        "metrics": {
            "latency_ms": float(resp_dict.get("metrics", {}).get("latency_ms", 0.0) or 0.0),
            "avg_similarity": float(resp_dict.get("metrics", {}).get("avg_similarity", 0.0) or 0.0),
            "top_similarity": float(resp_dict.get("metrics", {}).get("top_similarity", 0.0) or 0.0),
            "total_chunks": int(resp_dict.get("metrics", {}).get("total_chunks", 0) or 0)
        }
    }

    return ChatResponse(**resp_dict_final)
# -----------------------------------------------------------------------------
# OTROS ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/")
async def root(request: Request):
    if not templates:
        return {"error": "Templates no configurados"}
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_initialized": rag_generator is not None,
        "sessions": list(sessions.keys())
    }
@app.get("/api/download_pdf/{nombre_o_codigo}")
async def download_pdf(nombre_o_codigo: str):
    try:
        search_term = unquote(nombre_o_codigo).strip().lower()
        
        print(f"[PDF] Buscando: '{search_term}'")
        
        # Buscar por código O por nombre de producto
        matching_files = []
        for pdf in pdf_path.glob("*.pdf"):
            filename_lower = pdf.stem.lower()
            # Buscar por código o por palabra clave (epóxico, uretano, etc.)
            if search_term in filename_lower or any(word in filename_lower for word in search_term.split()):
                matching_files.append(pdf)
        
        if not matching_files:
            print(f"[PDF] No encontrado. Archivos disponibles:")
            for f in list(pdf_path.glob("*.pdf"))[:5]:
                print(f"  - {f.name}")
            raise HTTPException(status_code=404, detail="PDF no encontrado")
        
        print(f"[PDF] ✓ Enviando: {matching_files[0].name}")
        return FileResponse(matching_files[0], media_type='application/pdf', filename=matching_files[0].name)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PDF ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/reset_context/{session_id}")
async def reset_context(session_id: str):
    if session_id not in sessions:
        return {"message": "Sesión no encontrada"}

    ctx = sessions[session_id]["context"]
    if hasattr(ctx, "clear_product"):
        try:
            ctx.clear_product()
        except Exception:
            pass
    if hasattr(ctx, "reset"):
        try:
            ctx.reset()
        except Exception:
            pass

    return {"message": f"Contexto de {session_id} reiniciado"}

@app.get("/api/pictograma/{filename}")
async def get_pictograma(filename: str):
    try:
        from urllib.parse import unquote
        filename = unquote(filename)
        file_path = images_path / filename
        
        print(f"\n[DEBUG PICTOGRAMA]")
        print(f"  Filename: {filename}")
        print(f"  Path: {file_path}")
        print(f"  ¿Existe? {file_path.exists()}")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {filename}")
        
        return FileResponse(file_path)
    except HTTPException:
        raise
    except Exception as e:
        print(f"  ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# MAIN (ejecución local)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor en http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)