"""
main.py — FastAPI con contexto conversacional por sesión.
"""
import sys
import time
import logging
import asyncio
import traceback
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import unquote
from jinja2 import FileSystemLoader, Environment
from starlette.templating import Jinja2Templates
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent.parent
src_root     = project_root / "src"
backend_root = src_root / "backend"

for p in [str(project_root), str(src_root), str(backend_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

warnings.filterwarnings("ignore")

from config import ProjectConfig
from rag_pipeline import RAGPipeline, ConversationalContext
try:
    from s3_sync import sync_vector_db
    S3_DISPONIBLE = True
except ImportError:
    S3_DISPONIBLE = False

# ── Modelos Pydantic ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:      str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer:              str
    sources:             List[Dict[str, Any]]
    structured_metadata: Optional[Dict[str, Any]] = None
    pictogramas:         List[str] = []
    metrics:             Dict[str, float] = {}

# ── App ─────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG FDS System", version="2.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En desarrollo puedes usar "*" para probar
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
config      = ProjectConfig()
db_path     = config.get_folder('vector_db')
images_path = config.get_folder('images')
pdf_path    = config.get_folder('raw_documents')

templates_dir = src_root / "frontend"

if templates_dir.exists():
    j2_env = Environment(loader=FileSystemLoader(str(templates_dir)))
    j2_env.globals.update({}) 
    templates = Jinja2Templates(directory=str(templates_dir))
    templates.env = j2_env
    templates.env.cache = None 
else:
    templates = None

rag_pipeline: Optional[RAGPipeline] = None
sessions: Dict[str, ConversationalContext] = {}

MAX_TIMEOUT = 600

# ── Globales ────────────────────────────────────────────────────────────
rag_pipeline: Optional[RAGPipeline] = None
sessions: Dict[str, ConversationalContext] = {}   # contexto por sesión

# ── Middleware logging ───────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n→ {request.method} {request.url}")
    t0 = time.time()
    try:
        response = await call_next(request)
        print(f"← {response.status_code} ({(time.time()-t0)*1000:.0f}ms)")
        return response
    except Exception as e:
        print(f"← 500 MIDDLEWARE ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ── Startup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    logging.getLogger('chromadb').setLevel(logging.ERROR)
    print("\n" + "="*70 + "\nIniciando RAG System v2...\n" + "="*70)

    db_vacia = not db_path.exists() or not any(db_path.rglob("*.sqlite3"))
    if S3_DISPONIBLE and db_vacia:
        print("  ☁️  vector_db/ vacío o inexistente — descargando de S3...")
        sync_vector_db(config, direccion="pull")
    elif not S3_DISPONIBLE and db_vacia:
        raise Exception(f"ChromaDB no encontrado en {db_path} y S3 no disponible")

    try:
        rag_pipeline = RAGPipeline(config)
        print("Sistema RAG listo.\n" + "="*70)
    except Exception:
        traceback.print_exc()
        raise

@app.get("/health")
async def health():
    return {
        "status":           "ok",
        "rag_initialized":  rag_pipeline is not None,
        "db_exists":        db_path.exists(),
        "s3_disponible":    S3_DISPONIBLE,
        "sessions_activas": list(sessions.keys()),
    }
 
# ── Lógica principal ─────────────────────────────────────────────────────
async def run_chat_logic(request: ChatRequest) -> Dict[str, Any]:
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG no inicializado")

    # Obtener o crear contexto de sesión
    sid = request.session_id or "default"
    if sid not in sessions:
        sessions[sid] = ConversationalContext()
        print(f"  [SESIÓN] Nueva: {sid}")

    ctx = sessions[sid]

    try:
        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: rag_pipeline.run(request.query, session_ctx=ctx)
        )

        sources = [{
            "producto":   str(s.get('producto',  'N/A')),
            "seccion":    str(s.get('seccion',   'N/A')),
            "similarity": float(s.get('similarity', 0.0)),
            "tipo":       str(s.get('tipo',      'N/A')),
        } for s in (response.sources or [])]

        meta = response.structured_metadata or {}
        structured_metadata = {
            "producto":      str(meta.get('producto',      '')),
            "fabricante":    str(meta.get('fabricante',    '')),
            "cas_principal": str(meta.get('cas_principal', '')),
            "pictogramas":   meta.get('pictogramas', []),
            "h_codes": [{"codigo": c['codigo'], "descripcion": c['descripcion']}
                        for c in meta.get('h_codes', [])],
            "p_codes": [{"codigo": c['codigo'], "descripcion": c['descripcion']}
                        for c in meta.get('p_codes', [])],
        }

        metrics = {
            "latency_ms":     float(response.latency_ms),
            "avg_similarity": float(response.metrics.avg_similarity),
            "top_similarity": float(response.metrics.top_similarity),
            "total_chunks":   float(response.metrics.total_chunks),
        }

        # MLflow opcional
        try:
            import mlflow
            mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
            mlflow.set_experiment("RAG_FDS_System")
            with mlflow.start_run(run_name=f"q_{int(time.time())}"):
                mlflow.log_param("query", request.query[:100])
                mlflow.log_param("session_id", sid)
                mlflow.log_metrics(metrics)
        except Exception as e:
            print(f"  [MLflow skip] {e}")

        return {
            "answer":              str(response.answer),
            "sources":             sources,
            "structured_metadata": structured_metadata,
            "pictogramas":         response.pictogramas or [],
            "metrics":             metrics,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── Endpoints ────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)

async def chat_endpoint(request: ChatRequest):
    try:
        result = await asyncio.wait_for(run_chat_logic(request), timeout=MAX_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout")
    print(f"Llegó query: {request.query}")
    return ChatResponse(**result)


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Bypass total de Jinja2. 
    Leemos el archivo index.html físicamente y lo enviamos.
    """
    # Intentamos localizar el archivo index.html
    index_path = src_root / "frontend" / "index.html"
    
    try:
        if index_path.exists():
            # Leemos el contenido del archivo directamente
            content = index_path.read_text(encoding="utf-8")
            return HTMLResponse(content=content)
        else:
            return HTMLResponse(
                content=f"<h1>Error</h1><p>No se encontró el archivo en: {index_path}</p>", 
                status_code=404
            )
    except Exception as e:
        print(f"Error cargando index.html: {e}")
        return HTMLResponse(content=f"<h1>Error Interno</h1><p>{str(e)}</p>", status_code=500)

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "rag_initialized": rag_pipeline is not None,
        "db_exists":       db_path.exists(),
        "sessions_activas": list(sessions.keys()),
    }


@app.get("/api/download_pdf/{nombre}")
async def download_pdf(nombre: str):
    term    = unquote(nombre).strip().lower()
    matches = [p for p in pdf_path.glob("*.pdf") if term in p.stem.lower()]
    if not matches:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    return FileResponse(matches[0], media_type='application/pdf', filename=matches[0].name)


@app.get("/api/pictograma/{filename}")
async def get_pictograma(filename: str):
    fp = images_path / unquote(filename)
    if not fp.exists():
        raise HTTPException(status_code=404, detail=f"No encontrada: {filename}")
    return FileResponse(fp)


@app.post("/api/reset_context/{session_id}")
async def reset_context(session_id: str):
    if session_id in sessions:
        sessions[session_id].reset()
        print(f"  [RESET] Sesión {session_id} reiniciada")
    return {"message": f"Contexto de {session_id} reiniciado"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)