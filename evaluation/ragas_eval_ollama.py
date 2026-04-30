"""
ragas_eval_ollama.py

Evaluación RAGAS completa del sistema RAG FDS usando modelos locales (Ollama).
Métricas: faithfulness, answer_relevancy, context_precision, context_recall,
          answer_correctness, answer_similarity.

Uso:
    # Con API corriendo:
    python evaluation/ragas_eval_ollama.py

    # Solo dataset (sin API, evalúa ground truths):
    python evaluation/ragas_eval_ollama.py --dataset-only
"""

import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional

# ─── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Config ────────────────────────────────────────────────────────────────
API_URL      = "http://localhost:8000/api/chat"
DATASET_PATH = Path(__file__).parent / "ragas_eval_dataset.json"
OUTPUT_PATH  = Path(__file__).parent / "ragas_eval_summary.json"
SESSION_ID   = f"ragas_eval_{int(time.time())}"

# Colores ANSI
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ═══════════════════════════════════════════════════════════════════════════
# MÉTRICAS PROPIAS (sin depender de ragas library para modo offline)
# ═══════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> set:
    """Tokenización simple para métricas de overlap."""
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)


def jaccard_similarity(text1: str, text2: str) -> float:
    """Similitud Jaccard entre dos textos."""
    t1, t2 = tokenize(text1), tokenize(text2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def answer_similarity_score(answer: str, ground_truth: str) -> float:
    """
    Similitud semántica aproximada entre respuesta y ground truth.
    Usa Jaccard sobre bigramas para capturar frases.
    """
    def get_bigrams(text):
        words = list(tokenize(text))
        return set(zip(words, words[1:])) if len(words) > 1 else set()

    bg1 = get_bigrams(answer)
    bg2 = get_bigrams(ground_truth)

    # Jaccard sobre bigramas
    if not bg1 and not bg2:
        return 1.0
    if not bg1 or not bg2:
        return 0.0
    bigram_score = len(bg1 & bg2) / len(bg1 | bg2)

    # Unigram overlap
    unigram_score = jaccard_similarity(answer, ground_truth)

    return (bigram_score * 0.4 + unigram_score * 0.6)


def faithfulness_score(answer: str, contexts: list) -> float:
    """
    Mide si la respuesta está basada en los contextos recuperados.
    Calcula qué porcentaje de tokens del answer aparece en los contextos.
    """
    if not answer or not contexts:
        return 0.0

    answer_tokens = tokenize(answer)
    if not answer_tokens:
        return 0.0

    combined_context = " ".join(contexts)
    context_tokens = tokenize(combined_context)

    # Cuántos tokens del answer están en el contexto
    overlap = len(answer_tokens & context_tokens)
    return overlap / len(answer_tokens)


def context_precision_score(question: str, contexts: list, ground_truth: str) -> float:
    """
    Mide si los contextos recuperados son relevantes para la pregunta.
    Calcula similitud promedio entre cada contexto y el ground truth.
    """
    if not contexts:
        return 0.0

    scores = [jaccard_similarity(ctx, ground_truth) for ctx in contexts]
    return sum(scores) / len(scores) if scores else 0.0


def context_recall_score(contexts: list, ground_truth: str) -> float:
    """
    Mide si los contextos recuperados cubren la información del ground truth.
    Qué porcentaje del ground truth está cubierto por los contextos.
    """
    if not contexts or not ground_truth:
        return 0.0

    gt_tokens = tokenize(ground_truth)
    if not gt_tokens:
        return 0.0

    combined_context = " ".join(contexts)
    context_tokens = tokenize(combined_context)

    overlap = len(gt_tokens & context_tokens)
    return overlap / len(gt_tokens)


def answer_relevancy_score(question: str, answer: str) -> float:
    """
    Mide si la respuesta es relevante para la pregunta.
    Usa overlap de tokens clave entre pregunta y respuesta.
    """
    if not question or not answer:
        return 0.0

    # Palabras clave de la pregunta (ignorar stopwords)
    stopwords = {'qué', 'cuáles', 'cuál', 'cómo', 'es', 'son', 'los', 'las',
                 'del', 'de', 'la', 'el', 'en', 'un', 'una', 'se', 'para',
                 'y', 'o', 'a', 'con', 'que', 'por', 'su', 'sus', 'al', 'le'}

    q_tokens = tokenize(question) - stopwords
    a_tokens = tokenize(answer) - stopwords

    if not q_tokens or not a_tokens:
        return 0.0

    overlap = len(q_tokens & a_tokens)
    # Precision: cuánto del question está en answer
    return overlap / len(q_tokens)


def answer_correctness_score(answer: str, ground_truth: str) -> float:
    """
    Combina similitud + factual correctness.
    Ponderación: 60% similitud semántica + 40% overlap de entidades clave (H codes, CAS, nombres).
    """
    import re

    sim = answer_similarity_score(answer, ground_truth)

    # Extraer entidades específicas de FDS
    def extract_entities(text):
        h_codes  = set(re.findall(r'H\d{3}', text))
        p_codes  = set(re.findall(r'P\d{3}', text))
        cas_nums = set(re.findall(r'\d{2,6}-\d{2}-\d', text))
        return h_codes | p_codes | cas_nums

    gt_entities  = extract_entities(ground_truth)
    ans_entities = extract_entities(answer)

    if gt_entities:
        entity_recall = len(gt_entities & ans_entities) / len(gt_entities)
    else:
        entity_recall = sim  # si no hay entidades específicas, usar similitud general

    return sim * 0.6 + entity_recall * 0.4


# ═══════════════════════════════════════════════════════════════════════════
# QUERY RAG API
# ═══════════════════════════════════════════════════════════════════════════

def query_rag(question: str, product_hint: Optional[str] = None) -> dict:
    """Consulta la API RAG y retorna el resultado completo."""
    query = question
    if product_hint:
        # Enriquecer query con producto si está disponible
        query = f"{question}" if product_hint.lower() in question.lower() else question

    try:
        resp = requests.post(
            API_URL,
            json={"query": query, "session_id": SESSION_ID, "use_context": True},
            timeout=None  # sin timeout — puede demorar en CPU
        )
        if resp.status_code != 200:
            return {"error": f"HTTP {resp.status_code}", "answer": "", "sources": [], "structured_metadata": {}}
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API no disponible", "answer": "", "sources": [], "structured_metadata": {}}
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": [], "structured_metadata": {}}


# ═══════════════════════════════════════════════════════════════════════════
# EVALUACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dataset(dataset: list, use_api: bool = True) -> list:
    """Evalúa cada pregunta del dataset y retorna resultados detallados."""
    results = []
    total = len(dataset)

    for i, case in enumerate(dataset, 1):
        question    = case["question"]
        ground_truth = case["ground_truth"]
        product     = case.get("product", "")
        seccion     = case.get("seccion", "")
        tipo        = case.get("tipo_query", "general")

        print(f"\n{BOLD}{'─'*70}{RESET}")
        print(f"{CYAN}[{i}/{total}] {tipo.upper()}{RESET}")
        print(f"{BOLD}❓ {question}{RESET}")

        if use_api:
            t0 = time.time()
            rag_result = query_rag(question, product)
            elapsed = (time.time() - t0) * 1000

            if "error" in rag_result and not rag_result.get("answer"):
                print(f"{RED}❌ Error: {rag_result['error']}{RESET}")
                results.append({
                    "question": question, "ground_truth": ground_truth,
                    "product": product, "seccion": seccion, "tipo": tipo,
                    "error": rag_result["error"],
                    "metrics": {k: 0.0 for k in ["faithfulness", "answer_relevancy",
                                                   "context_precision", "context_recall",
                                                   "answer_similarity", "answer_correctness"]}
                })
                continue

            answer   = rag_result.get("answer", "")
            sources  = rag_result.get("sources", [])
            contexts = [s.get("content_preview", "") for s in sources if s.get("content_preview")]
            meta     = rag_result.get("structured_metadata") or {}

        else:
            # Modo offline: evaluar solo ground truth vs ground truth (baseline)
            answer   = ground_truth
            contexts = [ground_truth]
            elapsed  = 0
            meta     = {}

        # ── Calcular métricas ──────────────────────────────────────────────
        metrics = {
            "faithfulness":      round(faithfulness_score(answer, contexts), 3),
            "answer_relevancy":  round(answer_relevancy_score(question, answer), 3),
            "context_precision": round(context_precision_score(question, contexts, ground_truth), 3),
            "context_recall":    round(context_recall_score(contexts, ground_truth), 3),
            "answer_similarity": round(answer_similarity_score(answer, ground_truth), 3),
            "answer_correctness":round(answer_correctness_score(answer, ground_truth), 3),
        }

        # ── Mostrar resultado ──────────────────────────────────────────────
        if answer:
            preview = answer[:200] + ("..." if len(answer) > 200 else "")
            print(f"\n{GREEN}💬 {preview}{RESET}")
        else:
            print(f"{YELLOW}⚠️  Sin respuesta{RESET}")

        print(f"\n{BOLD}📊 Métricas:{RESET}")
        for metric, val in metrics.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            color = GREEN if val >= 0.6 else (YELLOW if val >= 0.4 else RED)
            print(f"  {metric:<22} {color}{val:.3f}{RESET} {bar}")

        # H/P codes detected
        h_codes = [c.get("codigo") for c in meta.get("codigos_h", [])]
        if h_codes:
            print(f"  ⚠️  Códigos H detectados: {h_codes}")

        results.append({
            "question":     question,
            "answer":       answer,
            "ground_truth": ground_truth,
            "product":      product,
            "seccion":      seccion,
            "tipo":         tipo,
            "contexts_count": len(contexts),
            "latency_ms":   round(elapsed),
            "metrics":      metrics,
        })

    return results


def print_summary(results: list):
    """Imprime resumen estadístico de todas las métricas."""
    print(f"\n{'═'*70}")
    print(f"{BOLD}{BLUE}         📊 RESUMEN EVALUACIÓN RAGAS — {datetime.now().strftime('%Y-%m-%d %H:%M')}{RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")

    valid = [r for r in results if "error" not in r]
    errors = len(results) - len(valid)

    print(f"\n  Queries evaluadas : {len(results)}")
    print(f"  Exitosas          : {len(valid)}")
    print(f"  Errores           : {errors}")

    if not valid:
        print(f"{RED}  Sin resultados válidos para analizar{RESET}")
        return {}

    metric_names = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "answer_similarity", "answer_correctness"]

    agg = {}
    print(f"\n  {'Métrica':<25} {'Min':>6} {'Max':>6} {'Prom':>6} {'Interpretación'}")
    print(f"  {'─'*65}")

    interpretations = {
        "faithfulness":      "¿respuesta basada en documentos?",
        "answer_relevancy":  "¿respuesta relevante a la pregunta?",
        "context_precision": "¿chunks recuperados son correctos?",
        "context_recall":    "¿chunks cubren el ground truth?",
        "answer_similarity": "¿similitud respuesta vs ground truth?",
        "answer_correctness":"¿correctitud factual (H/P/CAS)?",
    }

    for metric in metric_names:
        vals = [r["metrics"][metric] for r in valid]
        mn, mx, avg = min(vals), max(vals), sum(vals)/len(vals)
        color = GREEN if avg >= 0.6 else (YELLOW if avg >= 0.4 else RED)
        agg[metric] = {"min": round(mn,3), "max": round(mx,3), "avg": round(avg,3)}
        interp = interpretations.get(metric, "")
        print(f"  {metric:<25} {mn:>6.3f} {mx:>6.3f} {color}{avg:>6.3f}{RESET}  {interp}")

    # Score compuesto (promedio de todas las métricas)
    composite = sum(agg[m]["avg"] for m in metric_names) / len(metric_names)
    color = GREEN if composite >= 0.6 else (YELLOW if composite >= 0.4 else RED)
    print(f"\n  {'SCORE COMPUESTO':<25} {'':>6} {'':>6} {color}{composite:>6.3f}{RESET}  ← métrica principal")

    # Tabla por tipo de query
    tipos = {}
    for r in valid:
        t = r.get("tipo", "general")
        if t not in tipos:
            tipos[t] = []
        tipos[t].append(r["metrics"]["answer_correctness"])

    print(f"\n  {'Tipo query':<20} {'Avg correctness':>16} {'N':>5}")
    print(f"  {'─'*45}")
    for tipo, vals in sorted(tipos.items()):
        avg = sum(vals)/len(vals)
        color = GREEN if avg >= 0.6 else (YELLOW if avg >= 0.4 else RED)
        print(f"  {tipo:<20} {color}{avg:>16.3f}{RESET} {len(vals):>5}")

    return {**agg, "composite": round(composite, 3)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del sistema FDS RAG")
    parser.add_argument("--dataset-only", action="store_true",
                        help="Evaluar solo el dataset sin llamar a la API")
    parser.add_argument("--questions", type=int, default=None,
                        help="Número de preguntas a evaluar (default: todas)")
    args = parser.parse_args()

    print(f"\n{'═'*70}")
    print(f"{BOLD}   🧪 EVALUACIÓN RAGAS — SISTEMA RAG FDS{RESET}")
    print(f"   Dataset: {DATASET_PATH}")
    print(f"   Output:  {OUTPUT_PATH}")
    print(f"{'═'*70}")

    # Cargar dataset
    if not DATASET_PATH.exists():
        print(f"{RED}❌ Dataset no encontrado: {DATASET_PATH}{RESET}")
        print(f"   Crea el archivo primero con las preguntas y ground truths.")
        sys.exit(1)

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.questions:
        dataset = dataset[:args.questions]

    print(f"\n  📋 Dataset cargado: {len(dataset)} preguntas")

    use_api = not args.dataset_only

    if use_api:
        # Verificar API
        try:
            h = requests.get("http://localhost:8000/health", timeout=5)
            info = h.json()
            rag_ok = info.get("rag_initialized", False)
            print(f"  🌐 API: {GREEN}conectada{RESET} | RAG: {'✅' if rag_ok else '❌'}")
            if not rag_ok:
                print(f"  {YELLOW}⚠️  RAG no inicializado — las respuestas pueden fallar{RESET}")
        except Exception:
            print(f"  {RED}❌ API no disponible en {API_URL}{RESET}")
            print(f"  Ejecuta: cd src/api && python main.py")
            print(f"  O usa --dataset-only para evaluar sin API\n")
            sys.exit(1)
    else:
        print(f"  📊 Modo: evaluación offline (ground truth vs ground truth)")

    # Ejecutar evaluación
    print(f"\n  ⏳ Iniciando evaluación...")
    t_start = time.time()

    results = evaluate_dataset(dataset, use_api=use_api)

    total_time = time.time() - t_start

    # Resumen
    agg_metrics = print_summary(results)

    # Guardar resultados
    output = {
        "timestamp":      datetime.now().isoformat(),
        "mode":           "api" if use_api else "offline",
        "dataset_size":   len(dataset),
        "total_time_s":   round(total_time, 1),
        "avg_latency_ms": round(sum(r.get("latency_ms", 0) for r in results) / len(results)) if results else 0,
        "aggregate_metrics": agg_metrics,
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ⏱  Tiempo total: {total_time:.1f}s")
    print(f"  💾 Resultados guardados: {OUTPUT_PATH}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()