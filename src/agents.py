"""
Three-agent orchestration:
  1. Cache Agent      - instant answer if this exact question was asked before
  2. Search Agent      - retrieves from the ingested PMP books via Vector Search
  3. Fallback Agent     - agentic general-knowledge answer when book results are weak/empty
Each response reports which agent produced it.
"""
import json
import re
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import firestore

from src import config

vertexai.init(project=config.PROJECT_ID, location=config.REGION)
_embed_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)

_metadata_cache: dict[str, dict] = {}   # datapoint_id -> {"text":..., "source":...}
_firestore_client = firestore.Client(project=config.PROJECT_ID)
CACHE_COLLECTION = "answer_cache"

WEAK_RESULT_DISTANCE_THRESHOLD = 0.75  # COSINE_DISTANCE; higher = less similar. Tune based on observed results.
GENERATION_TEMPERATURE = 0.2  # low = consistent, source-grounded answers; not overly robotic


def _call_gemini(prompt: str, system_instruction: str = None) -> str:
    """Thin wrapper around a single Gemini call."""
    model = GenerativeModel(config.GENERATION_MODEL, system_instruction=system_instruction)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=GENERATION_TEMPERATURE),
    )
    return response.text


def _normalize_query(query: str) -> str:
    """Normalize a query for cache key matching (lowercase, collapse whitespace, strip punctuation)."""
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


# ---------- Agent 1: Cache Agent ----------

def check_cache(query: str) -> dict | None:
    """Return a cached result if this exact question was answered before (persisted in Firestore)."""
    key = _normalize_query(query)
    doc = _firestore_client.collection(CACHE_COLLECTION).document(key).get()
    return doc.to_dict() if doc.exists else None


def store_in_cache(query: str, result: dict):
    """Persist the answer in Firestore, keyed by the normalized query, so it survives restarts."""
    key = _normalize_query(query)
    # Don't persist per-request timing/agent fields — those get recomputed/overwritten on cache hits.
    cacheable = {k: v for k, v in result.items() if k not in ("agent_used", "timings_ms", "total_latency_ms")}
    _firestore_client.collection(CACHE_COLLECTION).document(key).set(cacheable)


# ---------- Planner ----------

PLANNER_SYSTEM = """You are a query planner for a PMP exam prep document Q&A system.
Given a user question, classify it into exactly one task_type:
- "lookup": a factual/definitional question answerable from the PMP reference books
- "compare": the user wants to compare/contrast how multiple sources treat the same topic
- "summarize": the user wants an overview/summary of a topic across the books

Respond ONLY with JSON: {"task_type": "...", "reasoning": "..."}"""


def plan_query(user_query: str) -> dict:
    raw = _call_gemini(user_query, system_instruction=PLANNER_SYSTEM)
    cleaned = raw.strip().strip("`").replace("json\n", "")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"task_type": "lookup", "reasoning": "fallback: could not parse planner output"}


# ---------- Agent 2: Search Agent (Vector Search over ingested books) ----------

def _load_metadata_cache():
    global _metadata_cache
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.BUCKET_NAME)
    for blob in bucket.list_blobs(prefix="_metadata/"):
        if not blob.name.endswith(".json"):
            continue
        entries = json.loads(blob.download_as_text())
        for entry in entries:
            _metadata_cache[entry["id"]] = entry


def retrieve_chunks(query: str, top_k: int = 5, doc_filter: str = None) -> list[dict]:
    """Embed the query, find nearest neighbors in Vector Search, resolve them to text/source."""
    if not _metadata_cache:
        _load_metadata_cache()

    query_embedding = _embed_model.get_embeddings([query])[0].values

    aiplatform.init(project=config.PROJECT_ID, location=config.REGION)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=config.VECTOR_INDEX_ENDPOINT_RESOURCE_NAME
    )
    response = endpoint.find_neighbors(
        deployed_index_id=config.DEPLOYED_INDEX_ID,
        queries=[query_embedding],
        num_neighbors=top_k,
    )

    results = []
    for neighbor in response[0]:
        entry = _metadata_cache.get(neighbor.id)
        if entry is None:
            continue
        if doc_filter and entry["source"] != doc_filter:
            continue
        results.append({"text": entry["text"], "source": entry["source"], "score": neighbor.distance})
    return results


def _results_are_weak(chunks: list[dict]) -> bool:
    """Cheap first filter: no chunks, or the best match is too dissimilar to trust."""
    if not chunks:
        return True
    best_distance = min(c["score"] for c in chunks)
    return best_distance > WEAK_RESULT_DISTANCE_THRESHOLD


CONFIDENCE_CHECK_SYSTEM = """You judge whether a drafted answer actually, substantively answers the question using the given context.
Respond with ONLY one word: CONFIDENT or NOT_CONFIDENT.
Use NOT_CONFIDENT in any of these cases:
- The context is thematically related but doesn't specifically cover what was asked (generic match, not the particular
  scenario, formula, or detail requested).
- The drafted answer says, admits, or implies that the context does not contain the information, that it "cannot be
  determined," "is not explicitly covered," or similar — even if that statement is itself accurate. An honest
  non-answer is still NOT_CONFIDENT, because the question deserves an actual answer from elsewhere.
Only respond CONFIDENT if the answer substantively and specifically resolves the question using real content from the context."""


NO_INFO_PHRASES = [
    "do not contain", "does not contain", "doesn't contain",
    "no information about", "no information on",
    "not mention", "does not mention", "doesn't mention",
    "cannot be determined", "can not be determined",
    "not explicitly", "is not covered", "are not covered",
    "not addressed in", "not discussed in",
    "i apologize, but the provided", "the provided context excerpts from",
]


def _answer_admits_no_info(draft_answer: str) -> bool:
    """Deterministic safety net: catch honest 'the context doesn't cover this' phrasing directly,
    rather than relying solely on an LLM judgment call that can be inconsistent."""
    lowered = draft_answer.lower()
    return any(phrase in lowered for phrase in NO_INFO_PHRASES)


def _check_answer_confidence(query: str, chunks: list[dict], draft_answer: str) -> bool:
    """Two-layer filter: deterministic phrase check first (cheap, reliable), then an LLM judgment
    call for subtler cases (generic-but-related content that doesn't explicitly admit a gap)."""
    if _answer_admits_no_info(draft_answer):
        return False

    context = "\n\n".join(c["text"] for c in chunks)
    prompt = (
        f"Question: {query}\n\n"
        f"Context used:\n{context}\n\n"
        f"Drafted answer:\n{draft_answer}\n\n"
        f"Is this draft answer well-supported and specific to the question? Respond CONFIDENT or NOT_CONFIDENT."
    )
    verdict = _call_gemini(prompt, system_instruction=CONFIDENCE_CHECK_SYSTEM).strip().upper()
    return verdict.startswith("CONFIDENT")


COMPARATOR_SYSTEM = """You compare how multiple source documents treat the same topic.
You will be given retrieved excerpts grouped by source document.
Identify agreements, differences, and which source is most restrictive/specific where relevant.
Always cite the source document name for each claim you make."""


def compare_across_documents(query: str, chunks_by_doc: dict[str, list[dict]]) -> str:
    context_blocks = []
    for doc_name, chunks in chunks_by_doc.items():
        excerpt_text = "\n".join(c["text"] for c in chunks)
        context_blocks.append(f"--- Source: {doc_name} ---\n{excerpt_text}")
    context = "\n\n".join(context_blocks)
    prompt = f"Question: {query}\n\nRetrieved excerpts:\n{context}\n\nProvide a grounded comparison with citations."
    return _call_gemini(prompt, system_instruction=COMPARATOR_SYSTEM)


SYNTHESIZER_SYSTEM = """You answer questions strictly using the provided context excerpts from PMP reference books.
Cite the source document for every claim (e.g. "According to pmp-book1.pdf, ...").
If the context doesn't fully answer the question, say what's missing rather than guessing."""


def synthesize_answer(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in chunks)
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
    return _call_gemini(prompt, system_instruction=SYNTHESIZER_SYSTEM)


# ---------- Agent 3: Fallback Agent (agentic general knowledge) ----------

FALLBACK_SYSTEM = """You are a PMP (Project Management Professional) exam expert.
The user's question could not be answered from their uploaded reference books —
either nothing relevant was found, or the match was too weak to trust.
Answer using your own general PMP knowledge instead.
Be clear this answer comes from general knowledge, not their specific uploaded books.
Still be accurate and exam-relevant (PMBOK concepts, ECO domains, agile/predictive/hybrid approaches, etc.)."""


def general_pmp_knowledge(query: str) -> str:
    """Agentic fallback: answer from the model's own PMP knowledge when book search comes up weak/empty."""
    prompt = f"Question: {query}\n\nAnswer using general PMP knowledge:"
    return _call_gemini(prompt, system_instruction=FALLBACK_SYSTEM)


# ---------- Orchestrator ----------

def answer_query(user_query: str) -> dict:
    """
    Three-agent pipeline, with per-stage latency tracked for optimization visibility:
      1. Cache Agent   - return instantly if this exact question was asked before
      2. Search Agent  - retrieve from the ingested PMP books, answer if results are strong
      3. Fallback Agent - agentic general-knowledge answer if results are weak/empty
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def _mark(label: str, t0: float):
        timings[label] = round((time.perf_counter() - t0) * 1000)  # ms

    t0 = time.perf_counter()
    cached = check_cache(user_query)
    _mark("cache_check_ms", t0)

    if cached is not None:
        result = dict(cached)
        result["agent_used"] = "Cache Agent (instant repeat answer)"
        result["timings_ms"] = timings
        result["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000)
        return result

    t0 = time.perf_counter()
    plan = plan_query(user_query)
    _mark("planner_ms", t0)
    task_type = plan.get("task_type", "lookup")

    t0 = time.perf_counter()
    if task_type == "compare":
        chunks = retrieve_chunks(user_query, top_k=10)
    else:
        chunks = retrieve_chunks(user_query, top_k=5)
    _mark("retrieval_ms", t0)

    if _results_are_weak(chunks):
        t0 = time.perf_counter()
        answer = general_pmp_knowledge(user_query)
        _mark("generation_ms", t0)
        result = {
            "task_type": task_type,
            "answer": answer,
            "sources": [],
            "agent_used": "Fallback Agent (general PMP knowledge, not from your books)",
        }
    else:
        t0 = time.perf_counter()
        if task_type == "compare":
            chunks_by_doc: dict[str, list[dict]] = {}
            for c in chunks:
                chunks_by_doc.setdefault(c["source"], []).append(c)
            draft_answer = compare_across_documents(user_query, chunks_by_doc)
        else:
            draft_answer = synthesize_answer(user_query, chunks)
        _mark("generation_ms", t0)

        t0 = time.perf_counter()
        is_confident = _check_answer_confidence(user_query, chunks, draft_answer)
        _mark("confidence_check_ms", t0)

        if is_confident:
            result = {
                "task_type": task_type,
                "answer": draft_answer,
                "sources": sorted({c["source"] for c in chunks}),
                "agent_used": "Search Agent (your uploaded PMP books)",
            }
        else:
            t0 = time.perf_counter()
            answer = general_pmp_knowledge(user_query)
            _mark("fallback_generation_ms", t0)
            result = {
                "task_type": task_type,
                "answer": answer,
                "sources": [],
                "agent_used": "Fallback Agent (general PMP knowledge, not from your books — book match was too generic)",
            }

    result["timings_ms"] = timings
    result["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000)

    store_in_cache(user_query, result)
    return result
