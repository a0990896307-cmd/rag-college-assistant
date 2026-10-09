"""FastAPI service: ingest, query, inspect.

Answer generation is pluggable. With no LLM configured the service still
returns the retrieved passages with citations, which is the honest
default: a RAG box that silently invents an answer is worse than one that
shows its evidence. The `grounded` flag in every response tells you which
mode produced the text.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .chunking import Chunker
from .embeddings import get_embedder
from .ingestion import Ingestor, SUPPORTED
from .retrieval import Retriever
from .store import Store

DB_PATH = os.getenv("RAG_DB", "rag.db")
TOP_K = int(os.getenv("RAG_TOP_K", "6"))

app = FastAPI(
    title="RAG Knowledge Assistant",
    description=(
        "Section-aware document ingestion and cited retrieval. "
        "Every answer carries the document, page and section it came from."
    ),
    version="0.2.0",
)

store = Store(DB_PATH)
embedder = get_embedder()
ingestor = Ingestor(store, embedder, Chunker())
retriever = Retriever(store, embedder)

WRONG_TYPE = "; ".join(sorted(SUPPORTED))


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=25)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|vector|bm25)$")


class SearchResponse(BaseModel):
    answer: str
    grounded: bool
    degraded_embedder: bool
    sources: list[dict]


def _compose(hits: list) -> str:
    """Extractive fallback: quote the passages rather than paraphrase them.

    Nothing is invented here, so every sentence in the answer is traceable
    to a citation. This is also the mode used for evaluation, since a
    generative answer cannot be scored for factual accuracy without a
    reference.
    """
    if not hits:
        return (
            "Nothing in the indexed documents matches that question. "
            "Try wording it with terms used in the source material."
        )
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(f"[{i}] {h.text}\n    — {h.citation}")
    return "\n\n".join(parts)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunks": store.count(),
        "documents": len(store.documents()),
        "embedder": "local" if not getattr(embedder, "degraded", False) else "hashing-fallback",
        "dimension": embedder.dimension,
    }


@app.get("/documents")
def documents() -> dict:
    return {"documents": store.documents()}


@app.delete("/documents/{name}")
def delete_document(name: str) -> dict:
    store.delete_document(name)
    return {"deleted": name, "chunks": store.count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported type {suffix!r}; accepted: {WRONG_TYPE}",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        return ingestor.ingest_bytes(file.filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/search", response_model=SearchResponse)
def search(req: QueryRequest) -> SearchResponse:
    result = retriever.retrieve(
        req.question, top_k=req.top_k, strategy=req.strategy
    )
    return SearchResponse(
        answer=_compose(result.hits),
        grounded=True,
        degraded_embedder=result.degraded,
        sources=[h.to_dict() for h in result.hits],
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Knowledge Assistant</title>
<style>
 :root{--bg:#faf9f7;--fg:#0e1015;--mut:#6b7280;--line:#e5e3df;--acc:#2563eb;--ok:#16a34a;--bad:#dc2626}
 *{box-sizing:border-box}
 body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
      background:var(--bg);color:var(--fg)}
 header{padding:20px 28px;border-bottom:1px solid var(--line);display:flex;
        align-items:baseline;gap:14px;flex-wrap:wrap}
 h1{font-size:17px;margin:0;font-weight:650}
 .stat{font-size:12px;color:var(--mut)}
 .stat b{color:var(--fg);font-weight:600}
 main{max-width:980px;margin:0 auto;padding:28px}
 form{display:flex;gap:10px;margin-bottom:22px}
 input[type=text]{flex:1;padding:11px 14px;border:1px solid var(--line);border-radius:8px;
   font-size:15px;background:#fff;color:var(--fg)}
 input[type=text]:focus{outline:2px solid var(--acc);outline-offset:-1px;border-color:transparent}
 select{padding:11px 12px;border:1px solid var(--line);border-radius:8px;background:#fff;
   font-size:14px;color:var(--fg)}
 button{padding:11px 20px;border:0;border-radius:8px;background:var(--acc);color:#fff;
   font-size:15px;font-weight:550;cursor:pointer}
 button:disabled{opacity:.55;cursor:not-allowed}
 .cite{border:1px solid var(--line);border-radius:9px;padding:13px 15px;margin-bottom:10px;
   background:#fff}
 .cite .meta{font-size:12px;color:var(--mut);margin-bottom:5px}
 .cite .meta b{color:var(--acc);font-weight:600}
 .cite .txt{font-size:14px}
 .score{float:right;font-size:11px;color:var(--mut)}
 .banner{padding:10px 14px;border-radius:8px;font-size:13px;margin-bottom:18px}
 .warn{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412}
 .empty{color:var(--mut);font-size:14px;padding:24px 0}
 code{background:#f1f0ed;padding:1px 5px;border-radius:4px;font-size:12px}
</style></head><body>
<header>
 <h1>RAG Knowledge Assistant</h1>
 <span class="stat">chunks <b id="n">…</b></span>
 <span class="stat">docs <b id="d">…</b></span>
 <span class="stat">embedder <b id="e">…</b></span>
</header>
<main>
 <div id="b"></div>
 <form id="f">
  <input type="text" id="q" placeholder="Ask a question, e.g. &quot;when is the first instalment due?&quot;" autofocus>
  <select id="s">
   <option value="hybrid">hybrid</option>
   <option value="vector">vector</option>
   <option value="bm25">bm25</option>
  </select>
  <button id="go">Search</button>
 </form>
 <div id="out"></div>
</main>
<script>
const $=i=>document.getElementById(i);
async function health(){
  try{
    const r=await fetch('/health').then(r=>r.json());
    $('n').textContent=r.chunks; $('d').textContent=r.documents;
    $('e').textContent=r.embedder;
    $('b').innerHTML = r.embedder.includes('fallback')
      ? '<div class="banner warn">Hashing fallback embedder in use. Lexical matching '
        + 'works, paraphrase matching is weak. Install sentence-transformers for full quality.</div>' : '';
  }catch(e){$('n').textContent='?'}
}
$('f').onsubmit=async e=>{
  e.preventDefault();
  const q=$('q').value.trim(); if(!q)return;
  $('go').disabled=true; $('out').innerHTML='<div class="empty">Searching…</div>';
  try{
    const r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,top_k:6,strategy:$('s').value})}).then(r=>r.json());
    if(!r.sources||!r.sources.length){
      $('out').innerHTML='<div class="empty">No matching passages. Try wording it with '
        +'terms from the source documents.</div>';
    }else{
      $('out').innerHTML=r.sources.map((s,i)=>
        '<div class="cite"><span class="score">score '+s.score+'</span>'
        +'<div class="meta"><b>['+(i+1)+']</b> '+esc(s.citation)+'</div>'
        +'<div class="txt">'+esc(s.text)+'</div></div>').join('');
    }
  }catch(err){$('out').innerHTML='<div class="empty">Request failed: '+esc(err)+'</div>'}
  $('go').disabled=false;
};
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
health();
</script></body></html>"""


def run() -> None:
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run()