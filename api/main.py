from __future__ import annotations

import sys
from pathlib import Path

# Make sure the engine is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_stability_engine.adapters import AnthropicChatAdapter, OpenAIChatAdapter
from agent_stability_engine.engine.embeddings import EmbeddingProvider
from agent_stability_engine.runners.benchmark import run_benchmark_suite

SUITE_PATH = Path(__file__).parent.parent / "examples" / "benchmarks" / "large_suite.json"

app = FastAPI(title="Stabilium API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    provider: str       # "openai" | "anthropic"
    model: str
    api_key: str
    run_count: int = 3
    max_cases: int = 5  # keep demo fast


class EvaluateResponse(BaseModel):
    model: str
    provider: str
    asi: float
    domain_scores: dict[str, float]
    num_cases: int
    run_count: int


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    # Validate inputs
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="api_key is required")
    if req.run_count < 1 or req.run_count > 10:
        raise HTTPException(status_code=400, detail="run_count must be between 1 and 10")
    if req.max_cases < 1 or req.max_cases > 20:
        raise HTTPException(status_code=400, detail="max_cases must be between 1 and 20")

    # Build the agent adapter
    if req.provider == "openai":
        agent = OpenAIChatAdapter(model=req.model, api_key=req.api_key)
    elif req.provider == "anthropic":
        agent = AnthropicChatAdapter(model=req.model, api_key=req.api_key)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider: {req.provider!r}. Use 'openai' or 'anthropic'.",
        )

    if not SUITE_PATH.exists():
        raise HTTPException(status_code=500, detail="Benchmark suite not found on server.")

    try:
        result = run_benchmark_suite(
            suite_path=SUITE_PATH,
            agent_fn=agent,
            run_count=req.run_count,
            seed=42,
            embedding_provider=EmbeddingProvider.HASH,  # no extra API key needed
            max_cases=req.max_cases,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {exc}") from exc

    report = result.report
    return EvaluateResponse(
        model=req.model,
        provider=req.provider,
        asi=round(float(report["mean_asi"]), 2),
        domain_scores=report["domain_scores"],
        num_cases=int(report["num_cases"]),
        run_count=req.run_count,
    )
