"""
Phase 2 — DeepEval LLM-evaluation runner.

What this script does
---------------------
1.  Spins up a TestClient against the FastAPI app (no live server needed —
    no network, no Render, no port binding).
2.  Creates a throwaway eval user via POST /api/auth/register.
3.  Uploads backend/tests/eval/eval_corpus.md via POST /api/documents/upload
    and polls the document status until processing is complete (or times out).
4.  Loads the 33 positive entries from golden_set.json (negatives have no
    expected answer, so DeepEval metrics are skipped for them).
5.  Calls POST /api/chat for each question using the TestClient (full app
    stack: routing → retrieval → generation → citations).
6.  Calls app.services.retrieval.retrieve_context() directly for each
    question to get the FULL chunk text (the /api/chat citation preview is
    capped at 300 chars).
7.  Builds DeepEval LLMTestCase objects and evaluates:
        Faithfulness           — answer only uses info from contexts
        AnswerRelevancy        — answer addresses the question
        ContextualPrecision    — retrieved chunks ranked: relevant first
        ContextualRecall       — expected facts present in contexts
8.  Writes results/eval_<timestamp>.json (timestamped, never overwritten).
9.  On the FIRST run, copies the file to eval/baseline_deepeval.json.
    On subsequent runs, prints a diff table vs. the baseline.

Usage
-----
    # from backend/ directory
    python -m eval.run_deepeval              # full run (all 33 positives)
    python -m eval.run_deepeval --sample 5  # quick sanity check (5 random)
    python -m eval.run_deepeval --retrieval-only  # skip LLM judge, just retrieve

Environment variables required (see backend/.env.example):
    GROQ_API_KEY
    GEMINI_API_KEY (or OPENAI_API_KEY — whatever generate_answer uses)
    DATABASE_URL   (async asyncpg URL — TestClient uses a real DB)
    DATABASE_URL_SYNC

Outputs
-------
    backend/results/eval_<timestamp>.json
    backend/eval/baseline_deepeval.json  (first run only — never overwritten)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — make `backend/` importable when run as `python -m eval.run_deepeval`
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env", override=False)

from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.database import AsyncSessionLocal
from app.services import retrieval as retrieval_svc
from app.services.evaluation import load_golden_set, GoldenEntry

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)

GOLDEN_FILE = BACKEND_DIR / "tests" / "eval" / "golden_set.json"
RESULTS_DIR = BACKEND_DIR / "results"
DEEPEVAL_BASELINE = BACKEND_DIR / "eval" / "baseline_deepeval.json"
EVAL_USER_EMAIL = f"eval_runner_{uuid.uuid4().hex[:8]}@kuerycore.internal"
EVAL_USER_PASS = "EvalRunner!2026"
UPLOAD_POLL_TIMEOUT_S = 120
UPLOAD_POLL_INTERVAL_S = 3


# ---------------------------------------------------------------------------
# Auth helpers (TestClient — synchronous)
# ---------------------------------------------------------------------------

def _register_and_login(client: TestClient) -> str:
    """Register a throwaway eval user and return a JWT access token."""
    r = client.post(
        "/api/auth/register",
        json={"email": EVAL_USER_EMAIL, "password": EVAL_USER_PASS},
    )
    if r.status_code not in (200, 201, 400):  # 400 = already exists on retry
        r.raise_for_status()

    r2 = client.post(
        "/api/auth/token",
        data={"username": EVAL_USER_EMAIL, "password": EVAL_USER_PASS},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    r2.raise_for_status()
    return r2.json()["access_token"]


# ---------------------------------------------------------------------------
# Document upload + polling (TestClient — synchronous)
# ---------------------------------------------------------------------------

def _upload_corpus(client: TestClient, token: str) -> str:
    """Upload eval_corpus.md and return the document_id."""
    corpus_path = BACKEND_DIR / "tests" / "eval" / "eval_corpus.md"
    logger.info("Uploading corpus: %s", corpus_path)
    with open(corpus_path, "rb") as fh:
        r = client.post(
            "/api/documents/upload",
            files={"file": ("eval_corpus.md", fh, "text/markdown")},
            headers={"Authorization": f"Bearer {token}"},
        )
    r.raise_for_status()
    doc_id = r.json()["document_id"]
    logger.info("Uploaded → document_id=%s  status=%s", doc_id, r.json().get("status"))
    return doc_id


def _poll_document_ready(client: TestClient, token: str, doc_id: str) -> None:
    """Block until the document status is 'completed' or timeout."""
    deadline = time.time() + UPLOAD_POLL_TIMEOUT_S
    while time.time() < deadline:
        r = client.get(
            f"/api/documents/{doc_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code == 200:
            status = r.json().get("status", "unknown")
            logger.info("  Document status: %s", status)
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError(f"Document processing failed for {doc_id}")
        time.sleep(UPLOAD_POLL_INTERVAL_S)
    raise TimeoutError(
        f"Document {doc_id} did not reach 'completed' in {UPLOAD_POLL_TIMEOUT_S}s"
    )


# ---------------------------------------------------------------------------
# Chat query (TestClient)
# ---------------------------------------------------------------------------

def _query_chat(
    client: TestClient, token: str, question: str, doc_id: str
) -> Dict[str, Any]:
    """Call POST /api/chat and return the full JSON response."""
    r = client.post(
        "/api/chat",
        json={
            "question": question,
            "document_id": doc_id,
            "top_k": 5,
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Full-text retrieval (bypass 300-char preview)
# ---------------------------------------------------------------------------

async def _retrieve_full_chunks(
    question: str, user_id: str, doc_id: str, top_k: int = 5
) -> List[str]:
    """
    Call retrieval_svc.retrieve_context() directly to get full chunk text.
    The /api/chat citation preview is truncated to 300 chars; we need the
    full text for ContextualPrecision and ContextualRecall metrics.
    """
    async with AsyncSessionLocal() as session:
        items = await retrieval_svc.retrieve_context(
            query=question,
            session=session,
            user_id=user_id,
            document_id=doc_id,
            top_k=top_k,
        )
    # items: List[Tuple[chunk_obj, score, filename]]
    return [item[0].content for item in items]


# ---------------------------------------------------------------------------
# DeepEval evaluation
# ---------------------------------------------------------------------------

def _build_test_cases(
    entries: List[GoldenEntry],
    client: TestClient,
    token: str,
    doc_id: str,
    user_id: str,
    retrieval_only: bool,
) -> List[Dict[str, Any]]:
    """
    For each positive entry:
      - call /api/chat → actual_output
      - call retrieve_context directly → retrieval_context (full text)
      - return a dict suitable for DeepEval LLMTestCase construction
    """
    results = []
    for i, entry in enumerate(entries, 1):
        logger.info("[%d/%d] %s — %s", i, len(entries), entry.id, entry.question[:60])
        chat_resp = _query_chat(client, token, entry.question, doc_id)
        actual_output = chat_resp.get("answer", "")
        # Full chunk text via direct retrieval service
        full_chunks = asyncio.get_event_loop().run_until_complete(
            _retrieve_full_chunks(entry.question, user_id, doc_id)
        )
        results.append(
            {
                "entry_id": entry.id,
                "question": entry.question,
                "actual_output": actual_output,
                "expected_output": " ".join(entry.answer_facts),
                "retrieval_context": full_chunks,
                "intent": entry.intent,
            }
        )
    return results


def _run_deepeval_metrics(test_cases_data: List[Dict]) -> List[Dict[str, Any]]:
    """Evaluate each test case with DeepEval metrics. Returns scored rows."""
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
    )
    from deepeval.test_case import LLMTestCase
    from eval.judge import GroqJudge

    judge = GroqJudge()

    metrics = [
        FaithfulnessMetric(model=judge, threshold=0.7),
        AnswerRelevancyMetric(model=judge, threshold=0.7),
        ContextualPrecisionMetric(model=judge, threshold=0.7),
        ContextualRecallMetric(model=judge, threshold=0.7),
    ]

    scored = []
    for row in test_cases_data:
        tc = LLMTestCase(
            input=row["question"],
            actual_output=row["actual_output"],
            expected_output=row["expected_output"],
            retrieval_context=row["retrieval_context"],
        )
        # Evaluate one at a time so we capture per-row scores
        evaluate([tc], metrics, ignore_errors=True)
        metric_scores = {}
        for m in metrics:
            name = type(m).__name__
            try:
                metric_scores[name] = {
                    "score": tc.get_metric_scores(m),
                    "passed": tc.get_metric_results(m),
                    "reason": getattr(m, "reason", None),
                }
            except Exception:
                metric_scores[name] = {"score": None, "passed": None, "reason": "error"}

        scored.append(
            {
                "entry_id": row["entry_id"],
                "question": row["question"],
                "actual_output": row["actual_output"],
                "expected_output": row["expected_output"],
                "metrics": metric_scores,
            }
        )
    return scored


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _compute_summary(scored: List[Dict]) -> Dict[str, Any]:
    metric_names = [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
    ]
    summary: Dict[str, Any] = {}
    for mname in metric_names:
        scores = [
            r["metrics"][mname]["score"]
            for r in scored
            if r["metrics"].get(mname, {}).get("score") is not None
        ]
        if scores:
            summary[mname] = {
                "mean": round(sum(scores) / len(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "n": len(scores),
            }
    return summary


def _write_results(scored: List[Dict], run_meta: Dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    payload = {
        "run_metadata": run_meta,
        "summary": _compute_summary(scored),
        "results": scored,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written → %s", out_path)
    return out_path


def _maybe_write_baseline(out_path: Path) -> None:
    DEEPEVAL_BASELINE.parent.mkdir(exist_ok=True)
    if not DEEPEVAL_BASELINE.exists():
        import shutil
        shutil.copy2(out_path, DEEPEVAL_BASELINE)
        logger.info("First run — baseline saved → %s", DEEPEVAL_BASELINE)


def _print_diff(current_path: Path) -> None:
    if not DEEPEVAL_BASELINE.exists():
        return
    base = json.loads(DEEPEVAL_BASELINE.read_text(encoding="utf-8"))
    curr = json.loads(current_path.read_text(encoding="utf-8"))
    base_sum = base.get("summary", {})
    curr_sum = curr.get("summary", {})
    print("\n── Regression diff vs baseline ─────────────────────")
    print(f"{'Metric':<32} {'Baseline':>10} {'Current':>10} {'Delta':>8}")
    print("─" * 64)
    for mname in [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
    ]:
        b = base_sum.get(mname, {}).get("mean")
        c = curr_sum.get(mname, {}).get("mean")
        if b is None or c is None:
            continue
        delta = c - b
        flag = "⚠ REGRESSION" if delta < -0.05 else ("✓" if delta >= 0 else "↓")
        print(f"{mname:<32} {b:>10.4f} {c:>10.4f} {delta:>+8.4f}  {flag}")
    print("─" * 64)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 DeepEval runner")
    p.add_argument("--sample", type=int, default=0, help="Evaluate N random positives (0 = all)")
    p.add_argument("--retrieval-only", action="store_true", help="Skip LLM judge; just retrieve + print chunks")
    p.add_argument("--seed", type=int, default=42, help="Random seed for --sample")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load golden set — positives only for DeepEval metrics
    all_entries = load_golden_set(GOLDEN_FILE)
    positives = [e for e in all_entries if e.expect_verdict == "pass"]
    if args.sample and args.sample < len(positives):
        random.seed(args.seed)
        positives = random.sample(positives, args.sample)
        logger.info("Sampled %d / %d positive entries (seed=%d)", len(positives), len(all_entries), args.seed)
    else:
        logger.info("Running all %d positive entries", len(positives))

    # Build a TestClient (no live server needed)
    with TestClient(app, raise_server_exceptions=True) as client:
        token = _register_and_login(client)
        doc_id = _upload_corpus(client, token)
        _poll_document_ready(client, token, doc_id)

        # Derive user_id from /api/auth/me (or extract from JWT)
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        me.raise_for_status()
        user_id = me.json().get("id") or me.json().get("user_id")

        test_cases_data = _build_test_cases(
            positives, client, token, doc_id, user_id, args.retrieval_only
        )

    if args.retrieval_only:
        for row in test_cases_data:
            print(f"\n── {row['entry_id']} ─────")
            print(f"Q: {row['question']}")
            for j, ctx in enumerate(row["retrieval_context"], 1):
                print(f"  [{j}] {ctx[:200]}…")
        return

    scored = _run_deepeval_metrics(test_cases_data)
    run_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": os.environ.get("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b"),
        "n_entries": len(scored),
        "sample_seed": args.seed if args.sample else None,
    }
    out_path = _write_results(scored, run_meta)
    _maybe_write_baseline(out_path)
    _print_diff(out_path)
    summary = _compute_summary(scored)
    print("\n── Summary ─────────────────────────────────────────")
    for mname, s in summary.items():
        print(f"  {mname:<32}  mean={s['mean']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}  n={s['n']}")
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()

