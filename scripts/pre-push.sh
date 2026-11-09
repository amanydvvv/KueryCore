#!/usr/bin/env bash
set -euo pipefail

echo "[pre-push] Step 1/3: Checking Python syntax and undefined names..."
python -m compileall -q backend/app backend/tests
if command -v ruff >/dev/null 2>&1; then
    ruff check backend/app --select F821,E999
fi

echo "[pre-push] Step 2/3: Verifying single Alembic migration head..."
HEAD_COUNT=$(cd backend && python -m alembic heads | grep -v "^$" | wc -l | tr -d ' ')
if [ "$HEAD_COUNT" -ne 1 ]; then
    echo "[pre-push] ERROR: Expected 1 Alembic head, found $HEAD_COUNT."
    cd backend && python -m alembic heads
    exit 1
fi

echo "[pre-push] Step 3/3: Running fast smoke tests (keep-alive workflow)..."
(cd backend && python -m pytest tests/test_keep_alive_workflow.py -q --tb=short)

echo "[pre-push] All guardrails passed."
