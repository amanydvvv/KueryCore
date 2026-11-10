"""
Phase 2 — DeepEval LLM evaluation test suite & regression gates.

Structure:
  - TestGoldenSetIntegrity: Fast offline integrity assertions on golden_set.json.
  - TestGroqJudgeUnit: Fast mocked unit tests for GroqJudge wrapper (CI-safe).
  - TestDeepEvalResults: Asserts quality thresholds against latest results/eval_*.json (if present).
  - TestRegressionGate: Asserts no metric regression vs eval/baseline_deepeval.json (if present).
  - TestLiveDeepEval: Live LLM judge execution (gated behind KUERYCORE_LIVE_EVAL=1 and valid GROQ_API_KEY).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

GOLDEN_FILE = BACKEND_DIR / "tests" / "eval" / "golden_set.json"
DEEPEVAL_BASELINE = BACKEND_DIR / "eval" / "baseline_deepeval.json"
RESULTS_DIR = BACKEND_DIR / "results"

FAITHFULNESS_THRESHOLD = 0.7
REGRESSION_THRESHOLD = 0.05  # allowed drop vs baseline before gate fires
MIN_POSITIVES_REQUIRED = 10  # minimum entries to constitute a valid run


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_positives():
    """Load positive entries from the golden set."""
    from app.services.evaluation import load_golden_set

    all_entries = load_golden_set(GOLDEN_FILE)
    return [e for e in all_entries if e.expect_verdict == "pass"]


@pytest.fixture(scope="session")
def latest_eval_result():
    """
    Return the most-recent eval_*.json from results/.
    Tests depending on this are skipped if no evaluation run has occurred.
    """
    if not RESULTS_DIR.exists():
        pytest.skip("No results/ directory — run `python -m eval.run_deepeval` first")
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not files:
        pytest.skip("No eval_*.json results — run `python -m eval.run_deepeval` first")
    latest = files[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def baseline_result():
    """Return baseline_deepeval.json, or None if it doesn't exist yet."""
    if not DEEPEVAL_BASELINE.exists():
        return None
    return json.loads(DEEPEVAL_BASELINE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Golden Set Integrity (Fast offline unit tests — runs by default in CI)
# ---------------------------------------------------------------------------

class TestGoldenSetIntegrity:
    """Sanity-check the golden dataset itself — zero LLM calls, zero network."""

    def test_golden_set_loads(self, golden_positives):
        """Golden set must have at least MIN_POSITIVES_REQUIRED positive entries."""
        assert len(golden_positives) >= MIN_POSITIVES_REQUIRED, (
            f"Golden set has only {len(golden_positives)} positive entries; "
            f"need >= {MIN_POSITIVES_REQUIRED}"
        )

    def test_no_empty_markers(self, golden_positives):
        """Every positive entry must have at least one expected_chunk_marker."""
        bad = [e.id for e in golden_positives if not e.expected_chunk_markers]
        assert not bad, f"Entries missing expected_chunk_markers: {bad}"

    def test_no_empty_answer_facts(self, golden_positives):
        """Every positive entry must have at least one answer_fact."""
        bad = [e.id for e in golden_positives if not e.answer_facts]
        assert not bad, f"Entries missing answer_facts: {bad}"

    def test_all_docs_reference_corpus(self, golden_positives):
        """Every 'docs' field must reference eval_corpus.md."""
        bad = [
            e.id
            for e in golden_positives
            if not any("eval_corpus" in d for d in e.docs)
        ]
        assert not bad, f"Entries not referencing eval_corpus.md: {bad}"

    def test_unique_ids(self, golden_positives):
        ids = [e.id for e in golden_positives]
        assert len(ids) == len(set(ids)), "Duplicate IDs in golden set"


# ---------------------------------------------------------------------------
# 2. GroqJudge Unit Tests (Fast mocked tests — runs by default in CI)
# ---------------------------------------------------------------------------

class TestGroqJudgeUnit:
    """Mocked unit tests for GroqJudge wrapper — verifies interface without live APIs."""

    def test_judge_init_defaults(self):
        from eval.judge import GroqJudge

        judge = GroqJudge()
        assert judge.get_model_name() in ("openai/gpt-oss-20b", os.environ.get("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b"))
        assert judge.temperature == 0.0

    def test_judge_custom_model_and_temp(self):
        from eval.judge import GroqJudge

        judge = GroqJudge(model="llama-3.3-70b-versatile", temperature=0.2)
        assert judge.get_model_name() == "llama-3.3-70b-versatile"
        assert judge.temperature == 0.2

    @patch("groq.Groq")
    def test_judge_generate_sync_mocked(self, mock_groq_cls):
        from eval.judge import GroqJudge

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Grounded verdict: Yes"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_cls.return_value = mock_client

        judge = GroqJudge()
        output = judge.generate("Evaluate this question")

        assert output == "Grounded verdict: Yes"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    @patch("groq.AsyncGroq")
    async def test_judge_a_generate_mocked(self, mock_async_groq_cls):
        from eval.judge import GroqJudge

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Async verdict: 1.0"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_async_groq_cls.return_value = mock_client

        judge = GroqJudge()
        output = await judge.a_generate("Evaluate this question asynchronously")

        assert output == "Async verdict: 1.0"
        mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# 3. DeepEval Saved Results Validation (Skipped gracefully if no eval run yet)
# ---------------------------------------------------------------------------

class TestDeepEvalResults:
    """Validate the most-recent eval run against quality thresholds."""

    def test_faithfulness_mean_above_threshold(self, latest_eval_result):
        """Mean faithfulness must be >= 0.7."""
        summary = latest_eval_result.get("summary", {})
        f = summary.get("FaithfulnessMetric", {})
        if not f:
            pytest.skip("FaithfulnessMetric not in results (retrieval-only run?)")
        mean = f["mean"]
        assert mean >= FAITHFULNESS_THRESHOLD, (
            f"Mean faithfulness {mean:.4f} < threshold {FAITHFULNESS_THRESHOLD}."
        )

    def test_answer_relevancy_mean_above_threshold(self, latest_eval_result):
        """Mean answer relevancy must be >= 0.7."""
        summary = latest_eval_result.get("summary", {})
        ar = summary.get("AnswerRelevancyMetric", {})
        if not ar:
            pytest.skip("AnswerRelevancyMetric not in results")
        assert ar["mean"] >= FAITHFULNESS_THRESHOLD, (
            f"Mean answer relevancy {ar['mean']:.4f} < {FAITHFULNESS_THRESHOLD}"
        )

    def test_sufficient_entries_evaluated(self, latest_eval_result):
        """The eval run must have evaluated at least MIN_POSITIVES_REQUIRED entries."""
        n = latest_eval_result.get("run_metadata", {}).get("n_entries", 0)
        assert n >= MIN_POSITIVES_REQUIRED, (
            f"Only {n} entries evaluated — need >= {MIN_POSITIVES_REQUIRED}"
        )

    def test_hallucination_rate_reported(self, latest_eval_result):
        """Hallucination rate (faithfulness < 0.7) must be below 15%."""
        results = latest_eval_result.get("results", [])
        if not results:
            pytest.skip("No result rows")
        f_scores = [
            r["metrics"].get("FaithfulnessMetric", {}).get("score")
            for r in results
        ]
        f_scores = [s for s in f_scores if s is not None]
        if not f_scores:
            pytest.skip("No faithfulness scores available")
        hallucinated = sum(1 for s in f_scores if s < FAITHFULNESS_THRESHOLD)
        rate = hallucinated / len(f_scores)
        assert rate < 0.15, (
            f"Hallucination rate {rate:.1%} ({hallucinated}/{len(f_scores)}) exceeds 15%"
        )


# ---------------------------------------------------------------------------
# 4. Regression Gate (Active only after baseline_deepeval.json is established)
# ---------------------------------------------------------------------------

class TestRegressionGate:
    """Fail if any metric mean drops > REGRESSION_THRESHOLD vs baseline."""

    METRIC_NAMES = [
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
    ]

    def _check_metric(self, metric_name, latest_eval_result, baseline_result):
        if baseline_result is None:
            pytest.skip("No baseline yet — run `python -m eval.run_deepeval` first to establish it")
        base_sum = baseline_result.get("summary", {})
        curr_sum = latest_eval_result.get("summary", {})
        b = base_sum.get(metric_name, {}).get("mean")
        c = curr_sum.get(metric_name, {}).get("mean")
        if b is None or c is None:
            pytest.skip(f"{metric_name} not present in both baseline and current results")
        delta = c - b
        assert delta >= -REGRESSION_THRESHOLD, (
            f"{metric_name} regression: baseline={b:.4f}, current={c:.4f}, "
            f"drop={abs(delta):.4f} > threshold={REGRESSION_THRESHOLD}"
        )

    def test_faithfulness_no_regression(self, latest_eval_result, baseline_result):
        self._check_metric("FaithfulnessMetric", latest_eval_result, baseline_result)

    def test_answer_relevancy_no_regression(self, latest_eval_result, baseline_result):
        self._check_metric("AnswerRelevancyMetric", latest_eval_result, baseline_result)

    def test_contextual_precision_no_regression(self, latest_eval_result, baseline_result):
        self._check_metric("ContextualPrecisionMetric", latest_eval_result, baseline_result)

    def test_contextual_recall_no_regression(self, latest_eval_result, baseline_result):
        self._check_metric("ContextualRecallMetric", latest_eval_result, baseline_result)


# ---------------------------------------------------------------------------
# 5. Live DeepEval Execution (Gated behind KUERYCORE_LIVE_EVAL=1 + live key)
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
class TestLiveDeepEval:
    """Live LLM evaluations. Gated strictly behind live environment credentials."""

    @pytest.fixture(autouse=True)
    def _require_live_eval_config(self):
        if not os.environ.get("KUERYCORE_LIVE_EVAL"):
            pytest.skip("Set KUERYCORE_LIVE_EVAL=1 to execute live LLM evaluation suite")
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key or groq_key.startswith("test-dummy"):
            pytest.skip("Valid non-dummy GROQ_API_KEY required for live evaluation")
        # Ensure deepeval is installed if running live eval
        pytest.importorskip("deepeval")

    def test_live_judge_connectivity(self):
        """Verify live Groq judge can score a basic prompt."""
        from eval.judge import GroqJudge

        judge = GroqJudge()
        response = judge.generate("Reply with one word: PASS")
        assert len(response.strip()) > 0
