"""Agent Stability Engine core package."""

from importlib.metadata import PackageNotFoundError, version

from agent_stability_engine.engine.evaluator import StabilityEvaluation, StabilityEvaluator
from agent_stability_engine.engine.sampling import MultiRunResult, MultiRunSampler
from agent_stability_engine.engine.self_healing import SelfHealingEngine, SelfHealingResult
from agent_stability_engine.engine.variance import EmbeddingVarianceScorer, VarianceResult
from agent_stability_engine.report.manifest import build_manifest
from agent_stability_engine.report.schema import REPORT_SCHEMA_VERSION, validate_report


def _resolve_version() -> str:
    for distribution_name in ("agent-stability-engine", "agent_stability_engine"):
        try:
            return version(distribution_name)
        except PackageNotFoundError:
            continue
    return "0+unknown"


__version__ = _resolve_version()

__all__ = [
    "EmbeddingVarianceScorer",
    "MultiRunResult",
    "MultiRunSampler",
    "REPORT_SCHEMA_VERSION",
    "SelfHealingEngine",
    "SelfHealingResult",
    "StabilityEvaluation",
    "StabilityEvaluator",
    "VarianceResult",
    "__version__",
    "build_manifest",
    "validate_report",
]
