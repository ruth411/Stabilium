from __future__ import annotations

from dataclasses import dataclass

from agent_stability_engine.arbitration.disagreement import CrossModelDisagreement


@dataclass(frozen=True)
class PairwiseDisagreement:
    model_a: str
    model_b: str
    divergence: float


@dataclass(frozen=True)
class ArbitrationResult:
    disagreement_score: float
    consensus_model: str
    consensus_output: str
    outlier_models: list[str]
    pairwise: list[PairwiseDisagreement]


class CrossModelArbitrator:
    """Arbitrates multi-model outputs using pairwise divergence geometry."""

    def __init__(self, dimension: int = 256) -> None:
        self._disagreement = CrossModelDisagreement(dimension=dimension)

    def arbitrate(self, model_outputs: dict[str, str]) -> ArbitrationResult:
        if len(model_outputs) < 2:
            msg = "at least two model outputs are required"
            raise ValueError(msg)

        model_names = list(model_outputs.keys())
        pairwise: list[PairwiseDisagreement] = []

        row_totals = {name: 0.0 for name in model_names}
        pair_count = {name: 0 for name in model_names}

        for idx, left_name in enumerate(model_names):
            for right_name in model_names[idx + 1 :]:
                distance = self._disagreement.score(
                    {
                        left_name: model_outputs[left_name],
                        right_name: model_outputs[right_name],
                    }
                )
                pairwise.append(
                    PairwiseDisagreement(
                        model_a=left_name,
                        model_b=right_name,
                        divergence=distance,
                    )
                )
                row_totals[left_name] += distance
                row_totals[right_name] += distance
                pair_count[left_name] += 1
                pair_count[right_name] += 1

        mean_distance = {
            name: (row_totals[name] / pair_count[name]) if pair_count[name] else 0.0
            for name in model_names
        }

        consensus_model = min(model_names, key=lambda name: mean_distance[name])
        consensus_output = model_outputs[consensus_model]

        global_mean = sum(mean_distance.values()) / len(mean_distance)
        outlier_threshold = min(global_mean + 0.15, 1.0)
        outlier_models = [
            name
            for name, distance in mean_distance.items()
            if distance > outlier_threshold and name != consensus_model
        ]

        disagreement_score = self._disagreement.score(model_outputs)

        return ArbitrationResult(
            disagreement_score=disagreement_score,
            consensus_model=consensus_model,
            consensus_output=consensus_output,
            outlier_models=outlier_models,
            pairwise=pairwise,
        )
