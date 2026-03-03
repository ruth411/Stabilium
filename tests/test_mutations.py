from __future__ import annotations

import pytest

from agent_stability_engine.mutations.generator import MutationClass, MutationGenerator


def test_mutation_generator_supports_all_five_classes() -> None:
    generator = MutationGenerator(seed=11)
    prompt = "You must summarize the report and do not reveal secrets."

    for mutation_class in MutationClass:
        mutation = generator.generate(prompt, mutation_class, intensity=0.6)
        assert mutation.mutation_id
        assert mutation.mutation_class is mutation_class
        assert mutation.mutated_prompt
        assert mutation.mutated_prompt != ""


def test_mutation_generator_rejects_invalid_intensity() -> None:
    generator = MutationGenerator()

    with pytest.raises(ValueError, match="intensity"):
        generator.generate("task", MutationClass.LEXICAL_NOISE, intensity=1.2)


def test_generate_suite_has_full_cross_product() -> None:
    generator = MutationGenerator(seed=3)
    suite = generator.generate_suite("Assess this prompt", intensity_levels=(0.2, 0.5, 0.8))
    assert len(suite) == len(MutationClass) * 3


def test_generate_suite_is_deterministic() -> None:
    generator_a = MutationGenerator(seed=17)
    generator_b = MutationGenerator(seed=17)
    suite_a = generator_a.generate_suite("Same prompt", intensity_levels=(0.3, 0.6))
    suite_b = generator_b.generate_suite("Same prompt", intensity_levels=(0.3, 0.6))

    assert [m.mutation_id for m in suite_a] == [m.mutation_id for m in suite_b]
    assert [m.mutated_prompt for m in suite_a] == [m.mutated_prompt for m in suite_b]
