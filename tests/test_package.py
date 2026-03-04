from __future__ import annotations

import agent_stability_engine


def test_package_exports_version() -> None:
    assert isinstance(agent_stability_engine.__version__, str)
    assert agent_stability_engine.__version__ != ""
