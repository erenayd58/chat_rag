from __future__ import annotations

from types import SimpleNamespace

import pytest

from components.chunker import FrozenV4Chunker, StructuralChunker, create_chunker
from components.chunker.frozen_v4_chunker import (
    FROZEN_AMSC_COMMIT,
    FROZEN_V4_CONFIG_HASH,
    load_frozen_v4_config,
)
from core.exceptions import ConfigurationException


def _settings(**updates):
    values = {
        "chunker_type": "structure_first",
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_v4_factory_selects_only_the_frozen_a4_configuration():
    chunker = create_chunker(_settings(chunker_type="v4"))

    assert isinstance(chunker, FrozenV4Chunker)
    assert chunker.get_config()["commit"] == FROZEN_AMSC_COMMIT
    assert chunker.get_config()["config_hash"] == FROZEN_V4_CONFIG_HASH
    assert chunker.config.ablation.composition == "a4"
    assert chunker.config.tokens.model_dump() == {
        "min_tokens": 160,
        "target_tokens": 700,
        "soft_max_tokens": 900,
        "hard_max_tokens": 1126,
    }


def test_the_default_factory_builds_the_structure_first_chunker():
    assert isinstance(create_chunker(_settings()), StructuralChunker)


def test_an_unknown_chunker_type_is_refused_by_name():
    with pytest.raises(ConfigurationException, match="Unsupported chunker type"):
        create_chunker(_settings(chunker_type="legacy"))


def test_v4_factory_rejects_runtime_tuning_params():
    with pytest.raises(ConfigurationException, match="accepts no runtime tuning"):
        create_chunker(
            _settings(),
            {"type": "v4", "params": {"max_tokens": 256, "threshold": 0.16}},
        )


def test_frozen_config_drift_fails_fast(tmp_path):
    source = load_frozen_v4_config().model_dump(mode="json")
    source["tokens"]["min_tokens"] = 161
    import yaml

    drifted = tmp_path / "v4.yaml"
    drifted.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigurationException, match="config drift"):
        load_frozen_v4_config(drifted)
