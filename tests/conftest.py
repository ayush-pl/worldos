import pytest

from worldos.config import WorldConfig, apply_overrides


@pytest.fixture
def small_config() -> WorldConfig:
    return apply_overrides(WorldConfig(), {"population.n_person_records": 4000, "economy.n_firms": 60})
