from datetime import datetime

import pytest

from hreval.domain.store import Store

FROZEN_NOW = datetime.fromisoformat("2026-03-10T09:00:00+01:00")


@pytest.fixture
def now():
    return FROZEN_NOW


@pytest.fixture
def store(now):
    return Store(frozen_now=now)
