"""Guard: fixture catalog stays aligned with generated joblib files."""

from __future__ import annotations

from tests.conftest import MODELS_DIR
from tests.model_fixtures import MODEL_FIXTURES, OBSOLETE_FIXTURES


def test_model_fixtures_catalog_matches_disk():
    on_disk = sorted(p.stem for p in MODELS_DIR.glob("*.joblib"))
    assert on_disk == sorted(MODEL_FIXTURES), (
        "Disk fixtures drift from tests/model_fixtures.MODEL_FIXTURES - "
        "run: python scripts/generate_models.py"
    )


def test_obsolete_fixture_names_not_on_disk():
    for name in OBSOLETE_FIXTURES:
        path = MODELS_DIR / f"{name}.joblib"
        assert not path.exists(), f"stale fixture still present: {path}"
