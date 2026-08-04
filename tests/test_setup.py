"""Smoke test Step 1 — pastikan config & device helper dapat di-import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_config_loads():
    from uvip_ai.config import settings

    assert settings.latency_target_ms == 700
    assert settings.uvip_device in ("auto", "cuda", "cpu")


def test_device_resolves():
    from uvip_ai.utils.device import get_device

    assert get_device() in ("cuda", "cpu")


def test_version():
    import uvip_ai

    assert uvip_ai.__version__
