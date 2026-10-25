import kb_common
import kb_harvest
import pytest


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """No test may touch real data: KB_HOME, registry, data and state all point into a temp folder."""
    home = tmp_path / "kb_home"
    monkeypatch.setattr(kb_common, "KB_HOME", home)
    monkeypatch.setattr(kb_common, "REGISTRY_PATH", home / "config" / "kb_registry.yaml")
    monkeypatch.setattr(kb_common, "ASSETS_ROOT", home / "kb")
    monkeypatch.setattr(kb_common, "STATE_ROOT", home / "state")
    monkeypatch.setattr(kb_harvest, "ASSETS_ROOT", home / "kb")
    kb_common._clients.clear()
    yield
    kb_common._clients.clear()
