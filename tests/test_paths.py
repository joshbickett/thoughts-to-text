"""Keep this fork's personal data separate, while honoring explicit overrides."""
import os
from pathlib import Path
import runpy
import sys

import pytest


PATHS_MODULE = Path(__file__).resolve().parents[1] / "lipflow" / "paths.py"


def _paths(monkeypatch, platform="darwin"):
    monkeypatch.setattr(sys, "platform", platform)
    return runpy.run_path(str(PATHS_MODULE))


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_default_data_is_separate_from_lipflow(monkeypatch, tmp_path, platform):
    monkeypatch.delenv("THOUGHTS_TO_TEXT_HOME", raising=False)
    monkeypatch.delenv("LIPFLOW_HOME", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    paths = _paths(monkeypatch, platform)
    expected = (os.path.expanduser("~/Library/Application Support/ThoughtsToText")
                if platform == "darwin" else str(tmp_path / "ThoughtsToText"))
    assert paths["HOME"] == expected
    assert paths["PERSONAL_MODELS"] == os.path.join(expected, "models")


def test_new_override_wins_over_legacy(monkeypatch, tmp_path):
    chosen = str(tmp_path / "chosen")
    monkeypatch.setenv("THOUGHTS_TO_TEXT_HOME", chosen)
    monkeypatch.setenv("LIPFLOW_HOME", str(tmp_path / "legacy"))
    assert _paths(monkeypatch)["HOME"] == chosen


@pytest.mark.parametrize("new_override", [None, ""])
def test_legacy_override_remains_supported(monkeypatch, tmp_path, new_override):
    if new_override is None:
        monkeypatch.delenv("THOUGHTS_TO_TEXT_HOME", raising=False)
    else:
        monkeypatch.setenv("THOUGHTS_TO_TEXT_HOME", new_override)
    legacy = str(tmp_path / "legacy")
    monkeypatch.setenv("LIPFLOW_HOME", legacy)
    assert _paths(monkeypatch)["HOME"] == legacy
