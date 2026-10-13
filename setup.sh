#!/usr/bin/env bash
# Install dependencies and download the models (~1.2 GB). Add --samples for the test clips.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This setup needs an Apple Silicon Mac and a native arm64 terminal."
  echo "On Linux, use ./setup-linux.sh instead."
  exit 1
fi
macos_version=$(sw_vers -productVersion)
if (( ${macos_version%%.*} < 14 )); then
  echo "Thoughts to Text needs macOS 14 or later for its current dependencies."
  exit 1
fi
if ! xcrun --find clang >/dev/null 2>&1; then
  echo "Install Apple's Command Line Tools with: xcode-select --install"
  echo "Finish that installation, then run ./setup.sh again."
  exit 1
fi

if ! command -v uv >/dev/null; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# Models, and with --samples the public-domain clips used by tests/test_pipeline.py
if [[ "${1:-}" == "--samples" ]]; then
  ./scripts/download-models.sh --samples
else
  ./scripts/download-models.sh
fi

# The small on-device cleanup model (~350 MB), so the first launch doesn't stall on it
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3-0.6B-4bit')" >/dev/null 2>&1 && echo "✓ cleanup model"

# The app bundle: its own permissions, Spotlight/Launchpad, Login Items
if [[ "${1:-}" != "--no-app" ]]; then
  uv run python scripts/make_app.py --dest /Applications
fi

echo
echo "Done. Open Thoughts to Text from Spotlight (or: open /Applications/Thoughts-to-Text.app)."
echo "The first launch walks you through permissions, your Wispr Flow words, and ~24 practice sentences."
