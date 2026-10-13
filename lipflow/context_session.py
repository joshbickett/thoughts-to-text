"""Keep each dictation's screen context attached to that recording, in memory only."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from .context import Context

MAX_CONTEXT_CHARS = 2500


def _text(value: str, limit: int) -> str:
    return "".join(c for c in value if c in "\n\t" or c.isprintable())[:limit]


@dataclass
class ContextSnapshot:
    focused: Context
    screen: Future | None = None

    def text(self, *, allow_screen: bool = True, wait: float = 0.02) -> str:
        """OCR normally finishes while the user mouths words; never hold up transcription."""
        if getattr(self.focused, "secure_field", False):
            return ""
        parts = []
        if self.focused.title:
            parts.append("Window title: " + _text(self.focused.title, 160))
        if self.focused.near_text:
            parts.append("Current text field:\n" + _text(self.focused.near_text, 600))
        if allow_screen and self.screen is not None:
            try:
                visible = self.screen.result(timeout=max(0, min(wait, 0.1)))
            except Exception:
                # Permission denied, timeout, a closed window, and OCR failures are optional context.
                visible = ""
            if isinstance(visible, str) and visible.strip():
                parts.append("Visible conversation or document:\n" + _text(visible.strip(), MAX_CONTEXT_CHARS))
        return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


class ContextCollector:
    def __init__(self, capture=None):
        if capture is None:
            from .screen_context import capture_screen_text
            capture = capture_screen_text
        self.capture = capture
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thoughts-screen")
        self.pending = None

    def start(self, focused: Context, *, use_screen: bool) -> ContextSnapshot:
        snapshot = ContextSnapshot(focused)
        # Do not build an OCR backlog if the user cancels and starts another dictation quickly.
        if use_screen and not getattr(focused, "secure_field", False):
            if self.pending is None or self.pending.done():
                self.pending = self.executor.submit(self.capture, focused, MAX_CONTEXT_CHARS)
                snapshot.screen = self.pending
        return snapshot

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
