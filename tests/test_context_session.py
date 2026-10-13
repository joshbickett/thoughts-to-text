from concurrent.futures import Future
from threading import Event

from lipflow.context import Context
from lipflow.context_session import ContextCollector, ContextSnapshot, MAX_CONTEXT_CHARS


def test_dictations_keep_their_own_context_after_focus_changes():
    pending = Future()
    first = ContextSnapshot(Context(title="First conversation", near_text="First reply"), pending)
    second = ContextSnapshot(Context(title="Second conversation", near_text="Second reply"))
    pending.set_result("Alice: Please send the proposal.")
    assert "Alice" in first.text()
    assert "Second" not in first.text()
    assert "Alice" not in second.text()


def test_unavailable_or_slow_ocr_preserves_text_field_context():
    pending = Future()
    snapshot = ContextSnapshot(Context(title="Draft", near_text="My current sentence"), pending)
    assert "My current sentence" in snapshot.text(wait=0)
    pending.set_exception(RuntimeError("Window closed"))
    assert "My current sentence" in snapshot.text()


def test_context_is_bounded_and_can_disable_screenshot_results():
    future = Future()
    future.set_result("private OCR " * 1000)
    snapshot = ContextSnapshot(Context(title="Draft", near_text="hello\x00there"), future)
    assert len(snapshot.text()) <= MAX_CONTEXT_CHARS
    assert "\x00" not in snapshot.text()
    assert "private OCR" not in snapshot.text(allow_screen=False)


def test_secure_field_never_reaches_ocr_or_prompt():
    ctx = Context()
    ctx.secure_field = True
    ctx.near_text = "sensitive text"
    calls = []
    collector = ContextCollector(lambda *args: calls.append(args))
    try:
        snapshot = collector.start(ctx, use_screen=True)
        assert snapshot.text() == ""
        assert snapshot.screen is None
        assert not calls
    finally:
        collector.close()


def test_cancel_and_restart_cannot_queue_a_screenshot_backlog():
    release = Event()
    started = Event()

    def capture(ctx, max_chars):
        started.set()
        release.wait(1)
        return ctx.title

    collector = ContextCollector(capture)
    try:
        first = collector.start(Context(title="First"), use_screen=True)
        assert started.wait(1)
        second = collector.start(Context(title="Second"), use_screen=True)
        assert second.screen is None
        release.set()
        assert first.screen.result(timeout=1) == "First"
        assert "First" not in second.text()
    finally:
        release.set()
        collector.close()
