"""Screen context stays local, scoped to one window, optional, and bounded."""
import contextlib
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from lipflow.context import Context
from lipflow import context, screen_context as screen


def window(number=7, pid=12, title="Document", bounds=(0, 0, 800, 600), **extra):
    return {
        "kCGWindowNumber": number,
        "kCGWindowOwnerPID": pid,
        "kCGWindowLayer": 0,
        "kCGWindowName": title,
        "kCGWindowBounds": dict(zip(("X", "Y", "Width", "Height"), bounds)),
        **extra,
    }


def test_window_selection_excludes_other_apps_and_non_document_layers():
    ctx = Context(app_pid=12, title="Document")
    target = window()
    candidates = [window(pid=99), window(number=8, kCGWindowLayer=3), target]
    assert screen._select_window_info(ctx, candidates) is target


def test_window_selection_uses_focused_ax_geometry_even_with_duplicate_titles():
    ctx = Context(app_pid=12, title="Document", window_bounds=(10, 20, 800, 600))
    target = window(number=8, bounds=(10, 20, 800, 600))
    assert screen._select_window_info(ctx, [window(), target]) is target
    assert screen._select_window_info(ctx, [window()]) is None


def test_window_selection_never_replaces_a_pinned_window():
    ctx = Context(app_pid=12, window_id=7)
    assert screen._select_window_info(ctx, [window(number=8), window(pid=99)]) is None


def test_missing_titles_uses_front_to_back_metadata_order():
    ctx = Context(app_pid=12, title="Document")
    target = window(title="")
    assert screen._select_window_info(ctx, [target, window(number=8, title="")]) is target
    assert screen._select_window_info(ctx, [window(title="Other document")]) is None


@pytest.mark.parametrize("bounds", [(0, 0, 0, 100), (0, 0, float("nan"), 100), (1, 2), None])
def test_invalid_window_bounds_rejected(bounds):
    assert screen._bounds(bounds) is None


def test_image_size_preserves_aspect_ratio_and_caps_pixels():
    assert screen._image_size(800, 600) == (1600, 1200)
    assert screen._image_size(4000, 2000) == (2048, 1024)
    assert screen._image_size(float("inf"), 100) is None


def observation(text, x, y, height=0.03):
    return NS(
        topCandidates_=lambda n: [NS(string=lambda: text)],
        boundingBox=lambda: NS(origin=NS(x=x, y=y), size=NS(height=height)),
    )


def test_ocr_reads_rows_top_to_bottom_and_removes_controls():
    observations = [
        observation("Bottom\x00 \u202eRow", 0.1, 0.1),
        observation("Right", 0.7, 0.802),
        observation("Left", 0.1, 0.8),
    ]
    assert screen._reading_order(observations, 100) == "Left Right\nBottom Row"
    assert screen._reading_order(observations, 4) == "Left"


def native_environment(monkeypatch):
    monkeypatch.setattr(screen.sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "objc", NS(autorelease_pool=contextlib.nullcontext))
    monkeypatch.setattr(screen, "_CAPTURE_LOCK", threading.Lock())
    monkeypatch.setattr(screen, "screen_capture_allowed", lambda: True)


def test_capture_returns_only_ocr_text_with_bounded_size(monkeypatch):
    native_environment(monkeypatch)
    image = object()
    monkeypatch.setattr(screen, "_capture_window_image", lambda ctx, deadline: image)
    seen = []

    def recognize(got_image, max_chars, requests):
        seen.append((got_image, max_chars))
        return "Project Phoenix"

    monkeypatch.setattr(screen, "_recognize_text", recognize)
    assert screen.capture_screen_text(Context(app_pid=12, window_id=7), 99999) == "Project Phoenix"
    assert seen == [(image, 6000)]


@pytest.mark.parametrize("ctx", [Context(), Context(app_pid=12), Context(app_pid=12, window_id=7, secure_field=True)])
def test_unsafe_or_missing_target_skips_all_native_work(monkeypatch, ctx):
    native_environment(monkeypatch)
    monkeypatch.setattr(screen, "screen_capture_allowed", lambda: pytest.fail("Must skip permission/native work"))
    assert screen.capture_screen_text(ctx) == ""


def test_denied_permission_never_prompts_or_captures(monkeypatch):
    native_environment(monkeypatch)
    monkeypatch.setattr(screen, "screen_capture_allowed", lambda: False)
    monkeypatch.setattr(screen, "request_screen_capture_permission", lambda: pytest.fail("Automatic prompt"))
    monkeypatch.setattr(screen, "_capture_window_image", lambda *a: pytest.fail("Capture while denied"))
    assert screen.capture_screen_text(Context(app_pid=12, window_id=7)) == ""


def test_missing_native_framework_is_optional(monkeypatch):
    native_environment(monkeypatch)
    monkeypatch.setitem(sys.modules, "ScreenCaptureKit", None)
    assert screen.capture_screen_text(Context(app_pid=12, window_id=7)) == ""


def test_ocr_timeout_cancels_and_does_not_accumulate_workers(monkeypatch):
    native_environment(monkeypatch)
    monkeypatch.setattr(screen, "_TIMEOUT", 0.05)
    monkeypatch.setattr(screen, "_capture_window_image", lambda *a: object())
    release, cancelled, exited = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def stalled_ocr(image, max_chars, requests):
        calls.append(True)
        requests.append(NS(cancel=cancelled.set))
        release.wait(1)
        exited.set()
        return "late result"

    monkeypatch.setattr(screen, "_recognize_text", stalled_ocr)
    ctx = Context(app_pid=12, window_id=7)
    try:
        assert screen.capture_screen_text(ctx) == ""
        assert cancelled.is_set()
        assert screen.capture_screen_text(ctx) == ""
        assert len(calls) == 1
    finally:
        release.set()
        assert exited.wait(1)


def test_screencapturekit_filters_exact_window_and_ignores_others(monkeypatch):
    target = NS(
        windowID=lambda: 7, owningApplication=lambda: NS(processID=lambda: 12),
        isOnScreen=lambda: True,
        frame=lambda: NS(size=NS(width=800, height=600)),
    )
    wrong = NS(windowID=lambda: 8)
    captured = []
    options = {}
    config = NS(**{name: lambda v, name=name: options.update({name: v}) for name in (
        "setWidth_", "setHeight_", "setShowsCursor_", "setIgnoreShadowsSingleWindow_", "setIncludeChildWindows_"
    )})
    sc = NS(
        SCShareableContent=NS(
            getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_=
                lambda exclude, onscreen, cb: cb(NS(windows=lambda: [wrong, target]), None)
        ),
        SCContentFilter=NS(alloc=lambda: NS(initWithDesktopIndependentWindow_=lambda w: w)),
        SCStreamConfiguration=NS(alloc=lambda: NS(init=lambda: config)),
        SCScreenshotManager=NS(captureImageWithFilter_configuration_completionHandler_=
            lambda filter, config, cb: (captured.append(filter), cb("image", None))),
    )
    monkeypatch.setitem(sys.modules, "ScreenCaptureKit", sc)
    deadline = screen.time.monotonic() + 1
    assert screen._capture_window_image(Context(app_pid=12, window_id=7), deadline) == "image"
    assert captured == [target]
    assert options["setIncludeChildWindows_"] is False
    assert options["setShowsCursor_"] is False
    captured.clear()
    assert screen._capture_window_image(Context(app_pid=99, window_id=7), deadline) is None
    assert captured == []


def test_secure_ax_field_value_is_never_read(monkeypatch):
    monkeypatch.setattr(context.sys, "platform", "darwin")
    app = NS(localizedName=lambda: "Browser", processIdentifier=lambda: 12)
    monkeypatch.setitem(sys.modules, "AppKit", NS(NSWorkspace=NS(sharedWorkspace=lambda: NS(frontmostApplication=lambda: app))))
    monkeypatch.setitem(sys.modules, "ApplicationServices", NS(AXUIElementCreateApplication=lambda pid: "app"))
    seen = []

    def ax(el, attr):
        seen.append(attr)
        return {"AXFocusedUIElement": "password", "AXSubrole": "AXSecureTextField"}.get(attr)

    monkeypatch.setattr(context, "_ax", ax)
    ctx = context.capture()
    assert ctx.secure_field is True
    assert ctx.app_pid == 12
    assert ctx.value == ctx.near_text == ""
    assert "AXValue" not in seen


def test_permission_helpers_are_noops_off_macos(monkeypatch):
    monkeypatch.setattr(screen.sys, "platform", "linux")
    assert not screen.screen_capture_allowed()
    assert not screen.request_screen_capture_permission()
