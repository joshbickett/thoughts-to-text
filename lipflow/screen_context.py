"""One in-memory screenshot of the dictation target, read locally with Apple Vision.

No display-wide capture, image files, network calls, or implicit permission requests.
Native operations are optional and isolated so dictation never waits on them.
"""
from __future__ import annotations

import math
import sys
import threading
import time
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context

_TIMEOUT = 2.0
_MAX_CHARS = 6000
_MAX_IMAGE_SIDE = 2048
# If a native API gets stuck, subsequent dictations skip OCR instead of piling up work.
_CAPTURE_LOCK = threading.Lock()


def screen_capture_allowed() -> bool:
    """Read permission status without prompting. Safe when frameworks are unavailable."""
    if sys.platform != "darwin":
        return False
    try:
        from Quartz import CGPreflightScreenCaptureAccess
        return bool(CGPreflightScreenCaptureAccess())
    except Exception:
        return False


def request_screen_capture_permission() -> bool:
    """Explicit UI action only: may show macOS's Screen Recording permission prompt."""
    if sys.platform != "darwin":
        return False
    try:
        from Quartz import CGRequestScreenCaptureAccess
        return bool(CGRequestScreenCaptureAccess())
    except Exception:
        return False


def _bounds(value) -> tuple[float, float, float, float] | None:
    try:
        if hasattr(value, "keys"):
            rect = tuple(float(value[k]) for k in ("X", "Y", "Width", "Height"))
        else:
            rect = tuple(float(v) for v in value)
        if len(rect) == 4 and all(math.isfinite(v) for v in rect) and min(rect[2:]) > 1:
            return rect
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return None


def _select_window_info(ctx: Context, windows) -> dict | None:
    """CGWindow's front-to-back list, restricted to the already-focused app/window."""
    if ctx.secure_field or ctx.app_pid <= 0:
        return None
    eligible = [w for w in windows if (
        int(w.get("kCGWindowOwnerPID", 0)) == ctx.app_pid
        and int(w.get("kCGWindowLayer", -1)) == 0
        and w.get("kCGWindowIsOnscreen", True)
        and float(w.get("kCGWindowAlpha", 1)) > 0
        and int(w.get("kCGWindowNumber", 0)) > 0
        and _bounds(w.get("kCGWindowBounds"))
    )]
    if ctx.window_id:
        return next((w for w in eligible if int(w["kCGWindowNumber"]) == ctx.window_id), None)
    expected = _bounds(ctx.window_bounds)
    if expected:
        matches = [w for w in eligible if all(
            abs(a - b) <= 3 for a, b in zip(expected, _bounds(w["kCGWindowBounds"]))
        )]
        # No geometry match is safer than guessing a different document window.
        if not matches:
            return None
        eligible = matches
    if ctx.title:
        matches = [w for w in eligible if w.get("kCGWindowName") == ctx.title]
        if matches:
            return matches[0]
        if not expected and any(w.get("kCGWindowName") for w in eligible):
            return None
    # CGWindow ordering identifies the top normal window when AX/title metadata is absent.
    return eligible[0] if eligible else None


def capture_window_identity(ctx: Context) -> None:
    """Add only metadata, called with Accessibility capture on the main thread."""
    if sys.platform != "darwin" or ctx.secure_field or not ctx.app_pid:
        return
    try:
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        selected = _select_window_info(ctx, windows or [])
        if selected:
            ctx.window_id = int(selected["kCGWindowNumber"])
            ctx.window_bounds = _bounds(selected["kCGWindowBounds"])
    except Exception:
        pass


def _image_size(width: float, height: float) -> tuple[int, int] | None:
    if not all(math.isfinite(n) and n > 1 for n in (width, height)):
        return None
    # Retina-sized text where possible; cap memory and Vision work on large displays.
    scale = min(2.0, _MAX_IMAGE_SIDE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _await_callback(start, deadline: float):
    done = threading.Event()
    result = []

    def completed(value, error):
        if error is None and _remaining(deadline) > 0:
            result.append(value)
        done.set()

    if _remaining(deadline) <= 0:
        return None
    start(completed)
    if not done.wait(_remaining(deadline)):
        return None
    return result[0] if result else None


def _capture_window_image(ctx: Context, deadline: float):
    import ScreenCaptureKit as SC

    content = _await_callback(
        lambda cb: SC.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
            True, True, cb
        ), deadline,
    )
    if content is None:
        return None
    window = next((w for w in content.windows() if (
        int(w.windowID()) == ctx.window_id
        and w.owningApplication() is not None
        and int(w.owningApplication().processID()) == ctx.app_pid
        and w.isOnScreen()
    )), None)
    if window is None:
        return None
    frame = window.frame()
    size = _image_size(float(frame.size.width), float(frame.size.height))
    if size is None:
        return None
    # An independent-window filter excludes windows over/under the target. Never
    # fall back to a display rectangle, which could include unrelated/private apps.
    window_filter = SC.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
    config = SC.SCStreamConfiguration.alloc().init()
    config.setWidth_(size[0])
    config.setHeight_(size[1])
    config.setShowsCursor_(False)
    config.setIgnoreShadowsSingleWindow_(True)
    if hasattr(config, "setIncludeChildWindows_"):
        config.setIncludeChildWindows_(False)
    return _await_callback(
        lambda cb: SC.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            window_filter, config, cb
        ), deadline,
    )


def _clean_text(text: str, max_chars: int) -> str:
    # Preserve line boundaries; remove controls, bidi overrides, and zero-width text.
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)[:max_chars].strip()


def _reading_order(observations, max_chars: int) -> str:
    pieces = []
    for observation in observations or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        rect = observation.boundingBox()
        x, y, height = float(rect.origin.x), float(rect.origin.y), float(rect.size.height)
        pieces.append((y + height / 2, x, height, str(candidates[0].string())))
    # Vision uses a bottom-left origin. Group near-equal baselines, then read left to right.
    rows = []
    for center, x, height, text in sorted(pieces, reverse=True):
        if rows and abs(rows[-1][0] - center) <= max(0.004, min(rows[-1][1], height) * 0.5):
            rows[-1][2].append((x, text))
        else:
            rows.append((center, height, [(x, text)]))
    return _clean_text("\n".join(" ".join(t for _, t in sorted(row)) for _, _, row in rows), max_chars)


def _recognize_text(image, max_chars: int, requests: list) -> str:
    import Vision

    request = Vision.VNRecognizeTextRequest.alloc().init()
    requests.append(request)
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
    request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
    ok, error = handler.performRequests_error_([request], None)
    return _reading_order(request.results(), max_chars) if ok and error is None else ""


def capture_screen_text(ctx: Context, max_chars: int = 2500) -> str:
    """Return bounded OCR text, or empty on denial/unavailability/timeout.

    Call once at dictation start from a background worker. The complete operation
    has a two-second budget, including OCR. Timed-out native work is cancelled when
    possible and cannot queue another capture until it exits.
    """
    if sys.platform != "darwin" or ctx.secure_field or ctx.app_pid <= 0 or ctx.window_id <= 0:
        return ""
    max_chars = max(0, min(int(max_chars), _MAX_CHARS))
    if not max_chars or not _CAPTURE_LOCK.acquire(blocking=False):
        return ""
    deadline = time.monotonic() + _TIMEOUT
    done = threading.Event()
    requests, result = [], []

    def run():
        try:
            import objc
            with objc.autorelease_pool():
                if not screen_capture_allowed() or not _remaining(deadline):
                    return
                image = _capture_window_image(ctx, deadline)
                if image is not None and _remaining(deadline):
                    text = _recognize_text(image, max_chars, requests)
                    if _remaining(deadline):
                        result.append(text)
        except Exception:
            pass
        finally:
            requests.clear()
            _CAPTURE_LOCK.release()
            done.set()

    try:
        threading.Thread(target=run, name="lipflow-screen-ocr", daemon=True).start()
    except Exception:
        _CAPTURE_LOCK.release()
        return ""
    if not done.wait(_remaining(deadline)):
        for request in list(requests):
            try:
                request.cancel()
            except Exception:
                pass
        return ""
    return result[0] if result else ""
