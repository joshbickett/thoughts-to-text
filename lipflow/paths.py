"""Where each person's data lives. Shared model weights stay in the repo's models/; everything
learned from *you* (clips, phrases, personal models, settings) goes here.

Override with THOUGHTS_TO_TEXT_HOME. LIPFLOW_HOME remains a compatibility fallback."""
import os
import sys

WINDOWS = sys.platform == "win32"
LINUX = sys.platform == "linux"
DESKTOP_TRAY = WINDOWS or LINUX

if WINDOWS:  # %APPDATA%\ThoughtsToText
    _DEFAULT_HOME = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "ThoughtsToText")
elif LINUX:
    _DEFAULT_HOME = os.path.join(
        os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
        "ThoughtsToText",
    )
else:
    _DEFAULT_HOME = os.path.expanduser("~/Library/Application Support/ThoughtsToText")
HOME = os.environ.get("THOUGHTS_TO_TEXT_HOME") or os.environ.get("LIPFLOW_HOME") or _DEFAULT_HOME
PERSONAL_MODELS = os.path.join(HOME, "models")
PERSONAL_VSR = os.path.join(PERSONAL_MODELS, "vsr_face.pth")
PERSONAL_LM = os.path.join(PERSONAL_MODELS, "lm_phrasing.pth")

# The app launcher's permissions belong to Thoughts to Text, not the terminal.
# On Windows nothing is granted per app, so the name only shows up in messages.
WHO = "Thoughts to Text" if (os.environ.get("THOUGHTS_TO_TEXT_APP") or os.environ.get("LIPFLOW_APP")
                             or DESKTOP_TRAY) else "your terminal"
