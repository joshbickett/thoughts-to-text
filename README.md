# Thoughts to Text

Hold a key, silently mouth a sentence, and release to paste text at your cursor.
Thoughts to Text is an independent fork of [Lipflow](https://github.com/amywork777/lipflow)
with a separate macOS app, its own settings and personal data, and optional screen text
context for dictation cleanup.

The webcam lip reader runs locally. Cleanup can use a local model or a remote provider.
On macOS, the app can read the text around your cursor and use local OCR to read the
active conversation or document. This gives the cleanup model useful names and phrasing;
it cannot reliably recover every word from ambiguous lip movements. Review the output
before using it where mistakes matter.

## Install on macOS

Use an Apple Silicon Mac with macOS 14 or later and Apple's Command Line Tools
(`xcode-select --install` if needed). Run setup from a native arm64 terminal, outside
Rosetta. Python 3.11–3.12 is supported; setup uses uv to find or install a compatible
interpreter. Allow several gigabytes for dependencies and model weights.

Clone the repository, then run setup:

```sh
git clone https://github.com/joshbickett/thoughts-to-text.git
cd thoughts-to-text
./setup.sh
open /Applications/Thoughts-to-Text.app
```

Setup installs dependencies, downloads the pretrained models, and builds a native menu-bar
app with a teal text-bubble icon. The app loads this checkout, so keep the checkout and its
`.venv` in place. It has its own bundle identity and does not replace `/Applications/Lipflow.app`.

First launch offers:

1. **Camera, Input Monitoring and Accessibility permissions** for Thoughts to Text.
   If a permission is enabled in System Settings but has not taken effect, restart the app.
2. **Optional Screen Recording permission** for local screen OCR. Dictation and text-field
   context can work without this permission.
3. **Optional Wispr Flow history import** to learn familiar words and phrasing.
4. **Practice and training** using recorded examples of your silent speech. The existing
   training flow compares the personalized lip reader with the stock model on held-out clips.

To rebuild an existing installation after updating this checkout:

```sh
uv sync
uv run python scripts/make_app.py
```

Quit the app before rebuilding, then reopen it. To launch at login, add
**Thoughts to Text** in System Settings → General → Login Items.

## Dictate

| Action | Result |
| --- | --- |
| Hold **Right Option**, mouth words, release | Dictate and paste |
| Double-tap **Right Option**, then tap again | Hands-free dictation |
| Press **Esc** while listening | Cancel |
| Menu → **Copy last dictation** or **Open history** | Retrieve previous output |

Choose a different shortcut, camera, cleanup provider, or context settings in the app.
Optional whisper mode uses local microphone audio together with the lip reader and
downloads an additional audio-visual model. Ordinary silent dictation does not use audio.

The command-line entry point is `thoughts-to-text`; `lipflow` remains a compatibility alias:

```sh
uv run thoughts-to-text --help
uv run thoughts-to-text doctor
uv run thoughts-to-text --cleanup codex
uv run thoughts-to-text --copy-only
uv run thoughts-to-text file talk.mp4 --start 10 --end 20
uv run thoughts-to-text import-wispr
```

Running from a terminal uses the terminal's macOS permissions. The native app has its own
permission entries.

## Screen context and privacy

When surrounding context is enabled, the app reads the active window title and nearby
text from the focused field through macOS Accessibility. With the separate screen-text
option and Screen Recording permission enabled, it captures the target window once at
dictation start and recognizes text with Apple Vision on this Mac. It does not capture
the entire desktop. Recognized secure fields are skipped, and unavailable or slow OCR
does not prevent dictation.

Screenshots and OCR context stay in memory; the app does not save or upload screenshots.
**Cloud cleanup sends text to the selected provider**, including the lip reader's guesses
and relevant context such as visible text, names, recent dictations, or imported phrases.
Choose local cleanup or offline rules to keep that processing on your computer. Disable
screen text in Settings to stop screenshot capture, or turn off surrounding app text and
names to disable surrounding app context. Recent dictations and personal vocabulary can
still be used by cleanup.

Camera frames and optional whisper audio are processed locally. Practice clips,
personalized models, phrases, settings, dictation history, and enabled correction-learning
data are stored locally. Wispr Flow history import reads its local database; relevant
imported phrases may later be used as cleanup context, including with a cloud provider.

## Cleanup providers

Cleanup works with the lip reader's candidate sentences; it does not replace the visual
speech model. A larger model or more context can still choose the wrong sentence.

| Provider | Processing | Setup |
| --- | --- | --- |
| ChatGPT through Codex CLI | Remote | Use an authenticated, compatible Codex installation; select it under Cleanup |
| Claude | Remote | Set `ANTHROPIC_API_KEY`; select Claude |
| Qwen through MLX | Local, Apple Silicon | Downloaded by setup or on first use |
| Ollama | Local server at `127.0.0.1:11434` | Install a local model, such as `qwen3:4b` |
| Offline rules | Local | No model or account needed |

**Automatic** uses Claude when Anthropic credentials are present; otherwise it tries MLX,
then Ollama, then offline rules. Choose a provider explicitly and check the active backend
shown in the Cleanup menu when you need a particular processing location. If a requested
backend is unavailable, the app can fall back to Automatic.

The Codex executable can be selected with `LIPFLOW_CODEX_BIN` or `codex_binary` in the
app's `settings.json`; otherwise it is found on `PATH`. Select its model with
`LIPFLOW_CODEX_MODEL` or `codex_model` in settings. Existing `LIPFLOW_*` provider overrides
are retained for compatibility. Remote providers have their own account, usage, and
data handling terms.

## Data and logs

| Platform | Personal data |
| --- | --- |
| macOS | `~/Library/Application Support/ThoughtsToText/` |
| Windows | `%APPDATA%\ThoughtsToText\` |
| Linux | `$XDG_DATA_HOME/ThoughtsToText/`, usually `~/.local/share/ThoughtsToText/` |

Set `THOUGHTS_TO_TEXT_HOME` to use another data directory. The legacy `LIPFLOW_HOME`
override is honored if the new variable is unset or empty. Without an explicit override,
this fork uses separate personal data from Lipflow. Shared pretrained weights are loaded
from this checkout's `models/` directory.

Native macOS app logs are in `~/Library/Logs/ThoughtsToText.log`.
The app bundle is `/Applications/Thoughts-to-Text.app`, with bundle identifier
`local.thoughtstotext.app`.

## Windows and Linux

The inherited cross-platform dictation paths are available; the new screen OCR integration
is macOS-only. These platforms default to **Right Ctrl** for dictation and use window-title
context.

On Windows, run `powershell -ExecutionPolicy Bypass -File setup.ps1` from this checkout,
then open **Thoughts to Text** from the Start menu. Enable desktop camera access in Windows
privacy settings if needed.

On Linux, run `./setup-linux.sh`, then `uv run thoughts-to-text`. Wayland clipboard/paste
requires `wl-clipboard` and `wtype` or `ydotool`; X11 requires `xclip` and `xdotool`.
The tray requires PyGObject and an AppIndicator typelib. Wayland global hotkeys need access
to the relevant `/dev/input` devices.

MLX cleanup is Apple Silicon-only. Windows and Linux can use a remote cleanup provider,
Ollama, or offline rules. Model execution uses the CPU unless a compatible GPU setup is
available. Run `uv run thoughts-to-text doctor` for local diagnostics.

## Model and attribution

The visual speech model is the Auto-AVSR LRS3 `LRS3_V_WER19.1` checkpoint, with a convolutional
front end, Conformer encoder, Transformer decoder, CTC, and a separate language model for
beam search. Its published benchmark name is not a measured accuracy guarantee for silent
dictation on your webcam. Training footage, lighting, pose, articulation, and vocabulary
all affect results.

This fork retains the Python package name `lipflow` and the upstream [MIT license](LICENSE),
including Amy Zhou's copyright. See [NOTICE](NOTICE) for third-party code and model
attributions, including [Auto-AVSR](https://github.com/mpc001/auto_avsr),
[Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages),
ESPnet, and Chaplin by Amanvir Parhar.

The upstream NOTICE identifies LRS3 as licensed for non-commercial research and instructs
users to treat the downloaded model weights accordingly. This fork does not change those
restrictions or provide a commercially cleared model. Model weights are downloaded
separately and are not covered by the application's MIT license.
