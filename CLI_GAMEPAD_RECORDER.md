# CLI Gamepad Recorder Usage Guide

## Purpose

This CLI tool captures gamepad/dance pad button presses in the background, allowing you to record your gameplay while the game window has focus.

## Why CLI instead of Web?

Modern web browsers stop polling gamepad input when the window loses focus (security/performance feature). This means the web recorder can't capture events while you're playing DDR. The CLI tool doesn't have this limitation.

## Requirements

```bash
pip install pygame
```

## Usage

1. Connect your gamepad/dance pad
2. Run the recorder:
   ```bash
   python pad_recorder_cli.py
   ```
3. Enter metadata (music name, difficulty)
4. Press Enter to start recording
5. Switch to your DDR game and play
6. When done, switch back to terminal and press Ctrl+C
7. Choose to export the log (saved to `raw_data/` directory)

## Output Format

The tool exports logs in the same CSV format as the web recorder:
- YAML frontmatter with metadata
- CSV columns: timestamp, button, pressed
- Button names: LEFT, RIGHT, UP, DOWN
- Timestamps in seconds with 3 decimal precision

## Troubleshooting

**No gamepad detected:**
- Ensure gamepad is connected before running the script
- On Linux, you may need proper permissions for `/dev/input/`
- Try running: `ls -l /dev/input/js*`

**Pygame audio error:**
- This can be ignored - audio isn't needed for gamepad capture
- To suppress: `SDL_AUDIODRIVER=dummy python pad_recorder_cli.py`

## Example Output Filename

```
2026-01-16_14-30-45_Lucky_Orb_(5_Medium).csv
```

Format: `{timestamp}_{music_name}_({difficulty}).csv`
