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

### Basic Usage

```bash
python pad_recorder_cli.py --music "Lucky Orb" --difficulty-number 5 --difficulty-name Medium
```

### Command Line Arguments

- `-m`, `--music`: Music name (default: "Unknown")
- `-n`, `--difficulty-number`: Difficulty number (1-10, optional)
- `-d`, `--difficulty-name`: Difficulty name (Easy/Medium/Hard/Expert/Challenge, optional)
- `-h`, `--help`: Show help message

### Examples

```bash
# Full metadata
python pad_recorder_cli.py --music "Lucky Orb" --difficulty-number 5 --difficulty-name Medium

# Short form
python pad_recorder_cli.py -m "My Song" -n 7 -d Hard

# Music name only
python pad_recorder_cli.py --music "Test Song"

# Use defaults (music will be "Unknown")
python pad_recorder_cli.py
```

### Recording Workflow

1. Connect your gamepad/dance pad
2. Run the recorder with desired metadata as arguments
3. Press Enter to start recording
4. Switch to your DDR game and play
5. When done, switch back to terminal and press Ctrl+C
6. Choose to export the log (saved to `raw_data/` directory)

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
