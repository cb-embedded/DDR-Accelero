# Arrow Visualization Module

This module provides StepMania-style arrow visualizations for comparing DDR arrow patterns.

## Overview

The `visualize_arrows.py` module creates visual comparisons of arrow patterns, displaying them in a vertical scrolling format similar to StepMania gameplay. This is useful for:
- Comparing original charts with predictions
- Visualizing different difficulty levels side-by-side
- Analyzing arrow patterns across songs

## Features

- **Dual-column layout**: Compare two arrow patterns side-by-side
- **StepMania-style visualization**: Vertical scrolling with color-coded arrows
- **Flexible time windows**: Extract any duration from any part of a song
- **Simple input format**: Easy-to-use event structure
- **PNG export**: High-quality output for documentation and analysis

## Functions

### `extract_sm_window(sm_path, diff_level, diff_type='medium', start_time=None, duration=10.0)`

Extracts a time window of arrows from a StepMania .sm file.

**Parameters:**
- `sm_path` (str): Path to .sm file
- `diff_level` (int): Difficulty level (e.g., 5 for Medium-5)
- `diff_type` (str): Difficulty type ('easy', 'medium', 'hard', 'challenge', 'beginner')
- `start_time` (float, optional): Start time in seconds. If None, uses middle of song
- `duration` (float): Duration of window in seconds (default: 10.0)

**Returns:**
- `list of dict`: Each dict has:
  - `'time'` (float): Time in seconds (relative to window start)
  - `'arrows'` (list of 4 ints): Binary vector [Left, Down, Up, Right] where 1=pressed, 0=not pressed

**Example:**
```python
events = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium', start_time=70.0, duration=10.0)
# Returns events like: [{'time': 0.5, 'arrows': [1, 0, 0, 0]}, ...]
```

### `visualize_arrows(player1_events, player2_events, duration=10.0, output_path=None, title1="Player 1", title2="Player 2")`

Creates a StepMania-style arrow visualization comparing two sets of arrows.

**Parameters:**
- `player1_events` (list): Events for left column (see format below)
- `player2_events` (list): Events for right column (see format below)
- `duration` (float): Duration of visualization window in seconds (default: 10.0)
- `output_path` (str, optional): Path to save PNG. If None, displays instead
- `title1` (str): Title for left column (default: "Player 1")
- `title2` (str): Title for right column (default: "Player 2")

**Event Format:**
Each event is a dict with:
- `'time'` (float): Time in seconds within the window (0 to duration)
- `'arrows'` (list of 4 ints): Binary vector [Left, Down, Up, Right]

**Example:**
```python
# Simple example with custom events
player1 = [
    {'time': 1.0, 'arrows': [1, 0, 0, 0]},  # Left arrow at 1s
    {'time': 2.0, 'arrows': [0, 1, 0, 0]},  # Down arrow at 2s
    {'time': 3.0, 'arrows': [0, 0, 1, 0]},  # Up arrow at 3s
    {'time': 4.0, 'arrows': [0, 0, 0, 1]},  # Right arrow at 4s
    {'time': 5.0, 'arrows': [1, 0, 1, 0]},  # Left+Up at 5s (double arrow)
]

player2 = [
    {'time': 1.1, 'arrows': [1, 0, 0, 0]},  # Slightly delayed prediction
    {'time': 2.0, 'arrows': [0, 1, 0, 0]},
    # ... more events
]

visualize_arrows(player1, player2, duration=10.0, 
                 output_path='comparison.png',
                 title1='Original Chart', title2='Predicted')
```

## Usage Examples

### Example 1: Compare two songs

```python
from visualize_arrows import extract_sm_window, visualize_arrows

# Extract middle 10 seconds from two different songs
lucky_events = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium')
seyana_events = extract_sm_window('sm_files/Seyana.sm', 6, 'medium')

# Visualize comparison
visualize_arrows(lucky_events, seyana_events,
                 output_path='song_comparison.png',
                 title1='Lucky Orb (Medium 5)',
                 title2='Seyana (Medium 6)')
```

### Example 2: Compare original chart with predictions

```python
from visualize_arrows import extract_sm_window, visualize_arrows

# Get original chart from .sm file
original = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium', 
                              start_time=50.0, duration=15.0)

# Your prediction events (same format)
predictions = [
    {'time': 0.2, 'arrows': [1, 0, 0, 0]},
    {'time': 1.5, 'arrows': [0, 1, 0, 0]},
    # ... more predicted events
]

# Visualize
visualize_arrows(original, predictions,
                 duration=15.0,
                 output_path='prediction_comparison.png',
                 title1='Original Chart',
                 title2='ML Predictions')
```

### Example 3: Run the demo

```bash
# Creates visualization comparing Lucky Orb and Seyana
python visualize_arrows.py
# Output: artifacts/arrow_comparison_demo.png
```

## Visualization Details

### Arrow Colors
- **Pink** (#FF69B4): Left arrow
- **Cyan** (#00CED1): Down arrow  
- **Yellow** (#FFD700): Up arrow
- **Red-Orange** (#FF4500): Right arrow

### Layout
- **X-axis**: 4 lanes (Left, Down, Up, Right)
- **Y-axis**: Time in seconds (flows bottom to top)
- **Arrows**: Triangular shapes pointing in their respective directions
- **Grid**: Horizontal lines at 1-second intervals for easy reading

### Arrow Shapes
- Left: Triangle pointing ◄ left
- Down: Triangle pointing ▼ down
- Up: Triangle pointing ▲ up
- Right: Triangle pointing ► right

## Integration with Other Agents

This module is designed for easy integration:

1. **Simple input format**: Just provide time + arrows (no complex dependencies)
2. **Flexible extraction**: Get arrow data from .sm files or create your own
3. **Standard output**: PNG files for documentation and analysis
4. **Modular design**: Import functions individually as needed

### For ML prediction visualization:
```python
# After running predictions on sensor data
from visualize_arrows import extract_sm_window, visualize_arrows

# Get ground truth
ground_truth = extract_sm_window(sm_file, diff_level, 'medium', 
                                  start_time=t_start, duration=duration)

# Format your predictions
predictions = []
for pred_time, pred_arrows in your_predictions:
    predictions.append({
        'time': pred_time - t_start,  # Relative to window start
        'arrows': pred_arrows.tolist()  # [L, D, U, R] binary
    })

# Visualize
visualize_arrows(ground_truth, predictions,
                 output_path='ml_results.png',
                 title1='Ground Truth',
                 title2='Model Predictions')
```

## Demo Output

Running `python visualize_arrows.py` produces `artifacts/arrow_comparison_demo.png`:
- **Left column**: Lucky Orb (Medium 5) - middle 10 seconds (23 arrow events)
- **Right column**: Seyana (Medium 6) - middle 10 seconds (19 arrow events)

This demonstrates the module's ability to extract and visualize arrow patterns from StepMania .sm files.

## Requirements

- numpy
- matplotlib

Already included in the project's `requirements.txt`.

## Notes

- Time flows from **bottom to top** (like StepMania gameplay)
- Double arrows (simultaneous presses) are displayed on the same horizontal line
- The visualization automatically scales to the specified duration
- Arrow events outside the time window are ignored
