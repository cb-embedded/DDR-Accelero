#!/usr/bin/env python3
"""
Arrow visualization tool for DDR-Accelero.

This module creates StepMania-style arrow visualizations for comparing
two sets of arrow patterns (e.g., original chart vs predictions).

The visualization shows arrows scrolling vertically, similar to StepMania gameplay:
- Left column: Player 1 arrows (e.g., original chart)
- Right column: Player 2 arrows (e.g., predictions)
- Each arrow type (Left, Down, Up, Right) has its own lane
- Time flows from bottom to top
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def extract_sm_window(sm_path, diff_level, diff_type='medium', start_time=None, duration=10.0):
    """
    Extract a time window of arrows from a StepMania .sm file.
    
    Args:
        sm_path: Path to .sm file
        diff_level: Difficulty level (int)
        diff_type: Difficulty type (e.g., 'easy', 'medium', 'hard')
        start_time: Start time in seconds (if None, uses middle of song)
        duration: Duration of window in seconds
        
    Returns:
        list of dicts: Each dict has 'time' (float) and 'arrows' (list of 4 ints [L,D,U,R])
    """
    with open(sm_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Extract BPM (use first BPM value if multiple)
    bpm_lines = [l for l in text.splitlines() if l.startswith("#BPMS:")]
    if not bpm_lines:
        raise ValueError(f"No BPM information found in {sm_path}")
    bpm_line = bpm_lines[0]
    bpm_str = bpm_line.split(":")[1].split(";")[0].split("=")[1].split(",")[0]
    bpm = float(bpm_str)
    sec_per_beat = 60.0 / bpm
    
    # Find chart at specified difficulty type and level
    blocks = text.split("#NOTES:")[1:]
    chart = None
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 6:
            if lines[2].strip(":").lower() == diff_type.lower() and int(lines[3].strip(":")) == diff_level:
                chart = lines[5:]
                break
    
    if chart is None:
        raise ValueError(f"Chart not found: {diff_type} level {diff_level}")
    
    # Parse measures
    measures = []
    cur = []
    for l in chart:
        if l == ";":
            if cur:
                measures.append(cur)
            break
        if l == ",":
            measures.append(cur)
            cur = []
        else:
            cur.append(l)
    
    # Extract arrow times and labels
    # Arrow format: LDUR (Left, Down, Up, Right)
    events = []
    t = 0.0
    for m in measures:
        n = len(m)
        for r in m:
            r = r.ljust(4, "0")
            # Only include rows with at least one arrow
            if '1' in r or '2' in r or '3' in r or '4' in r:
                # Convert to binary arrows (1=pressed, 0=not pressed)
                arrow_vec = [1 if c in ['1', '2', '3', '4'] else 0 for c in r]
                events.append({'time': t, 'arrows': arrow_vec})
            t += (4 * sec_per_beat) / n
    
    # Find total duration
    total_duration = events[-1]['time'] if events else 0
    
    # If start_time is None, use middle of song
    if start_time is None:
        start_time = (total_duration - duration) / 2.0
        start_time = max(0, start_time)  # Don't go negative
    
    # Filter events within time window
    end_time = start_time + duration
    windowed_events = [e for e in events if start_time <= e['time'] <= end_time]
    
    # Adjust times to be relative to window start
    for e in windowed_events:
        e['time'] = e['time'] - start_time
    
    return windowed_events


def visualize_arrows(player1_events, player2_events, duration=10.0, output_path=None,
                     title1="Player 1", title2="Player 2"):
    """
    Create a StepMania-style arrow visualization comparing two sets of arrows.
    
    Args:
        player1_events: List of dicts with 'time' (float) and 'arrows' (list of 4 ints)
        player2_events: List of dicts with 'time' (float) and 'arrows' (list of 4 ints)
        duration: Duration of visualization window in seconds
        output_path: Path to save PNG (if None, displays instead)
        title1: Title for left column
        title2: Title for right column
        
    Returns:
        None (saves or displays the figure)
    """
    # Arrow names and colors
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    arrow_colors = ['#FF69B4', '#00CED1', '#FFD700', '#FF4500']  # Pink, Cyan, Yellow, Red-Orange
    
    # Create figure with two columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
    
    # Configure both axes
    for ax, events, title in [(ax1, player1_events, title1), (ax2, player2_events, title2)]:
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0, duration)
        ax.set_xlabel('Arrow Lane', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(arrow_names, fontsize=9)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Draw vertical lane separators
        for i in range(4):
            ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw arrows
        arrow_size = 0.35  # Size of arrow marker
        for event in events:
            t = event['time']
            for lane, pressed in enumerate(event['arrows']):
                if pressed:
                    # Draw arrow as a colored square/diamond
                    if arrow_names[lane] == 'Left':
                        # Triangle pointing left
                        triangle = patches.Polygon([
                            [lane - arrow_size/2, t],
                            [lane + arrow_size/2, t + arrow_size/2],
                            [lane + arrow_size/2, t - arrow_size/2]
                        ], facecolor=arrow_colors[lane], edgecolor='black', linewidth=1.5)
                        ax.add_patch(triangle)
                    elif arrow_names[lane] == 'Down':
                        # Triangle pointing down
                        triangle = patches.Polygon([
                            [lane, t - arrow_size/2],
                            [lane + arrow_size/2, t + arrow_size/2],
                            [lane - arrow_size/2, t + arrow_size/2]
                        ], facecolor=arrow_colors[lane], edgecolor='black', linewidth=1.5)
                        ax.add_patch(triangle)
                    elif arrow_names[lane] == 'Up':
                        # Triangle pointing up
                        triangle = patches.Polygon([
                            [lane, t + arrow_size/2],
                            [lane + arrow_size/2, t - arrow_size/2],
                            [lane - arrow_size/2, t - arrow_size/2]
                        ], facecolor=arrow_colors[lane], edgecolor='black', linewidth=1.5)
                        ax.add_patch(triangle)
                    elif arrow_names[lane] == 'Right':
                        # Triangle pointing right
                        triangle = patches.Polygon([
                            [lane + arrow_size/2, t],
                            [lane - arrow_size/2, t + arrow_size/2],
                            [lane - arrow_size/2, t - arrow_size/2]
                        ], facecolor=arrow_colors[lane], edgecolor='black', linewidth=1.5)
                        ax.add_patch(triangle)
    
    # Main title
    fig.suptitle('DDR Arrow Pattern Comparison', fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    """
    Main function - creates a demo visualization comparing two songs.
    
    Visualizes:
    - Left: Lucky Orb (Medium 5) - middle 10 seconds
    - Right: Seyana (Medium 6) - middle 10 seconds
    """
    print("="*70)
    print("DDR ARROW VISUALIZATION - DEMO")
    print("="*70)
    
    # Extract Lucky Orb middle 10 seconds
    print("\n[1/3] Extracting Lucky Orb (Medium 5) - middle 10s...")
    lucky_events = extract_sm_window('sm_files/Lucky Orb.sm', 5, 'medium', 
                                      start_time=None, duration=10.0)
    print(f"  Found {len(lucky_events)} arrow events")
    
    # Extract Seyana middle 10 seconds
    print("\n[2/3] Extracting Seyana (Medium 6) - middle 10s...")
    seyana_events = extract_sm_window('sm_files/Seyana.sm', 6, 'medium',
                                       start_time=None, duration=10.0)
    print(f"  Found {len(seyana_events)} arrow events")
    
    # Create visualization
    print("\n[3/3] Creating visualization...")
    output_dir = Path('artifacts')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'arrow_comparison_demo.png'
    
    visualize_arrows(
        lucky_events, 
        seyana_events,
        duration=10.0,
        output_path=output_path,
        title1="Lucky Orb (Medium 5)",
        title2="Seyana (Medium 6)"
    )
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"Output: {output_path}")
    print("="*70)


if __name__ == '__main__':
    main()
