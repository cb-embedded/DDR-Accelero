#!/usr/bin/env python3
"""
Experiment 2: Parse StepMania .sm files and extract note timing information.

This script parses .sm files to extract:
- BPM (beats per minute)
- Offset (initial delay)
- Note patterns and their timestamps
This will be used for alignment with sensor data.
"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import sys


@dataclass
class SMNote:
    """Represents a single note/arrow in StepMania."""
    time: float  # Time in seconds
    arrows: str  # 4-character string: left, down, up, right (0=none, 1=tap, 2=hold start, 3=hold end)
    
    def get_active_arrows(self):
        """Return list of arrow indices that are active (non-zero)."""
        arrow_names = ['left', 'down', 'up', 'right']
        return [arrow_names[i] for i, val in enumerate(self.arrows) if val != '0']


@dataclass
class SMChart:
    """Represents a StepMania chart."""
    title: str
    artist: str
    bpm: float
    offset: float
    difficulty: str
    notes: List[SMNote]
    
    def get_note_density(self, window_sec=1.0):
        """Calculate note density (notes per second) over time."""
        if not self.notes:
            return []
        
        max_time = self.notes[-1].time
        densities = []
        
        for t in range(0, int(max_time) + 1, int(window_sec)):
            window_notes = [n for n in self.notes if t <= n.time < t + window_sec]
            densities.append((t, len(window_notes) / window_sec))
        
        return densities


def parse_sm_file(sm_path):
    """Parse a StepMania .sm file and extract chart information."""
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract metadata
    title_match = re.search(r'#TITLE:([^;]+);', content)
    artist_match = re.search(r'#ARTIST:([^;]+);', content)
    offset_match = re.search(r'#OFFSET:([^;]+);', content)
    bpm_match = re.search(r'#BPMS:([^;]+);', content)
    
    title = title_match.group(1).strip() if title_match else "Unknown"
    artist = artist_match.group(1).strip() if artist_match else "Unknown"
    offset = float(offset_match.group(1)) if offset_match else 0.0
    
    # Parse BPM (can have multiple values, take the first for simplicity)
    bpm = 120.0  # default
    if bpm_match:
        bpm_str = bpm_match.group(1).strip()
        # Format: beat=bpm or just bpm
        if '=' in bpm_str:
            bpm = float(bpm_str.split('=')[1].split(',')[0])
        else:
            bpm = float(bpm_str.split(',')[0])
    
    # Find all charts in the file
    charts = []
    chart_pattern = r'#NOTES:\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^:]+):\s*([^;]+);'
    
    for match in re.finditer(chart_pattern, content, re.MULTILINE | re.DOTALL):
        chart_type = match.group(1).strip()
        difficulty = match.group(3).strip()
        notes_data = match.group(6).strip()
        
        # Only process dance-single charts
        if 'dance-single' not in chart_type:
            continue
        
        # Parse notes
        notes = parse_notes(notes_data, bpm, offset)
        
        chart = SMChart(
            title=title,
            artist=artist,
            bpm=bpm,
            offset=offset,
            difficulty=difficulty,
            notes=notes
        )
        charts.append(chart)
    
    return charts


def parse_notes(notes_data, bpm, offset):
    """Parse note data from a chart."""
    notes = []
    
    # Split by comma to get measures (4 beats each)
    measures = notes_data.split(',')
    
    # Calculate time per beat
    beat_duration = 60.0 / bpm
    
    current_time = offset
    
    for measure in measures:
        # Split measure into lines (each line is a subdivision of the measure)
        lines = [line.strip() for line in measure.strip().split('\n') if line.strip() and len(line.strip()) == 4]
        
        if not lines:
            continue
        
        # Each measure is 4 beats, subdivided into len(lines) parts
        subdivision_duration = (4 * beat_duration) / len(lines)
        
        for i, line in enumerate(lines):
            # Check if any arrow is active
            if any(c != '0' for c in line):
                note_time = current_time + (i * subdivision_duration)
                notes.append(SMNote(time=note_time, arrows=line))
        
        # Move to next measure
        current_time += 4 * beat_duration
    
    return notes


def main():
    # Get the sm_files directory
    sm_dir = Path(__file__).parent.parent / 'sm_files'
    
    if not sm_dir.exists():
        print(f"Error: sm_files directory not found at {sm_dir}")
        sys.exit(1)
    
    # List all .sm files
    sm_files = list(sm_dir.glob('*.sm'))
    
    if not sm_files:
        print("No .sm files found in sm_files directory")
        sys.exit(1)
    
    print(f"Found {len(sm_files)} StepMania files\n")
    
    # Analyze Lucky Orb as an example (matches one of our sensor captures)
    target_file = None
    for f in sm_files:
        if 'Lucky' in f.name and 'Orb' in f.name:
            target_file = f
            break
    
    if not target_file:
        if not sm_files:
            print("No .sm files found")
            sys.exit(1)
        target_file = sm_files[0]
    
    print(f"Analyzing: {target_file.name}\n")
    
    # Parse the file
    charts = parse_sm_file(target_file)
    
    print(f"Found {len(charts)} charts in this file:\n")
    
    for chart in charts:
        print(f"Difficulty: {chart.difficulty}")
        print(f"  Title: {chart.title}")
        print(f"  Artist: {chart.artist}")
        print(f"  BPM: {chart.bpm}")
        print(f"  Offset: {chart.offset}s")
        print(f"  Total notes: {len(chart.notes)}")
        
        if chart.notes:
            print(f"  Duration: {chart.notes[-1].time:.2f}s")
            print(f"  First note at: {chart.notes[0].time:.2f}s")
            
            # Show first few notes
            print(f"\n  First 5 notes:")
            for i, note in enumerate(chart.notes[:5]):
                arrows = note.get_active_arrows()
                print(f"    {i+1}. Time: {note.time:.3f}s, Arrows: {', '.join(arrows)}")
        
        print()
    
    # Analyze note density for Medium difficulty
    medium_chart = None
    for chart in charts:
        if 'Medium' in chart.difficulty or 'medium' in chart.difficulty.lower():
            medium_chart = chart
            break
    
    if medium_chart:
        print("Note density analysis (Medium difficulty):")
        densities = medium_chart.get_note_density(window_sec=5.0)
        print(f"  Average density: {sum(d[1] for d in densities) / len(densities):.2f} notes/second")
        print(f"  Peak density: {max(d[1] for d in densities):.2f} notes/second")
    
    print("\n" + "="*50)
    print("Experiment 2 completed successfully!")
    print("="*50)
    print("\nKey observations:")
    print("1. StepMania files contain timing information (BPM, offset)")
    print("2. Notes are timestamped relative to the music start")
    print("3. Next step: align sensor data timestamps with note timestamps")


if __name__ == '__main__':
    main()
