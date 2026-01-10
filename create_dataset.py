#!/usr/bin/env python3
"""
Create a labeled dataset from DDR sensor captures and SM files.

This script:
1. Aligns sensor data with arrow events using the biomechanical approach
2. Extracts windowed samples centered on arrow events
3. Creates X (9-channel sensor data) and Y (arrow labels) dataset
4. Generates visualization PNGs for verification
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import zipfile


def load_sensor_data(capture_path, dt=0.01):
    """Load and interpolate all sensor data (accel, gyro, magneto)."""
    with zipfile.ZipFile(capture_path, 'r') as zf:
        # Load accelerometer (Gravity)
        with zf.open('Gravity.csv') as f:
            acc = pd.read_csv(f)
        
        # Load gyroscope
        with zf.open('Gyroscope.csv') as f:
            gyro = pd.read_csv(f)
        
        # Load magnetometer
        with zf.open('Magnetometer.csv') as f:
            mag = pd.read_csv(f)
    
    # Find common time range
    t_min = max(acc["seconds_elapsed"].min(), 
                gyro["seconds_elapsed"].min(), 
                mag["seconds_elapsed"].min())
    t_max = min(acc["seconds_elapsed"].max(), 
                gyro["seconds_elapsed"].max(), 
                mag["seconds_elapsed"].max())
    
    # Create uniform time grid
    t_i = np.arange(t_min, t_max, dt)
    
    # Interpolate all sensors to uniform grid
    sensors = {}
    for axis in ['x', 'y', 'z']:
        sensors[f'acc_{axis}'] = np.interp(t_i, acc["seconds_elapsed"].to_numpy(), acc[axis].to_numpy())
        sensors[f'gyro_{axis}'] = np.interp(t_i, gyro["seconds_elapsed"].to_numpy(), gyro[axis].to_numpy())
        sensors[f'mag_{axis}'] = np.interp(t_i, mag["seconds_elapsed"].to_numpy(), mag[axis].to_numpy())
    
    return t_i, sensors


def parse_sm_file(sm_path, diff_level):
    """Parse SM file and extract arrow events with timing."""
    with open(sm_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    # Extract BPM
    bpm_line = [l for l in text.splitlines() if l.startswith("#BPMS:")][0]
    bpm = float(bpm_line.split(":")[1].split(";")[0].split("=")[1])
    sec_per_beat = 60.0 / bpm
    
    # Find Medium chart at specified level
    blocks = text.split("#NOTES:")[1:]
    chart = None
    for b in blocks:
        lines = [l.strip() for l in b.splitlines() if l.strip()]
        if len(lines) >= 6:
            if lines[2].strip(":").lower() == "medium" and int(lines[3].strip(":")) == diff_level:
                chart = lines[5:]
                break
    
    if chart is None:
        raise ValueError(f"Could not find Medium chart with difficulty {diff_level}")
    
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
    times = []
    arrows = []
    t = 0.0
    for m in measures:
        n = len(m)
        for r in m:
            r = r.ljust(4, "0")
            # Only include rows with at least one arrow
            if '1' in r or '2' in r or '3' in r or '4' in r:
                # Convert to binary arrows (1=pressed, 0=not pressed)
                arrow_vec = [1 if c in ['1', '2', '3', '4'] else 0 for c in r]
                times.append(t)
                arrows.append(arrow_vec)
            t += (4 * sec_per_beat) / n
    
    return np.array(times), np.array(arrows), bpm


def align_with_biomechanical(t_sensor, ax, t_chart, arrows, dt=0.01):
    """
    Align sensor data with arrow events using biomechanical approach.
    Uses LEFT ARROW ONLY as in align_clean.py.
    Returns time offset.
    """
    TAU = 0.10         # biomechanical time constant (s)
    BP_LOW = 0.5       # bandpass low (Hz)
    BP_HIGH = 8.0      # bandpass high (Hz)
    
    # Normalize accelerometer
    ax_norm = (ax - ax.mean()) / ax.std()
    
    # Create left arrow signal
    left_arrows = arrows[:, 0]  # First column is left arrow
    
    # Resample to uniform grid
    t_chart_i = np.arange(t_chart.min(), t_chart.max(), dt)
    s_i = np.interp(t_chart_i, t_chart, left_arrows)
    
    # Biomechanical transformation
    # 1) Edge events
    e = np.diff(s_i, prepend=0)
    
    # 2) Exponential decay kernel (causal)
    k_t = np.arange(0, 0.5, dt)
    kernel = np.exp(-k_t / TAU)
    kernel /= kernel.sum()
    
    # 3) Convolution
    p = np.convolve(e, kernel, mode="same")
    
    # 4) Bandpass filter
    fs = 1.0 / dt
    b, a = butter(2, [BP_LOW/(fs/2), BP_HIGH/(fs/2)], btype="band")
    p = filtfilt(b, a, p)
    
    p = (p - p.mean()) / p.std()
    
    # FFT correlation
    n = min(len(ax_norm), len(p))
    ax_corr = ax_norm[:n]
    p_corr = p[:n]
    
    corr = np.fft.ifft(np.fft.fft(ax_corr) * np.conj(np.fft.fft(p_corr))).real
    lags = np.arange(n) * dt
    
    peak_idx = np.argmax(corr)
    offset = lags[peak_idx]
    
    return offset


def create_dataset(t_sensor, sensors, t_arrows, arrows, offset, window_size=1.0):
    """
    Create dataset with X (sensor windows) and Y (arrow labels).
    
    Args:
        t_sensor: Time array for sensor data
        sensors: Dict of sensor channels (9 total)
        t_arrows: Time array for arrow events
        arrows: Array of arrow labels [N x 4]
        offset: Time offset from alignment (add to arrow times)
        window_size: Half-window size in seconds (default 1.0s)
    
    Returns:
        X: Array of sensor windows [N x window_samples x 9]
        Y: Array of arrow labels [N x 4]
        t_centers: Center time of each window
    """
    dt = t_sensor[1] - t_sensor[0]
    window_samples = int(window_size / dt)
    
    X = []
    Y = []
    t_centers = []
    
    # Align arrow times
    t_arrows_aligned = t_arrows + offset
    
    for i, t_arrow in enumerate(t_arrows_aligned):
        # Find center index in sensor data
        center_idx = np.searchsorted(t_sensor, t_arrow)
        
        # Check if window is within bounds
        if center_idx - window_samples < 0 or center_idx + window_samples >= len(t_sensor):
            continue
        
        # Extract window for all 9 channels
        window = []
        for channel in ['acc_x', 'acc_y', 'acc_z', 
                       'gyro_x', 'gyro_y', 'gyro_z',
                       'mag_x', 'mag_y', 'mag_z']:
            window.append(sensors[channel][center_idx - window_samples:center_idx + window_samples])
        
        X.append(np.array(window).T)  # Shape: [window_samples*2, 9]
        Y.append(arrows[i])
        t_centers.append(t_arrow)
    
    return np.array(X), np.array(Y), np.array(t_centers)


def visualize_sample(idx, X_sample, Y_sample, t_center, t_arrows, arrows, 
                     window_size=1.0, dt=0.01, output_path=None):
    """
    Visualize a single sample with sensor data and arrow chronogram.
    
    Args:
        idx: Sample index
        X_sample: Sensor data window [window_samples x 9]
        Y_sample: Arrow label [4]
        t_center: Center time of this sample
        t_arrows: All arrow times
        arrows: All arrow labels
        window_size: Half-window size in seconds
        dt: Time step
        output_path: Path to save PNG
    """
    window_samples = X_sample.shape[0] // 2
    t_window = np.arange(-window_samples, window_samples) * dt
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid: 9 sensor plots + 1 chronogram
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
    
    channel_names = ['Accel X', 'Accel Y', 'Accel Z',
                     'Gyro X', 'Gyro Y', 'Gyro Z',
                     'Mag X', 'Mag Y', 'Mag Z']
    
    # Plot 9 sensor channels
    for i in range(9):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.plot(t_window, X_sample[:, i], linewidth=0.8)
        ax.set_title(channel_names[i], fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot arrow chronogram
    ax_chrono = fig.add_subplot(gs[4, :])
    
    # Show arrows in time window
    arrow_names = ['Left', 'Down', 'Up', 'Right']
    colors = ['blue', 'green', 'orange', 'red']
    
    # Plot all arrows in the vicinity
    t_min_plot = t_center - window_size * 2
    t_max_plot = t_center + window_size * 2
    
    for i, arrow_name in enumerate(arrow_names):
        # Find arrows in plot range
        for j, t_arr in enumerate(t_arrows):
            if t_min_plot <= t_arr <= t_max_plot and arrows[j][i] == 1:
                ax_chrono.axvline(t_arr - t_center, color=colors[i], 
                                 alpha=0.3, linewidth=2)
    
    # Highlight the center arrow (label)
    for i, arrow_name in enumerate(arrow_names):
        if Y_sample[i] == 1:
            ax_chrono.axvline(0, color=colors[i], linestyle='-', 
                            linewidth=3, label=f'{arrow_name} (LABEL)')
    
    ax_chrono.set_xlim([-window_size, window_size])
    ax_chrono.set_xlabel('Time relative to center (s)', fontsize=10)
    ax_chrono.set_ylabel('Arrow Events', fontsize=10)
    ax_chrono.set_title(f'Arrow Chronogram - Label: {[arrow_names[i] for i, v in enumerate(Y_sample) if v == 1]}', 
                       fontsize=11)
    ax_chrono.legend(fontsize=8)
    ax_chrono.grid(True, alpha=0.3)
    
    # Add text with label info
    label_str = '+'.join([arrow_names[i] for i, v in enumerate(Y_sample) if v == 1])
    fig.suptitle(f'Sample #{idx} - Label: {label_str}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        print("\nUsage: python create_dataset.py <capture.zip> <song.sm> <difficulty_level> <num_samples>")
        print("Example: python create_dataset.py 'raw_data/Lucky_Orb_5_Medium-....zip' 'sm_files/Lucky Orb.sm' 5 10")
        sys.exit(1)
    
    capture_path = Path(sys.argv[1])
    sm_path = Path(sys.argv[2])
    diff_level = int(sys.argv[3])
    num_samples = int(sys.argv[4])
    
    print("="*70)
    print("DDR DATASET CREATION")
    print("="*70)
    
    # Load sensor data
    print("\n[1/5] Loading sensor data (accel, gyro, magneto)...")
    t_sensor, sensors = load_sensor_data(capture_path)
    print(f"  Duration: {t_sensor[-1] - t_sensor[0]:.1f}s, Samples: {len(t_sensor)}")
    print(f"  9 channels: acc_x,y,z, gyro_x,y,z, mag_x,y,z")
    
    # Parse SM file
    print("\n[2/5] Parsing .sm file...")
    t_arrows, arrows, bpm = parse_sm_file(sm_path, diff_level)
    print(f"  BPM: {bpm}, Arrow events: {len(t_arrows)}")
    
    # Align using biomechanical approach
    print("\n[3/5] Aligning sensor data with arrows...")
    offset = align_with_biomechanical(t_sensor, sensors['acc_x'], t_arrows, arrows)
    print(f"  Offset: {offset:.3f}s")
    
    # Create dataset
    print("\n[4/5] Creating dataset...")
    X, Y, t_centers = create_dataset(t_sensor, sensors, t_arrows, arrows, offset)
    print(f"  Dataset size: {len(X)} samples")
    print(f"  X shape: {X.shape} (samples x window_length x 9_channels)")
    print(f"  Y shape: {Y.shape} (samples x 4_arrows)")
    
    # Generate visualizations
    print(f"\n[5/5] Generating {num_samples} sample visualizations...")
    out_dir = Path('artifacts')
    out_dir.mkdir(exist_ok=True)
    
    # Select samples evenly distributed
    if len(X) < num_samples:
        num_samples = len(X)
        print(f"  Note: Only {num_samples} samples available")
    
    indices = np.linspace(0, len(X)-1, num_samples, dtype=int)
    
    # Align arrow times for visualization
    t_arrows_aligned = t_arrows + offset
    
    for i, idx in enumerate(indices):
        output_path = out_dir / f'dataset_sample_{i:02d}.png'
        visualize_sample(i, X[idx], Y[idx], t_centers[idx], 
                        t_arrows_aligned, arrows, 
                        output_path=output_path)
    
    print("\n" + "="*70)
    print("DATASET CREATION COMPLETE")
    print("="*70)
    print(f"Total samples: {len(X)}")
    print(f"Visualizations saved: {num_samples} PNGs in {out_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
