#!/usr/bin/env python3
"""
Visualize DDR session data: gamepad inputs and sensor readings
"""

import struct
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np
from scipy import signal


def parse_csv_frontmatter(csv_path):
    """Parse CSV with YAML-like frontmatter"""
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Parse frontmatter
    if lines[0].strip() == '---':
        frontmatter = {}
        i = 1
        while i < len(lines) and lines[i].strip() != '---':
            if ':' in lines[i]:
                key, value = lines[i].split(':', 1)
                frontmatter[key.strip()] = value.strip()
            i += 1
        
        # Skip header line after second ---
        data_start = i + 2  # Skip '---' and 'timestamp_ms,button,state'
    else:
        frontmatter = {}
        data_start = 1  # Skip header
    
    # Parse data
    events = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) == 3:
            timestamp_ms = int(parts[0])
            button = parts[1]
            state = int(parts[2])
            events.append((timestamp_ms, button, state))
    
    return frontmatter, events


def parse_ddrbin(ddrbin_path):
    """Parse .ddrbin sensor file"""
    with open(ddrbin_path, 'rb') as f:
        # Read header
        header = f.read(32)
        magic = struct.unpack('<I', header[0:4])[0]
        version = struct.unpack('<B', header[4:5])[0]
        start_time_ns = struct.unpack('<Q', header[5:13])[0]
        
        if magic != 0x44444143:
            raise ValueError("Invalid .ddrbin file")
        
        # Read records
        accel_data = {'t': [], 'x': [], 'y': [], 'z': []}
        gyro_data = {'t': [], 'x': [], 'y': [], 'z': []}
        
        while True:
            record = f.read(21)
            if len(record) < 21:
                break
            
            sensor_type = struct.unpack('<B', record[0:1])[0]
            timestamp_ns = struct.unpack('<Q', record[1:9])[0]
            x = struct.unpack('<f', record[9:13])[0]
            y = struct.unpack('<f', record[13:17])[0]
            z = struct.unpack('<f', record[17:21])[0]
            
            # Convert to ms relative to start
            rel_ms = (timestamp_ns - start_time_ns) / 1e6
            
            if sensor_type == 1:  # Accelerometer
                accel_data['t'].append(rel_ms)
                accel_data['x'].append(x)
                accel_data['y'].append(y)
                accel_data['z'].append(z)
            elif sensor_type == 2:  # Gyroscope
                gyro_data['t'].append(rel_ms)
                gyro_data['x'].append(x)
                gyro_data['y'].append(y)
                gyro_data['z'].append(z)
    
    return start_time_ns, accel_data, gyro_data


def align_signals(pad_events, accel_data):
    """Align gamepad and accelerometer data using cross-correlation"""
    
    # Create time grid (1ms resolution)
    max_time = max(
        max([t for t, _, _ in pad_events]) if pad_events else 0,
        max(accel_data['t']) if accel_data['t'] else 0
    )
    time_grid = np.arange(0, max_time + 100, 1.0)  # 1ms bins
    
    # Create button activity signal (count of button presses per ms)
    pad_signal = np.zeros(len(time_grid))
    for ts, btn, state in pad_events:
        if state == 1:  # Only count presses
            idx = int(ts)
            if 0 <= idx < len(pad_signal):
                pad_signal[idx] += 1
    
    # Create accelerometer activity signal (magnitude of acceleration changes)
    accel_signal = np.zeros(len(time_grid))
    if len(accel_data['t']) > 1:
        for i in range(1, len(accel_data['t'])):
            t = accel_data['t'][i]
            # Calculate magnitude change
            dx = accel_data['x'][i] - accel_data['x'][i-1]
            dy = accel_data['y'][i] - accel_data['y'][i-1]
            dz = accel_data['z'][i] - accel_data['z'][i-1]
            magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
            
            idx = int(t)
            if 0 <= idx < len(accel_signal):
                accel_signal[idx] = max(accel_signal[idx], magnitude)
    
    # Smooth signals
    window = 50  # 50ms window
    pad_signal = np.convolve(pad_signal, np.ones(window)/window, mode='same')
    accel_signal = np.convolve(accel_signal, np.ones(window)/window, mode='same')
    
    # Compute cross-correlation
    correlation = signal.correlate(accel_signal, pad_signal, mode='full')
    lags = signal.correlation_lags(len(accel_signal), len(pad_signal), mode='full')
    
    # Find optimal lag
    optimal_lag = lags[np.argmax(correlation)]
    
    print(f"Cross-correlation offset: {optimal_lag:.1f} ms")
    
    return optimal_lag, correlation, lags


def plot_session(csv_path, ddrbin_path):
    """Create visualization of gamepad and sensor data"""
    
    # Parse files
    frontmatter, pad_events = parse_csv_frontmatter(csv_path)
    start_time_ns, accel_data, gyro_data = parse_ddrbin(ddrbin_path)
    
    # Normalize timestamps to start at 0
    if pad_events:
        first_pad_ts = pad_events[0][0]
        pad_events = [(ts - first_pad_ts, btn, state) for ts, btn, state in pad_events]
    
    if accel_data['t']:
        first_accel_ts = accel_data['t'][0]
        accel_data['t'] = [t - first_accel_ts for t in accel_data['t']]
    
    if gyro_data['t']:
        first_gyro_ts = gyro_data['t'][0]
        gyro_data['t'] = [t - first_gyro_ts for t in gyro_data['t']]
    
    # Align using cross-correlation
    offset_ms, correlation, lags = align_signals(pad_events, accel_data)
    
    # Apply offset to pad events
    pad_events = [(ts + offset_ms, btn, state) for ts, btn, state in pad_events]
    
    # Plot cross-correlation in separate window
    fig_corr, ax_corr = plt.subplots(figsize=(12, 6))
    ax_corr.plot(lags, correlation, 'b-', linewidth=1)
    ax_corr.axvline(offset_ms, color='r', linestyle='--', linewidth=2, label=f'Optimal offset: {offset_ms:.1f} ms')
    
    # Calculate FWHM (Full Width at Half Maximum)
    max_corr = np.max(correlation)
    half_max = max_corr / 2
    max_idx = np.argmax(correlation)
    
    # Find left and right points at half maximum
    left_idx = max_idx
    while left_idx > 0 and correlation[left_idx] > half_max:
        left_idx -= 1
    
    right_idx = max_idx
    while right_idx < len(correlation) - 1 and correlation[right_idx] > half_max:
        right_idx += 1
    
    fwhm_left = lags[left_idx]
    fwhm_right = lags[right_idx]
    fwhm_width = fwhm_right - fwhm_left
    
    # Draw FWHM visualization
    ax_corr.axhline(half_max, color='g', linestyle=':', linewidth=1, alpha=0.5)
    ax_corr.axvline(fwhm_left, color='g', linestyle=':', linewidth=1, alpha=0.5)
    ax_corr.axvline(fwhm_right, color='g', linestyle=':', linewidth=1, alpha=0.5)
    ax_corr.plot([fwhm_left, fwhm_right], [half_max, half_max], 'g-', linewidth=2, 
                 marker='|', markersize=15, label=f'FWHM: {fwhm_width:.1f} ms')
    
    ax_corr.set_xlabel('Lag (ms)')
    ax_corr.set_ylabel('Cross-correlation')
    ax_corr.set_title('Cross-correlation: Gamepad vs Accelerometer Activity')
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend()
    fig_corr.tight_layout()
    
    # Get session info
    pad_start_epoch = int(frontmatter.get('start_time', 0))
    pad_start_dt = datetime.fromtimestamp(pad_start_epoch)
    sensor_start_dt = datetime.fromtimestamp(start_time_ns / 1e9)
    
    # Create figure with 7 subplots
    fig, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f'DDR Session - {pad_start_dt.strftime("%Y-%m-%d %H:%M:%S")}', 
                 fontsize=14, fontweight='bold')
    
    # Subplot 0: Gamepad buttons
    ax_pad = axes[0]
    button_map = {'l': 0, 'd': 1, 'u': 2, 'r': 3}
    button_labels = ['Left', 'Down', 'Up', 'Right']
    colors = {'l': 'blue', 'd': 'orange', 'u': 'green', 'r': 'red'}
    
    # Track button press intervals
    button_press_start = {}
    for timestamp_ms, button, state in pad_events:
        if button in button_map:
            y = button_map[button]
            if state == 1:  # Press
                button_press_start[button] = timestamp_ms
            elif state == 0 and button in button_press_start:  # Release
                start_time = button_press_start[button]
                duration = timestamp_ms - start_time
                # Draw rectangle for press duration
                rect = mpatches.Rectangle((start_time, y - 0.4), duration, 0.8,
                                         facecolor=colors[button], alpha=0.6,
                                         edgecolor=colors[button], linewidth=1)
                ax_pad.add_patch(rect)
                del button_press_start[button]
    
    # Draw any buttons still pressed at end
    if pad_events:
        last_time = pad_events[-1][0]
        for button, start_time in button_press_start.items():
            y = button_map[button]
            duration = last_time - start_time
            rect = mpatches.Rectangle((start_time, y - 0.4), duration, 0.8,
                                     facecolor=colors[button], alpha=0.6,
                                     edgecolor=colors[button], linewidth=1)
            ax_pad.add_patch(rect)
    
    ax_pad.set_yticks(range(4))
    ax_pad.set_yticklabels(button_labels)
    ax_pad.set_ylabel('Gamepad')
    ax_pad.set_title('Button Press Intervals')
    ax_pad.grid(True, alpha=0.3)
    ax_pad.set_ylim(-0.5, 3.5)
    
    # Subplot 1: Accelerometer X
    ax_accel_x = axes[1]
    ax_accel_x.plot(accel_data['t'], accel_data['x'], 'b-', linewidth=0.5, alpha=0.7)
    ax_accel_x.set_ylabel('Accel X (m/s²)')
    ax_accel_x.set_title('Accelerometer X-axis')
    ax_accel_x.grid(True, alpha=0.3)
    
    # Subplot 2: Accelerometer Y
    ax_accel_y = axes[2]
    ax_accel_y.plot(accel_data['t'], accel_data['y'], 'g-', linewidth=0.5, alpha=0.7)
    ax_accel_y.set_ylabel('Accel Y (m/s²)')
    ax_accel_y.set_title('Accelerometer Y-axis')
    ax_accel_y.grid(True, alpha=0.3)
    
    # Subplot 3: Accelerometer Z
    ax_accel_z = axes[3]
    ax_accel_z.plot(accel_data['t'], accel_data['z'], 'r-', linewidth=0.5, alpha=0.7)
    ax_accel_z.set_ylabel('Accel Z (m/s²)')
    ax_accel_z.set_title('Accelerometer Z-axis')
    ax_accel_z.grid(True, alpha=0.3)
    
    # Subplot 4: Gyroscope X
    ax_gyro_x = axes[4]
    ax_gyro_x.plot(gyro_data['t'], gyro_data['x'], 'c-', linewidth=0.5, alpha=0.7)
    ax_gyro_x.set_ylabel('Gyro X (rad/s)')
    ax_gyro_x.set_title('Gyroscope X-axis')
    ax_gyro_x.grid(True, alpha=0.3)
    
    # Subplot 5: Gyroscope Y
    ax_gyro_y = axes[5]
    ax_gyro_y.plot(gyro_data['t'], gyro_data['y'], 'm-', linewidth=0.5, alpha=0.7)
    ax_gyro_y.set_ylabel('Gyro Y (rad/s)')
    ax_gyro_y.set_title('Gyroscope Y-axis')
    ax_gyro_y.grid(True, alpha=0.3)
    
    # Subplot 6: Gyroscope Z
    ax_gyro_z = axes[6]
    ax_gyro_z.plot(gyro_data['t'], gyro_data['z'], 'y-', linewidth=0.5, alpha=0.7)
    ax_gyro_z.set_ylabel('Gyro Z (rad/s)')
    ax_gyro_z.set_xlabel('Time (ms)')
    ax_gyro_z.set_title('Gyroscope Z-axis')
    ax_gyro_z.grid(True, alpha=0.3)
    
    # Add info text
    info_text = (
        f"Pad start: {pad_start_dt.strftime('%H:%M:%S')}\n"
        f"Sensor start: {sensor_start_dt.strftime('%H:%M:%S')}\n"
        f"Pad events: {len(pad_events)}\n"
        f"Accel samples: {len(accel_data['t'])}\n"
        f"Gyro samples: {len(gyro_data['t'])}"
    )
    fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


def main():
    # Find files in current directory
    csv_files = list(Path('.').glob('*_pad_record.csv'))
    ddrbin_files = list(Path('.').glob('sensor_data_*.ddrbin'))
    
    if not csv_files:
        print("ERROR: No pad record CSV file found")
        return
    
    if not ddrbin_files:
        print("ERROR: No sensor .ddrbin file found")
        return
    
    # Use the first matching files
    csv_path = csv_files[0]
    ddrbin_path = ddrbin_files[0]
    
    print(f"Visualizing:")
    print(f"  Gamepad: {csv_path}")
    print(f"  Sensors: {ddrbin_path}")
    print()
    
    plot_session(csv_path, ddrbin_path)


if __name__ == '__main__':
    main()
