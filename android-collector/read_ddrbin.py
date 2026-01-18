#!/usr/bin/env python3
"""
DDR Binary Format Reader

Reads and parses .ddrbin sensor data files from the DDR Collector Android app.

File format:
- 32-byte header with magic number, version, and start timestamp
- Stream of 21-byte sensor data records (sensor type, timestamp, x, y, z)

Usage:
    python read_ddrbin.py <file.ddrbin>
"""

import struct
import sys
from pathlib import Path


def read_ddrbin(filename):
    """Read and parse a .ddrbin sensor data file"""
    
    filepath = Path(filename)
    if not filepath.exists():
        print(f"ERROR: File not found: {filename}")
        return
    
    with open(filename, 'rb') as f:
        # Read header (32 bytes)
        header = f.read(32)
        
        if len(header) < 32:
            print("ERROR: File too small to contain valid header")
            return
        
        # Parse header
        magic = struct.unpack('<I', header[0:4])[0]
        version = struct.unpack('<B', header[4:5])[0]
        timestamp_start = struct.unpack('<Q', header[5:13])[0]
        
        print("=" * 60)
        print("DDR Binary Format File")
        print("=" * 60)
        print(f"Magic number:    0x{magic:08X} (expected: 0x44444143 'DDAC')")
        print(f"Format version:  {version}")
        print(f"Start timestamp: {timestamp_start} ns")
        print(f"                 ({timestamp_start / 1e9:.3f} seconds since epoch)")
        print()
        
        if magic != 0x44444143:
            print("ERROR: Invalid magic number! This may not be a valid .ddrbin file.")
            return
        
        # Read data records
        accel_samples = []
        gyro_samples = []
        record_count = 0
        
        print("Sample Data (first 10 records):")
        print("-" * 60)
        
        while True:
            record = f.read(21)
            if len(record) < 21:
                break
            
            # Parse record
            sensor_type = struct.unpack('<B', record[0:1])[0]
            timestamp_ns = struct.unpack('<Q', record[1:9])[0]
            x = struct.unpack('<f', record[9:13])[0]
            y = struct.unpack('<f', record[13:17])[0]
            z = struct.unpack('<f', record[17:21])[0]
            
            record_count += 1
            
            if sensor_type == 1:
                accel_samples.append((timestamp_ns, x, y, z))
                sensor_name = "accel"
            elif sensor_type == 2:
                gyro_samples.append((timestamp_ns, x, y, z))
                sensor_name = "gyro "
            else:
                sensor_name = "unknown"
            
            # Print first few samples
            if record_count <= 10:
                print(f"{sensor_name} | t={timestamp_ns:20} | x={x:8.4f} y={y:8.4f} z={z:8.4f}")
        
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"Total records:          {record_count}")
        print(f"Accelerometer samples:  {len(accel_samples)}")
        print(f"Gyroscope samples:      {len(gyro_samples)}")
        
        if accel_samples:
            duration_accel = (accel_samples[-1][0] - accel_samples[0][0]) / 1e9
            rate_accel = len(accel_samples) / duration_accel if duration_accel > 0 else 0
            print(f"Accelerometer rate:     {rate_accel:.1f} Hz")
        
        if gyro_samples:
            duration_gyro = (gyro_samples[-1][0] - gyro_samples[0][0]) / 1e9
            rate_gyro = len(gyro_samples) / duration_gyro if duration_gyro > 0 else 0
            print(f"Gyroscope rate:         {rate_gyro:.1f} Hz")
        
        file_size = filepath.stat().st_size
        print(f"File size:              {file_size} bytes ({file_size / 1024:.1f} KB)")
        print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python read_ddrbin.py <file.ddrbin>")
        print()
        print("Example:")
        print("    python read_ddrbin.py sensor_data_2026-01-18_12-34-56.ddrbin")
        sys.exit(1)
    
    read_ddrbin(sys.argv[1])


if __name__ == '__main__':
    main()
