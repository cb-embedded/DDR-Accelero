#!/usr/bin/env python3
"""
Debug script to inspect raw .ddrbin data
"""

import struct
from pathlib import Path


def debug_ddrbin(ddrbin_path):
    """Read and display detailed info from .ddrbin file"""
    
    print("=" * 70)
    print(f"DEBUG: {ddrbin_path}")
    print("=" * 70)
    
    with open(ddrbin_path, 'rb') as f:
        # Read header (32 bytes)
        header = f.read(32)
        
        print("\n--- HEADER (32 bytes) ---")
        print(f"Raw bytes: {header.hex()}")
        
        magic = struct.unpack('<I', header[0:4])[0]
        version = struct.unpack('<B', header[4:5])[0]
        start_time_ns = struct.unpack('<Q', header[5:13])[0]
        
        print(f"Magic:        0x{magic:08X} ({'DDAC' if magic == 0x44444143 else 'INVALID'})")
        print(f"Version:      {version}")
        print(f"Start time:   {start_time_ns} ns")
        print(f"              {start_time_ns / 1e9:.3f} seconds since epoch")
        
        # Read and display first 20 records
        print("\n--- DATA RECORDS (first 20) ---")
        print(f"{'#':<4} {'Type':<6} {'Timestamp (ns)':<20} {'Rel (ms)':<12} {'X':<10} {'Y':<10} {'Z':<10}")
        print("-" * 90)
        
        record_count = 0
        accel_count = 0
        gyro_count = 0
        
        for i in range(20):
            record = f.read(21)
            if len(record) < 21:
                print(f"\nEnd of file at record {i}")
                break
            
            sensor_type = struct.unpack('<B', record[0:1])[0]
            timestamp_ns = struct.unpack('<Q', record[1:9])[0]
            x = struct.unpack('<f', record[9:13])[0]
            y = struct.unpack('<f', record[13:17])[0]
            z = struct.unpack('<f', record[17:21])[0]
            
            rel_ms = (timestamp_ns - start_time_ns) / 1e6
            
            type_name = "accel" if sensor_type == 1 else ("gyro" if sensor_type == 2 else f"?{sensor_type}")
            
            print(f"{i:<4} {type_name:<6} {timestamp_ns:<20} {rel_ms:<12.3f} {x:<10.4f} {y:<10.4f} {z:<10.4f}")
            
            if sensor_type == 1:
                accel_count += 1
            elif sensor_type == 2:
                gyro_count += 1
        
        # Count all records
        print("\n--- COUNTING ALL RECORDS ---")
        total_accel = accel_count
        total_gyro = gyro_count
        
        while True:
            record = f.read(21)
            if len(record) < 21:
                break
            
            sensor_type = struct.unpack('<B', record[0:1])[0]
            if sensor_type == 1:
                total_accel += 1
            elif sensor_type == 2:
                total_gyro += 1
        
        print(f"Total accelerometer records: {total_accel}")
        print(f"Total gyroscope records:     {total_gyro}")
        print(f"Total records:               {total_accel + total_gyro}")
        
        # File size check
        file_size = Path(ddrbin_path).stat().st_size
        expected_records = (file_size - 32) // 21
        print(f"\nFile size:                   {file_size} bytes")
        print(f"Expected records (by size):  {expected_records}")
        print(f"Actual records:              {total_accel + total_gyro}")


def main():
    ddrbin_files = list(Path('.').glob('sensor_data_*.ddrbin'))
    
    if not ddrbin_files:
        print("ERROR: No .ddrbin file found")
        return
    
    for ddrbin_file in ddrbin_files:
        debug_ddrbin(ddrbin_file)


if __name__ == '__main__':
    main()
