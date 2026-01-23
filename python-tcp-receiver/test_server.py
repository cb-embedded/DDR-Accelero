#!/usr/bin/env python3
"""
Simple TCP test server that receives and prints sensor data.
Useful for testing without the GUI.
"""

import socket
import sys

def main():
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print(f"Starting TCP test server on port {port}...")
    print("Waiting for connection...")
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen(1)
    
    try:
        client_socket, client_address = server_socket.accept()
        print(f"Connected to {client_address[0]}:{client_address[1]}")
        print("\nReceiving data (showing first 20 samples)...")
        print("Format: timestamp_ns,timestamp_ms,sensor_type,x,y,z")
        print("-" * 80)
        
        buffer = ""
        sample_count = 0
        accel_count = 0
        gyro_count = 0
        
        while True:
            data = client_socket.recv(4096)
            if not data:
                break
            
            buffer += data.decode('utf-8', errors='ignore')
            
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                
                if line:
                    if sample_count < 20:
                        print(line)
                    
                    sample_count += 1
                    if 'accel' in line:
                        accel_count += 1
                    elif 'gyro' in line:
                        gyro_count += 1
                    
                    if sample_count % 100 == 0:
                        print(f"\rTotal samples: {sample_count} (Accel: {accel_count}, Gyro: {gyro_count})", end='', flush=True)
        
        print(f"\n\nConnection closed.")
        print(f"Final count: {sample_count} samples (Accel: {accel_count}, Gyro: {gyro_count})")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        server_socket.close()

if __name__ == '__main__':
    main()
