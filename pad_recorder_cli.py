#!/usr/bin/env python3
"""
CLI Gamepad Recorder for DDR dance pad input capture.
Can run in background and captures events even when the terminal loses focus.
"""

import pygame
import time
import sys
import os
import argparse
from datetime import datetime

class GamepadRecorder:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.recording = False
        self.log = []
        self.start_time = None
        self.absolute_start_time = None
        self.button_states = [False, False, False, False]
        self.btn_names = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        self.joystick = None
        
    def initialize_gamepad(self):
        """Initialize the first available gamepad."""
        if pygame.joystick.get_count() == 0:
            print("No gamepad/joystick detected. Please connect a gamepad.")
            return False
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad connected: {self.joystick.get_name()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        return True
    
    def get_timestamp(self):
        """Get elapsed time since recording started."""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def add_event(self, button, pressed):
        """Add a button event to the log."""
        if not self.recording:
            return
        
        self.log.append({
            'button': self.btn_names[button],
            'pressed': pressed,
            'timestamp': f"{self.get_timestamp():.3f}"
        })
    
    def poll_gamepad(self):
        """Poll gamepad state and detect button changes."""
        if self.joystick is None:
            return False
        
        pygame.event.pump()
        
        for i in range(min(4, self.joystick.get_numbuttons())):
            pressed = self.joystick.get_button(i)
            
            if pressed != self.button_states[i]:
                self.button_states[i] = pressed
                
                if pressed:
                    print(f"  [{self.btn_names[i]}] PRESSED")
                    self.add_event(i, True)
                else:
                    print(f"  [{self.btn_names[i]}] RELEASED")
                    self.add_event(i, False)
        
        return True
    
    def start_recording(self):
        """Start recording gamepad events."""
        self.recording = True
        self.log = []
        self.start_time = time.time()
        self.absolute_start_time = datetime.now()
        self.button_states = [False, False, False, False]
        print("\n=== RECORDING STARTED ===")
        print("Press buttons on your gamepad. Events will be captured.")
    
    def stop_recording(self):
        """Stop recording gamepad events."""
        self.recording = False
        print("\n=== RECORDING STOPPED ===")
        print(f"Total events captured: {len(self.log)}")
    
    def export_log(self, music_name, difficulty_number, difficulty_name):
        """Export the recorded log to a CSV file with YAML frontmatter."""
        if len(self.log) == 0:
            print("No events recorded. Nothing to export.")
            return None
        
        # Format difficulty string
        difficulty_str = ''
        if difficulty_number and difficulty_name:
            difficulty_str = f"{difficulty_number}_{difficulty_name}"
        elif difficulty_number:
            difficulty_str = str(difficulty_number)
        elif difficulty_name:
            difficulty_str = difficulty_name
        
        # Format timestamp for filename
        now = self.absolute_start_time or datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Format timestamp for YAML
        iso_timestamp = now.isoformat()
        absolute_offset_seconds = f"{self.absolute_start_time.timestamp():.3f}" if self.absolute_start_time else 'null'
        
        # Sanitize music name for filename
        sanitized_music_name = ''.join(c if c.isalnum() or c in '_-' else '_' for c in music_name)
        
        # Build filename
        filename = timestamp_str
        if sanitized_music_name:
            filename += f"_{sanitized_music_name}"
        if difficulty_str:
            filename += f"_({difficulty_str})"
        filename += '.csv'
        
        # Create YAML frontmatter
        # Format difficulty_number as string (matching web app behavior) or 'null'
        diff_num_str = str(difficulty_number) if difficulty_number is not None else 'null'
        
        yaml_header = f"""---
music_name: "{music_name}"
difficulty_number: {diff_num_str}
difficulty_name: {f'"{difficulty_name}"' if difficulty_name else 'null'}
recorded_at: "{iso_timestamp}"
absolute_start_time: "{iso_timestamp}"
absolute_offset_seconds: {absolute_offset_seconds}
event_count: {len(self.log)}
---
"""
        
        # Create CSV content
        csv_lines = ['timestamp,button,pressed']
        for event in self.log:
            csv_lines.append(f"{event['timestamp']},{event['button']},{1 if event['pressed'] else 0}")
        csv_content = '\n'.join(csv_lines)
        
        full_content = yaml_header + csv_content
        
        # Save to file
        output_path = os.path.join('raw_data', filename)
        os.makedirs('raw_data', exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(full_content)
        
        print(f"\nLog exported to: {output_path}")
        return output_path
    
    def run(self, music_name, difficulty_number, difficulty_name):
        """Main recording loop."""
        if not self.initialize_gamepad():
            return
        
        print("\n" + "="*50)
        print("  DDR GAMEPAD RECORDER (CLI)")
        print("="*50)
        print("\nThis application captures gamepad input in the background.")
        print("You can switch to other windows (like your DDR game) and")
        print("the events will continue to be recorded.\n")
        
        print("Metadata:")
        print(f"  Music: {music_name}")
        print(f"  Difficulty: {difficulty_number if difficulty_number else '-'} {difficulty_name or '-'}")
        
        input("\nPress Enter to start recording...")
        self.start_recording()
        
        print("Press Ctrl+C to stop recording and export.\n")
        
        # Main loop
        try:
            while True:
                # Poll gamepad
                if not self.poll_gamepad():
                    print("Gamepad disconnected!")
                    break
                
                # Small delay to avoid busy-waiting
                time.sleep(0.001)  # 1ms poll rate
        
        except KeyboardInterrupt:
            print("\n\nRecording stopped by user.")
        
        self.stop_recording()
        
        # Export log
        if len(self.log) > 0:
            export = input("\nExport log? (y/n): ").strip().lower()
            if export == 'y':
                self.export_log(music_name, difficulty_number, difficulty_name)
        
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(
        description='CLI Gamepad Recorder for DDR dance pad input capture.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --music "Lucky Orb" --difficulty-number 5 --difficulty-name Medium
  %(prog)s -m "My Song" -n 7 -d Hard
  %(prog)s --music "Test Song"
        """
    )
    
    parser.add_argument(
        '-m', '--music',
        dest='music_name',
        default='Unknown',
        help='Music name (default: Unknown)'
    )
    
    parser.add_argument(
        '-n', '--difficulty-number',
        dest='difficulty_number',
        type=int,
        default=None,
        help='Difficulty number (1-10)'
    )
    
    parser.add_argument(
        '-d', '--difficulty-name',
        dest='difficulty_name',
        default=None,
        choices=['Easy', 'Medium', 'Hard', 'Expert', 'Challenge'],
        help='Difficulty name'
    )
    
    args = parser.parse_args()
    
    recorder = GamepadRecorder()
    recorder.run(args.music_name, args.difficulty_number, args.difficulty_name)

if __name__ == '__main__':
    main()
