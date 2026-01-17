#!/usr/bin/env python3
"""
CLI Gamepad Recorder for DDR dance pad input capture.
Can run in background and captures events even when the terminal loses focus.
"""

import pygame
import time
from datetime import datetime

class GamepadRecorder:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        now = datetime.now()
        self.log_file = now.strftime("%Y_%m_%d_%H_%M_%S_pad_record.csv")
        self.joystick = None
        self.button_states = [False, False, False, False]
        self.btn_names = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        self.start_time = time.time()
        # Write CSV header
        with open(self.log_file, 'w') as f:
            f.write('timestamp,button,pressed\n')

    def initialize_gamepad(self):
        """Initialize the first available gamepad."""
        if pygame.joystick.get_count() == 0:
            print("No gamepad/joystick detected.")
            return False
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad: {self.joystick.get_name()}")
        return True

    def handle_button_event(self, btn_idx, pressed):
        self.button_states[btn_idx] = pressed
        ts = f"{time.time() - self.start_time:.3f}"
        btn = self.btn_names[btn_idx]
        print(f"{ts}s {btn} {'DOWN' if pressed else 'UP'}")
        with open(self.log_file, 'a') as f:
            f.write(f"{ts},{btn},{1 if pressed else 0}\n")
        return True

    def run(self):
        """Main recording loop using event-driven logic."""
        if not self.initialize_gamepad():
            return
        try:
            while True:
                event = pygame.event.wait()  # Blocks until an event is available
                if event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
                    btn_idx = event.button
                    if btn_idx < 4:
                        pressed = event.type == pygame.JOYBUTTONDOWN
                        self.handle_button_event(btn_idx, pressed)
        except KeyboardInterrupt:
            pass
        pygame.quit()

def main():
    recorder = GamepadRecorder()
    recorder.run()

if __name__ == '__main__':
    main()
