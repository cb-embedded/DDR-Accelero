# Makefile for C Gamepad Recorders (Windows)

CC = gcc
CFLAGS = -Wall -O2

# DirectInput version
pad_recorder_dinput.exe: pad_recorder_dinput.c
	$(CC) $(CFLAGS) pad_recorder_dinput.c -o pad_recorder_dinput.exe -ldinput8 -ldxguid -lole32 -loleaut32

# XInput version  
pad_recorder_xinput.exe: pad_recorder_xinput.c
	$(CC) $(CFLAGS) pad_recorder_xinput.c -o pad_recorder_xinput.exe -lxinput

# Raw HID version
pad_recorder_rawhid.exe: pad_recorder_rawhid.c
	$(CC) $(CFLAGS) pad_recorder_rawhid.c -o pad_recorder_rawhid.exe -lhid -lsetupapi

all: pad_recorder_dinput.exe pad_recorder_xinput.exe pad_recorder_rawhid.exe

clean:
	del /Q *.exe *.o *.csv 2>nul || true

.PHONY: all clean
