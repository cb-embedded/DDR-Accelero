/*
 * XInput Gamepad Recorder for Windows
 * Captures gamepad button presses using XInput API (Xbox controllers)
 * Compile: gcc pad_recorder_xinput.c -o pad_recorder_xinput -lxinput
 */

#include <windows.h>
#include <xinput.h>
#include <stdio.h>
#include <time.h>

static FILE* g_logfile = NULL;
static clock_t g_start_time;
static BOOL g_button_states[4] = {FALSE, FALSE, FALSE, FALSE};
static const char* g_btn_names[] = {"LEFT", "RIGHT", "UP", "DOWN"};

void LogButtonEvent(int btn_idx, BOOL pressed) {
    double elapsed = (double)(clock() - g_start_time) / CLOCKS_PER_SEC;
    g_button_states[btn_idx] = pressed;
    printf("%.3fs %s %s\n", elapsed, g_btn_names[btn_idx], pressed ? "DOWN" : "UP");
    fprintf(g_logfile, "%.3f,%s,%d\n", elapsed, g_btn_names[btn_idx], pressed ? 1 : 0);
    fflush(g_logfile);
}

BOOL FindController(DWORD* pUserIndex) {
    XINPUT_STATE state;
    DWORD i;
    
    for (i = 0; i < XUSER_MAX_COUNT; i++) {
        ZeroMemory(&state, sizeof(XINPUT_STATE));
        if (XInputGetState(i, &state) == ERROR_SUCCESS) {
            *pUserIndex = i;
            return TRUE;
        }
    }
    return FALSE;
}

void PollGamepad(DWORD userIndex) {
    XINPUT_STATE state;
    WORD buttons;
    static WORD prev_buttons = 0;
    
    ZeroMemory(&state, sizeof(XINPUT_STATE));
    
    if (XInputGetState(userIndex, &state) != ERROR_SUCCESS) {
        return;
    }
    
    buttons = state.Gamepad.wButtons;
    
    if ((buttons & XINPUT_GAMEPAD_DPAD_LEFT) != (prev_buttons & XINPUT_GAMEPAD_DPAD_LEFT)) {
        LogButtonEvent(0, (buttons & XINPUT_GAMEPAD_DPAD_LEFT) != 0);
    }
    if ((buttons & XINPUT_GAMEPAD_DPAD_RIGHT) != (prev_buttons & XINPUT_GAMEPAD_DPAD_RIGHT)) {
        LogButtonEvent(1, (buttons & XINPUT_GAMEPAD_DPAD_RIGHT) != 0);
    }
    if ((buttons & XINPUT_GAMEPAD_DPAD_UP) != (prev_buttons & XINPUT_GAMEPAD_DPAD_UP)) {
        LogButtonEvent(2, (buttons & XINPUT_GAMEPAD_DPAD_UP) != 0);
    }
    if ((buttons & XINPUT_GAMEPAD_DPAD_DOWN) != (prev_buttons & XINPUT_GAMEPAD_DPAD_DOWN)) {
        LogButtonEvent(3, (buttons & XINPUT_GAMEPAD_DPAD_DOWN) != 0);
    }
    
    prev_buttons = buttons;
}

int main() {
    DWORD userIndex;
    char filename[256];
    time_t now;
    struct tm* timeinfo;
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(filename, sizeof(filename), "%Y_%m_%d_%H_%M_%S_pad_record_xinput.csv", timeinfo);
    
    g_logfile = fopen(filename, "w");
    if (!g_logfile) {
        printf("Failed to create log file.\n");
        return 1;
    }
    fprintf(g_logfile, "timestamp,button,pressed\n");
    
    if (!FindController(&userIndex)) {
        printf("No XInput controller found.\n");
        printf("XInput only supports Xbox 360/One controllers.\n");
        fclose(g_logfile);
        return 1;
    }
    
    printf("XInput Gamepad Recorder\n");
    printf("Controller found at index: %lu\n", userIndex);
    printf("Recording to: %s\n", filename);
    printf("Press Ctrl+C to stop.\n\n");
    
    g_start_time = clock();
    
    while (1) {
        PollGamepad(userIndex);
        Sleep(10);
    }
    
    fclose(g_logfile);
    
    return 0;
}
