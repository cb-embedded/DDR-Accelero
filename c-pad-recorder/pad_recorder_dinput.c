/*
 * DirectInput Gamepad Recorder for Windows
 * Captures gamepad button presses using DirectInput API
 * Compile: gcc pad_recorder_dinput.c -o pad_recorder_dinput -ldinput8 -ldxguid -lole32 -loleaut32
 */

#define DIRECTINPUT_VERSION 0x0800
#include <windows.h>
#include <dinput.h>
#include <stdio.h>
#include <time.h>

#define SAFE_RELEASE(p) { if(p) { (p)->lpVtbl->Release(p); (p)=NULL; } }

static LPDIRECTINPUT8 g_pDI = NULL;
static LPDIRECTINPUTDEVICE8 g_pJoystick = NULL;
static FILE* g_logfile = NULL;
static clock_t g_start_time;
static BOOL g_button_states[4] = {FALSE, FALSE, FALSE, FALSE};
static const char* g_btn_names[] = {"LEFT", "RIGHT", "UP", "DOWN"};

BOOL CALLBACK EnumJoysticksCallback(const DIDEVICEINSTANCE* pdidInstance, VOID* pContext) {
    HRESULT hr;
    
    hr = IDirectInput8_CreateDevice(g_pDI, &pdidInstance->guidInstance, &g_pJoystick, NULL);
    if (FAILED(hr)) {
        return DIENUM_CONTINUE;
    }
    
    printf("Found gamepad: %s\n", pdidInstance->tszProductName);
    return DIENUM_STOP;
}

BOOL InitDirectInput(HWND hWnd) {
    HRESULT hr;
    
    hr = DirectInput8Create(GetModuleHandle(NULL), DIRECTINPUT_VERSION, 
                            &IID_IDirectInput8, (VOID**)&g_pDI, NULL);
    if (FAILED(hr)) {
        printf("Failed to create DirectInput8: 0x%lx\n", hr);
        return FALSE;
    }
    
    hr = IDirectInput8_EnumDevices(g_pDI, DI8DEVCLASS_GAMECTRL, 
                                    EnumJoysticksCallback, NULL, DIEDFL_ATTACHEDONLY);
    if (FAILED(hr)) {
        printf("Failed to enumerate devices: 0x%lx\n", hr);
        return FALSE;
    }
    
    if (g_pJoystick == NULL) {
        printf("No gamepad found.\n");
        return FALSE;
    }
    
    hr = IDirectInputDevice8_SetDataFormat(g_pJoystick, &c_dfDIJoystick2);
    if (FAILED(hr)) {
        printf("Failed to set data format: 0x%lx\n", hr);
        return FALSE;
    }
    
    hr = IDirectInputDevice8_SetCooperativeLevel(g_pJoystick, hWnd, 
                                                   DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);
    if (FAILED(hr)) {
        printf("Failed to set cooperative level: 0x%lx\n", hr);
        return FALSE;
    }
    
    hr = IDirectInputDevice8_Acquire(g_pJoystick);
    if (FAILED(hr)) {
        printf("Failed to acquire device: 0x%lx\n", hr);
        return FALSE;
    }
    
    return TRUE;
}

void LogButtonEvent(int btn_idx, BOOL pressed) {
    double elapsed = (double)(clock() - g_start_time) / CLOCKS_PER_SEC;
    g_button_states[btn_idx] = pressed;
    printf("%.3fs %s %s\n", elapsed, g_btn_names[btn_idx], pressed ? "DOWN" : "UP");
    fprintf(g_logfile, "%.3f,%s,%d\n", elapsed, g_btn_names[btn_idx], pressed ? 1 : 0);
    fflush(g_logfile);
}

void PollGamepad() {
    DIJOYSTATE2 js;
    HRESULT hr;
    int i;
    
    hr = IDirectInputDevice8_Poll(g_pJoystick);
    if (FAILED(hr)) {
        hr = IDirectInputDevice8_Acquire(g_pJoystick);
        return;
    }
    
    hr = IDirectInputDevice8_GetDeviceState(g_pJoystick, sizeof(DIJOYSTATE2), &js);
    if (FAILED(hr)) {
        return;
    }
    
    for (i = 0; i < 4; i++) {
        BOOL pressed = (js.rgbButtons[i] & 0x80) != 0;
        if (pressed != g_button_states[i]) {
            LogButtonEvent(i, pressed);
        }
    }
}

int main() {
    HWND hWnd;
    char filename[256];
    time_t now;
    struct tm* timeinfo;
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(filename, sizeof(filename), "%Y_%m_%d_%H_%M_%S_pad_record_dinput.csv", timeinfo);
    
    g_logfile = fopen(filename, "w");
    if (!g_logfile) {
        printf("Failed to create log file.\n");
        return 1;
    }
    fprintf(g_logfile, "timestamp,button,pressed\n");
    
    hWnd = GetConsoleWindow();
    if (!InitDirectInput(hWnd)) {
        fclose(g_logfile);
        return 1;
    }
    
    printf("DirectInput Gamepad Recorder\n");
    printf("Recording to: %s\n", filename);
    printf("Press Ctrl+C to stop.\n\n");
    
    g_start_time = clock();
    
    while (1) {
        PollGamepad();
        Sleep(10);
    }
    
    SAFE_RELEASE(g_pJoystick);
    SAFE_RELEASE(g_pDI);
    fclose(g_logfile);
    
    return 0;
}
