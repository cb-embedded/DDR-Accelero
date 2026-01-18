/*
 * Raw HID Gamepad Recorder for Windows
 * Captures gamepad button presses using Raw Input HID API
 * Compile: gcc pad_recorder_rawhid.c -o pad_recorder_rawhid -lhid -lsetupapi
 */

#include <windows.h>
#include <hidsdi.h>
#include <setupapi.h>
#include <stdio.h>
#include <time.h>

#pragma comment(lib, "hid.lib")
#pragma comment(lib, "setupapi.lib")

static FILE* g_logfile = NULL;
static clock_t g_start_time;
static BYTE g_prev_buttons[128] = {0};
static const char* g_btn_names[] = {"BTN0", "BTN1", "BTN2", "BTN3"};

void LogButtonEvent(int btn_idx, BOOL pressed) {
    double elapsed = (double)(clock() - g_start_time) / CLOCKS_PER_SEC;
    if (btn_idx < 4) {
        printf("%.3fs %s %s\n", elapsed, g_btn_names[btn_idx], pressed ? "DOWN" : "UP");
        fprintf(g_logfile, "%.3f,%s,%d\n", elapsed, g_btn_names[btn_idx], pressed ? 1 : 0);
    } else {
        printf("%.3fs BTN%d %s\n", elapsed, btn_idx, pressed ? "DOWN" : "UP");
        fprintf(g_logfile, "%.3f,BTN%d,%d\n", elapsed, btn_idx, pressed ? 1 : 0);
    }
    fflush(g_logfile);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_INPUT: {
            UINT dwSize;
            LPBYTE lpb;
            RAWINPUT* raw;
            PHIDP_PREPARSED_DATA preparsedData;
            HIDP_CAPS caps;
            HIDP_BUTTON_CAPS buttonCaps[20];
            USHORT capsLength;
            USAGE usages[128];
            ULONG usageLength;
            int i;
            
            GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &dwSize, sizeof(RAWINPUTHEADER));
            lpb = (LPBYTE)malloc(dwSize);
            if (lpb == NULL) {
                fclose(g_logfile);
                return 0;
            }
            
            if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) != dwSize) {
                free(lpb);
                return 0;
            }
            
            raw = (RAWINPUT*)lpb;
            
            if (raw->header.dwType == RIM_TYPEHID) {
                GetRawInputDeviceInfo(raw->header.hDevice, RIDI_PREPARSEDDATA, NULL, &dwSize);
                preparsedData = (PHIDP_PREPARSED_DATA)malloc(dwSize);
                if (preparsedData == NULL) {
                    free(lpb);
                    return 0;
                }
                GetRawInputDeviceInfo(raw->header.hDevice, RIDI_PREPARSEDDATA, preparsedData, &dwSize);
                
                HidP_GetCaps(preparsedData, &caps);
                
                capsLength = caps.NumberInputButtonCaps;
                HidP_GetButtonCaps(HidP_Input, buttonCaps, &capsLength, preparsedData);
                
                if (capsLength > 0) {
                    usageLength = buttonCaps[0].Range.UsageMax - buttonCaps[0].Range.UsageMin + 1;
                    if (usageLength > 128) usageLength = 128;
                    
                    NTSTATUS status = HidP_GetUsages(HidP_Input, buttonCaps[0].UsagePage, 0,
                                                     usages, &usageLength, preparsedData,
                                                     (PCHAR)raw->data.hid.bRawData,
                                                     raw->data.hid.dwSizeHid);
                    
                    if (status == HIDP_STATUS_SUCCESS) {
                        BYTE current_buttons[128] = {0};
                        
                        for (i = 0; i < (int)usageLength; i++) {
                            int btn_idx = usages[i] - buttonCaps[0].Range.UsageMin;
                            if (btn_idx >= 0 && btn_idx < 128) {
                                current_buttons[btn_idx] = 1;
                            }
                        }
                        
                        for (i = 0; i < 128; i++) {
                            if (current_buttons[i] != g_prev_buttons[i]) {
                                LogButtonEvent(i, current_buttons[i]);
                                g_prev_buttons[i] = current_buttons[i];
                            }
                        }
                    }
                }
                
                free(preparsedData);
            }
            
            free(lpb);
            return 0;
        }
        
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
    }
    
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

BOOL RegisterRawInput(HWND hwnd) {
    RAWINPUTDEVICE rid[1];
    
    rid[0].usUsagePage = 0x01;
    rid[0].usUsage = 0x05;
    rid[0].dwFlags = RIDEV_INPUTSINK;
    rid[0].hwndTarget = hwnd;
    
    if (!RegisterRawInputDevices(rid, 1, sizeof(rid[0]))) {
        printf("Failed to register raw input device.\n");
        return FALSE;
    }
    
    return TRUE;
}

int main() {
    WNDCLASSEX wc = {0};
    HWND hwnd;
    MSG msg;
    char filename[256];
    time_t now;
    struct tm* timeinfo;
    
    time(&now);
    timeinfo = localtime(&now);
    strftime(filename, sizeof(filename), "%Y_%m_%d_%H_%M_%S_pad_record_rawhid.csv", timeinfo);
    
    g_logfile = fopen(filename, "w");
    if (!g_logfile) {
        printf("Failed to create log file.\n");
        return 1;
    }
    fprintf(g_logfile, "timestamp,button,pressed\n");
    
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "RawHIDRecorder";
    
    if (!RegisterClassEx(&wc)) {
        printf("Failed to register window class.\n");
        fclose(g_logfile);
        return 1;
    }
    
    hwnd = CreateWindowEx(0, "RawHIDRecorder", "HID Recorder", 0,
                          0, 0, 0, 0, HWND_MESSAGE, NULL, 
                          GetModuleHandle(NULL), NULL);
    
    if (!hwnd) {
        printf("Failed to create window.\n");
        fclose(g_logfile);
        return 1;
    }
    
    if (!RegisterRawInput(hwnd)) {
        DestroyWindow(hwnd);
        fclose(g_logfile);
        return 1;
    }
    
    printf("Raw HID Gamepad Recorder\n");
    printf("Recording to: %s\n", filename);
    printf("Press Ctrl+C to stop.\n\n");
    printf("Listening for HID gamepad events...\n");
    
    g_start_time = clock();
    
    while (GetMessage(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    fclose(g_logfile);
    
    return 0;
}
