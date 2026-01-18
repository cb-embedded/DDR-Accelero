cmake_minimum_required(VERSION 3.10)
project(pad_recorder C)

set(CMAKE_C_STANDARD 11)

add_executable(pad_recorder_dinput pad_recorder_dinput.c)

target_link_libraries(pad_recorder_dinput 
    dinput8 
    dxguid 
    ole32 
    oleaut32
)

target_compile_options(pad_recorder_dinput PRIVATE -Wall -O2)
