cmake_minimum_required(VERSION 3.10) # Specify minimum CMake version
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(C_LSTM_PIM) # Set the project name

# Add executable or library targets
add_executable(dev dev_src/main.c) # Add an executable target

# Set include directories
target_include_directories(dev PUBLIC include)

add_custom_target(docker
    COMMAND docker run --rm -v /home/patrickli/Space/lstm_c_pim:/app -w /app johnramsden/upmem bash
)