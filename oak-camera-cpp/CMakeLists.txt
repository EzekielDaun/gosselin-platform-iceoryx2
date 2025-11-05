cmake_minimum_required(VERSION 3.20)

set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX} ${CMAKE_PREFIX_PATH})

# Create a project with name 'oak-camera-app'
set(TARGET_NAME oak-camera-app)
project(${TARGET_NAME})

# Depthai dependency
find_package(depthai REQUIRED)
message(STATUS "Found depthai: ${depthai_DIR}")

# Dependencies (optional, only used for example)
find_package(OpenCV REQUIRED)

# IceOryx2 dependency
# set(CMAKE_PREFIX_PATH "/Users/yifeiliu/Documents/gosselin-platform-iceoryx2/.pixi/envs/oak-camera" ${CMAKE_PREFIX_PATH})
# find_package(iceoryx2 REQUIRED)
find_package(iceoryx2-cxx CONFIG REQUIRED)
message(STATUS "Found iceoryx2-cxx: ${iceoryx2-cxx_DIR}")

# Add source files
add_executable("${TARGET_NAME}"
src/main.cpp
)

# Link with libraries
target_link_libraries(${TARGET_NAME}
    PUBLIC
        depthai::core
        ${OpenCV_LIBS} # optional, used for example
        iceoryx2-cxx::static-lib-cxx
)

# Set some errors
if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
endif()

# Set compiler features (c++17)
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)

# Windows - Add runtime dependencies
if(WIN32)
    if(CMAKE_VERSION VERSION_LESS "3.21")
        message(WARNING "CMake version less than 3.21 - automatic DLL handling not available. Make sure to copy required DLLs to the same folder as .exe")
    else()
        # TARGET_RUNTIME_DLLS generator expression available since CMake 3.21
        set(depthai_dll_libraries "$<TARGET_RUNTIME_DLLS:${TARGET_NAME}>")
        # Copy the required dlls
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND
            ${CMAKE_COMMAND} -E copy_if_different ${depthai_dll_libraries} $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()