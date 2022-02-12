# Sets the minimum version of CMake required to build your native library.
# This ensures that a certain set of CMake features is available to
# your build.

cmake_minimum_required(VERSION 3.4.1)

set(SRC
        TensorflowRunner.cc
        TensorflowRunnerJNI.cpp
        tflite_gpu_runner.cc
        tflite_model_loader.cc
     )

add_library( native-lib
             SHARED
              ${SRC} )

set(INCLUDES
        ${CMAKE_SOURCE_DIR}/../includes/
        .)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER_INIT g++)

target_include_directories (native-lib PUBLIC ${INCLUDES})

set(IMPORT_DIR ${CMAKE_SOURCE_DIR}/../libs)

# shared
add_library(tflite SHARED IMPORTED)
set_target_properties(tflite
        PROPERTIES IMPORTED_LOCATION
        ${IMPORT_DIR}/${ANDROID_ABI}/libtensorflowlite.so)

add_library(tfgpudelegate SHARED IMPORTED)
set_target_properties(tfgpudelegate
        PROPERTIES IMPORTED_LOCATION
        ${IMPORT_DIR}/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so)

add_definitions(-DTF_LITE_DISABLE_X86_NEON)

target_link_libraries(native-lib tflite tfgpudelegate -landroid -llog -lEGL -lGLESv2)
