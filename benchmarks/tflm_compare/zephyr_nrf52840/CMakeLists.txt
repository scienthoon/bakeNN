cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(bakenn_tflm_comparison C CXX)

set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)
target_sources(app PRIVATE src/main.cpp src/model_data.cc)

# Zephyr's TFLM module ships the reference kernels but does not expose the
# TFLM `OPTIMIZED_KERNEL_DIR=cmsis_nn` switch.  Keep the default build
# reference-only and make CMSIS-NN an explicit, reproducible opt-in.
#
# BAKENN_CMSIS_NN_ROOT must point at a pinned CMSIS-NN checkout.  The optional
# BAKENN_CMSIS_CORE_ROOT points at CMSIS/Core/Include (the Zephyr CMSIS module
# is a convenient source on Arm boards).  All CMSIS-NN sources are compiled
# into this image with section GC enabled by the target toolchain; unused
# kernels therefore do not become part of the final ELF.
set(BAKENN_CMSIS_NN_ROOT "" CACHE PATH "Pinned CMSIS-NN checkout")
set(BAKENN_CMSIS_CORE_ROOT "" CACHE PATH "CMSIS Core/Include directory")
set(BAKENN_TFLM_ROOT "" CACHE PATH "TFLM checkout containing the CMSIS-NN wrapper")
option(BAKENN_USE_CMSIS_NN "Use TFLM CMSIS-NN wrappers and kernels" OFF)

if(BAKENN_USE_CMSIS_NN)
  if(NOT BAKENN_CMSIS_NN_ROOT OR NOT EXISTS
     "${BAKENN_CMSIS_NN_ROOT}/Include/arm_nnfunctions.h")
    message(FATAL_ERROR
      "BAKENN_USE_CMSIS_NN requires BAKENN_CMSIS_NN_ROOT with Include/arm_nnfunctions.h")
  endif()
  if(NOT BAKENN_CMSIS_CORE_ROOT OR NOT EXISTS
     "${BAKENN_CMSIS_CORE_ROOT}/cmsis_compiler.h")
    message(FATAL_ERROR
      "BAKENN_USE_CMSIS_NN requires BAKENN_CMSIS_CORE_ROOT/CMSIS/Core/Include")
  endif()
  if(NOT BAKENN_TFLM_ROOT OR NOT EXISTS
     "${BAKENN_TFLM_ROOT}/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc")
    message(FATAL_ERROR
      "BAKENN_USE_CMSIS_NN requires BAKENN_TFLM_ROOT with the cmsis_nn wrapper")
  endif()
  if(NOT EXISTS
     "${BAKENN_TFLM_ROOT}/tensorflow/lite/micro/kernels/cmsis_nn/conv.cc")
    message(FATAL_ERROR
      "BAKENN_USE_CMSIS_NN requires the TFLM cmsis_nn Conv2D wrapper")
  endif()

  # The pinned TFLM wrapper uses the historical CMSIS/NN/Include path.  Keep
  # that compatibility path in the build tree instead of patching vendored
  # TFLM sources or relying on a user's global include tree.
  set(BAKENN_CMSIS_COMPAT_ROOT
      "${CMAKE_BINARY_DIR}/bakenn_cmsis_compat/CMSIS/NN")
  file(MAKE_DIRECTORY "${BAKENN_CMSIS_COMPAT_ROOT}")
  file(CREATE_LINK "${BAKENN_CMSIS_NN_ROOT}/Include"
       "${BAKENN_CMSIS_COMPAT_ROOT}/Include" SYMBOLIC COPY_ON_ERROR)

  file(GLOB_RECURSE BAKENN_CMSIS_NN_SOURCES CONFIGURE_DEPENDS
       "${BAKENN_CMSIS_NN_ROOT}/Source/*.c")
  if(NOT BAKENN_CMSIS_NN_SOURCES)
    message(FATAL_ERROR "No CMSIS-NN C sources found under ${BAKENN_CMSIS_NN_ROOT}/Source")
  endif()

  set(BAKENN_CMSIS_FC_WRAPPER
      "${BAKENN_TFLM_ROOT}/tensorflow/lite/micro/kernels/cmsis_nn/fully_connected.cc")
  set(BAKENN_CMSIS_CONV_WRAPPER
      "${BAKENN_TFLM_ROOT}/tensorflow/lite/micro/kernels/cmsis_nn/conv.cc")
  # Zephyr's module already compiles the reference fully_connected.cc.  The
  # CMSIS wrapper exports the same generic registration symbol, so rename only
  # that unused entry point in the wrapper translation unit and explicitly use
  # its Register_FULLY_CONNECTED_INT8() entry from main.cpp.
  set_source_files_properties(${BAKENN_CMSIS_FC_WRAPPER}
    PROPERTIES COMPILE_DEFINITIONS
      "Register_FULLY_CONNECTED=Register_FULLY_CONNECTED_CMSIS_NN")
  set_source_files_properties(${BAKENN_CMSIS_CONV_WRAPPER}
    PROPERTIES COMPILE_DEFINITIONS
      "Register_CONV_2D=Register_CONV_2D_CMSIS_NN")
  target_sources(app PRIVATE ${BAKENN_CMSIS_FC_WRAPPER}
    ${BAKENN_CMSIS_CONV_WRAPPER}
    ${BAKENN_CMSIS_NN_SOURCES})
  target_include_directories(app PRIVATE
    "${CMAKE_BINARY_DIR}/bakenn_cmsis_compat"
    "${BAKENN_CMSIS_NN_ROOT}/Include"
    "${BAKENN_CMSIS_CORE_ROOT}")
  # The pinned 2021 TFLM wrapper still names the old ARM_MATH_SUCCESS status;
  # CMSIS-NN v4 renamed it to ARM_CMSIS_NN_SUCCESS.  Keep the compatibility
  # shim local to this opt-in target rather than modifying vendored sources.
  target_compile_definitions(app PRIVATE CMSIS_NN BAKENN_USE_CMSIS_NN
    ARM_MATH_SUCCESS=ARM_CMSIS_NN_SUCCESS)
endif()

# The Zephyr 2.7 TFLM headers contain intentionally unused virtual parameters.
# Keep warnings fatal for the runner while suppressing only that upstream case.
target_compile_options(app PRIVATE -Wall -Wextra -Werror -Wno-unused-parameter)
