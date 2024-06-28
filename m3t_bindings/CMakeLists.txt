cmake_minimum_required(VERSION 3.15)
project(pym3t LANGUAGES CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# TODO: to remove
set(CMAKE_BUILD_TYPE Debug)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# TODO: remove?
option(USE_AZURE_KINECT "Use Azure Kinect" OFF)
option(USE_REALSENSE "Use RealSense D435" OFF)
option(USE_GTEST "Use gtest" OFF)
option(USE_XFEATURES2D "Use OpenCV xfeatures2d" OFF)

# if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
#   cmake_policy(SET CMP0148 OLD) # required for current pybind11
# endif()

# Fix this error:
# /usr/bin/ld: libm3t_ext.a(dummy_camera.cpp.o): relocation R_X86_64_PC32 against symbol
# `_ZSt4cerr@@GLIBCXX_3.4' can not be used when making a shared object; recompile with -fPIC
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

IF(USE_REALSENSE)
  ADD_DEFINITIONS(-DPYM3T_WITH_REALSENSE)
  LIST(APPEND CFLAGS_DEPENDENCIES "-DPYM3T_WITH_REALSENSE")
ENDIF(USE_REALSENSE)

include(FetchContent)
FetchContent_Declare(
  m3t
  GIT_REPOSITORY "https://github.com/agimus-project/3DObjectTracking"
  GIT_TAG "master"
  SOURCE_SUBDIR "M3T" CMAKE_ARGS "-DUSE_AZURE_KINECT=${USE_AZURE_KINECT}"
  "-DUSE_REALSENSE=${USE_REALSENSE}" "-DUSE_GTEST=${USE_GTEST}" "-DUSE_XFEATURES2D=${USE_XFEATURES2D}")
FetchContent_MakeAvailable(m3t)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Create library including the region modality base class
add_library(m3t_and_region_modality_base src/pym3t/region_modality_base.cpp include/pym3t/region_modality_base.h)
target_compile_features(m3t_and_region_modality_base PUBLIC cxx_std_17)
target_include_directories(
  m3t_and_region_modality_base
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${m3t_SOURCE_DIR}/include>)
target_link_libraries(m3t_and_region_modality_base PUBLIC m3t)


nanobind_add_module(_pym3t_mod MODULE src/pym3t/pym3t.cpp)
target_link_libraries(_pym3t_mod PUBLIC m3t_and_region_modality_base)
target_include_directories(
  _pym3t_mod PUBLIC $<INSTALL_INTERFACE:include>
                    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

install(TARGETS _pym3t_mod DESTINATION pym3t)
