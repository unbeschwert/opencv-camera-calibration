cmake_minimum_required(VERSION 3.12)
project(CameraCalibrator VERSION 1.0 DESCRIPTION "camera calibration using opencv in c++ programming language")

set(CAMERA_CALIBRATOR_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

set(CMAKE_INSTALL_PREFIX "/usr/local/")
set(CMAKE_EXPORT_COMPILE_COMMANDS True)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -Werror -Wextra -pedantic)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LIBRARY_NAME camera_calibrator)

configure_package_config_file("Config.cmake.in"
    "${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}")

find_package(OpenCV REQUIRED)

add_library(${LIBRARY_NAME} SHARED src/CameraCalibration.cc)
set_target_properties(${LIBRARY_NAME} PROPERTIES SOVERSION 1 PUBLIC_HEADER include/CameraCalibration.hh)
target_link_libraries(${LIBRARY_NAME} ${OpenCV_LIBS})
target_include_directories(${LIBRARY_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} PRIVATE ${CAMERA_CALIBRATOR_INCLUDE_DIR})

install(TARGETS ${LIBRARY_NAME} 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install (FILES ${CMAKE_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake 
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME}")
