cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(3DModel)

find_package(PCL 1.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable (3DModel model_builder.cpp)
target_link_libraries (3DModel ${PCL_LIBRARIES})