cmake_minimum_required(VERSION 3.10)
project(cev_localization LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

# CEV
find_package(cev_msgs REQUIRED)

# Add subdirectory for the library
add_subdirectory(cev_planner)

# Include directories for project headers
include_directories(
    include
    cev_planner/include
)

# Source files in the main src folder
set(SOURCES
)

# Declare the executable

# Install additional resources
install(DIRECTORY launch config
    DESTINATION share/${PROJECT_NAME})

# Ament package configuration
ament_package()
