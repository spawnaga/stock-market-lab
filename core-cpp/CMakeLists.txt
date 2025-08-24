cmake_minimum_required(VERSION 3.10)
project(CPP_Backend)

set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)
find_package(PQXX REQUIRED)
find_package(CURL REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(/usr/include/jsoncpp)

# Add executable
add_executable(cpp_backend 
    src/main.cpp
)

# Link libraries
target_link_libraries(cpp_backend 
    ${Boost_LIBRARIES}
    websocketpp
    pqxx
    ${CURL_LIBRARIES}
)