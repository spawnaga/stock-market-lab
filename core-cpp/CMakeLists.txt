cmake_minimum_required(VERSION 3.10)
project(CPP_Backend)

set(CMAKE_CXX_STANDARD 17)

# ---------------------------------------------------------
# Dependencies via pkg-config
# ---------------------------------------------------------
find_package(PkgConfig REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread)

# PostgreSQL C++ (pqxx)
pkg_check_modules(PQXX REQUIRED libpqxx)
include_directories(${PQXX_INCLUDE_DIRS})
link_directories(${PQXX_LIBRARY_DIRS})

# JsonCpp
pkg_check_modules(JSONCPP REQUIRED jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})
link_directories(${JSONCPP_LIBRARY_DIRS})

# ---------------------------------------------------------
# Build target
# ---------------------------------------------------------
add_executable(cpp_backend 
    src/main.cpp
)

# Link libraries
target_link_libraries(cpp_backend 
    ${Boost_LIBRARIES}
    ${PQXX_LIBRARIES}
    ${JSONCPP_LIBRARIES}
)
