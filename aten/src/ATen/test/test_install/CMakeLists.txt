<<<<<<< HEAD
cmake_minimum_required(VERSION 3.0)
=======
cmake_minimum_required(VERSION 3.5)
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
find_package(ATen REQUIRED)
include_directories(${ATEN_INCLUDE_DIR})

# C++17
if(not MSVC)
    set(CMAKE_CXX_FLAGS "--std=c++17 ${CMAKE_CXX_FLAGS}")
endif()
add_executable(main main.cpp)
target_link_libraries(main ${ATEN_LIBRARIES})
