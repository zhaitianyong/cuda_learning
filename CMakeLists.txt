project("cuda")
cmake_minimum_required(VERSION 2.8)

# 查找cuda 
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


#add_subdirectory(addVector)
add_subdirectory(helloWorld)
add_subdirectory(dotVector)
