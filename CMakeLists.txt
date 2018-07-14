cmake_minimum_required(VERSION 3.10)
project(DigitRecogniser)

set(CMAKE_CXX_STANDARD 14)

add_executable(DigitRecogniser main.cpp)
target_link_libraries(DigitRecogniser caffe)