project(ceres_learning)
cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops -Wall -Wextra")


# Find required packages
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

include_directories(.)


# List of executables
set(EXECUTABLES
    least_sq
    auto_derivative
    numeric_derivative
    analytic_derivative
    powell_function
    curve_fitting
    Rat43_analytic
    Rat43_auto
    imu_calibration
    imu_calibration_auto
    imu_calibration_manual
    imu_calibration_manual_2
    imu_calibration_so3
    chapter_1_basic
)

# Add the executables and link the libraries
foreach(executable ${EXECUTABLES})
  add_executable(${executable} ${executable}.cpp)
  target_link_libraries(${executable} PRIVATE
      Ceres::ceres
      Eigen3::Eigen
      Sophus::Sophus
  )
endforeach()

