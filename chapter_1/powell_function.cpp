#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <iostream>
#include <math.h>

struct F1 {
  template <typename T>
  bool operator()(const T *const x1, const T *const x2, T *residual) const {
    residual[0] = x1[0] + 10.0 * x2[0];
    return true;
  }
};

struct F2 {
  template <typename T>
  bool operator()(const T *const x3, const T *const x4, T *residual) const {
    residual[0] = sqrt(5) * (x3[0] - x4[0]);
    return true;
  }
};

struct F3 {
  template <typename T>
  bool operator()(const T *const x2, const T *const x3, T *residual) const {
    residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
    return true;
  }
};

struct F4 {
  template <typename T>
  bool operator()(const T *const x1, const T *const x4, T *residual) const {
    residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
    return true;
  }
};

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value.
  double x1 = 3.0, x1_init = 3.0;
  double x2 = -1.0, x2_init = -1.0;
  double x3 = 0.0, x3_init = 0.0;
  double x4 = 1.0, x4_init = 1.0;

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  auto cost_function_1 = new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1);
  auto cost_function_2 = new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2);
  auto cost_function_3 = new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3);
  auto cost_function_4 = new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4);
  problem.AddResidualBlock(cost_function_1, NULL, &x1, &x2);
  problem.AddResidualBlock(cost_function_2, NULL, &x3, &x4);
  problem.AddResidualBlock(cost_function_3, NULL, &x2, &x3);
  problem.AddResidualBlock(cost_function_4, NULL, &x1, &x4);

  // Run the solver!
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x1 : " << x1_init << " -> " << x1 << "\n";
  std::cout << "x2 : " << x2_init << " -> " << x2 << "\n";
  std::cout << "x3 : " << x3_init << " -> " << x3 << "\n";
  std::cout << "x4 : " << x4_init << " -> " << x4 << "\n";
  
  return 0;
}