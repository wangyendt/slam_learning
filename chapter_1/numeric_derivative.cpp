#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value.
  double initial_x = 5.0;
  double x = initial_x;

  // Build the problem.
  ceres::Problem problem;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
//   ceres::CostFunction* cost_function =
//       new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
//   problem.AddResidualBlock(cost_function, nullptr, &x);
  ceres::CostFunction* cost_function =
    new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
        new NumericDiffCostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, &x);

  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x
            << " -> " << x << "\n";
  return 0;
}
