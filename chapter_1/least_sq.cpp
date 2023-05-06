#include <ceres/ceres.h>
#include <iostream>

using namespace std;

struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, const T* const y, T* residual) const {
    residual[0] = (T(10.0) - x[0])*(T(10.0 - y[0]));
    return true;
  }
};

class MyCostFunction : public ceres::SizedCostFunction<1, 1, 1>{
public:
  MyCostFunction(const double &x, const double &y) : x_(x), y_(y) {}
  virtual ~MyCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double *residuals,
                        double** jacobians) const{
    double x = parameters[0][0];
    double y = parameters[1][0];
    residuals[0] = (10.0 - x) * (10.0 - y);
    if (jacobians!= NULL) {
      jacobians[0][0] = -(10.0 - y);
      jacobians[1][0] = -(10.0 - x);
    }
    return true;
  }
private:
  double x_;
  double y_;
};

int main(int argc, char** argv) {
  // 初始化参数
  double initial_x = 5.0;
  double initial_y = 6.0;
  double x = initial_x;
  double y = initial_y;
  cout << "Initial x: " << initial_x << endl;
  cout << "Initial y: " << initial_x << endl;

  // 创建最小二乘问题
  ceres::Problem problem;
  ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1>(new CostFunctor);
  problem.AddResidualBlock(cost_function, NULL, &x ,&y);

  // 设置参数为常量，并求解
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  problem.SetParameterBlockConstant(&x);
  problem.SetParameterBlockConstant(&y);
  ceres::Solver::Summary summary_const;
  ceres::Solve(options, &problem, &summary_const);
  cout << "Constant x: " << x << endl;
  cout << "Constant y: " << y << endl;
  // cout << summary_const.FullReport() << endl;

  // 设置参数为非常量，并求解
  x = initial_x;
  y = initial_y;
  ceres::Solver::Summary summary_nonconst;
  problem.SetParameterBlockVariable(&x);
  problem.SetParameterBlockVariable(&y);
  ceres::Solve(options, &problem, &summary_nonconst);
  cout << "Non-Constant x: " << x << endl;
  cout << "Non-Constant y: " << y << endl;
  // cout << summary_nonconst.FullReport() << endl;


  x = initial_x;
  y = initial_y;
  ceres::CostFunction* cost_function2 = new MyCostFunction(x, y);
  ceres::Solver::Summary summary_manual;
  ceres::Problem problem2;
  problem2.AddResidualBlock(cost_function2, NULL, &x ,&y);
  // problem2.SetParameterBlockConstant(&x);
  ceres::Solve(options, &problem2, &summary_manual);
  cout << "Manual x: " << x << endl;
  cout << "Manual y: " << y << endl;

  return 0;
}
