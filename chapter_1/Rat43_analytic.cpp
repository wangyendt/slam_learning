#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <iostream>
#include <math.h>

const int kNumObservations = 15;
// clang-format off
const double data[] = {
    1.0E0,   16.08E0,
    2.0E0,   33.83E0,
    3.0E0,   65.80E0,
    4.0E0,   97.20E0,
    5.0E0,  191.55E0,
    6.0E0,  326.20E0,
    7.0E0,  386.87E0,
    8.0E0,  520.53E0,
    9.0E0,  590.03E0,
    10.0E0, 651.92E0,
    11.0E0, 724.93E0,
    12.0E0, 699.56E0,
    13.0E0, 689.96E0,
    14.0E0, 637.56E0,
    15.0E0, 717.41E0,
};


class Rat43Analytic : public ceres::SizedCostFunction<1, 1, 1, 1, 1>{
    public:
    Rat43Analytic(double x, double y) : x_(x), y_(y){
    }
    virtual ~Rat43Analytic(){}
    virtual bool Evaluate(double const * const * parameters,
    double* residuals,
    double ** jacobians)const{
        const double b1 = parameters[0][0];
        const double b2 = parameters[0][1];
        const double b3 = parameters[0][2];
        const double b4 = parameters[0][3];
        residuals[0] = b1 * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) - y_;
        if (!jacobians) return true;
        double * jacobian = jacobians[0];
        if (!jacobian) return true;
        jacobian[0] = pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4);
        jacobian[1] = -b1 * exp(b2 - b3 * x_) * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4 - 1.0) / b4;
        jacobian[2] = x_ * b1 * exp(b2 - b3 * x_) * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4 - 1.0) / b4;
        jacobian[3] = b1 * log(1 + exp(b2 - b3 * x_)) * pow(1.0 + exp(b2 - b3 * x_), -1.0 / b4) / b4 / b4;
        return true;
    }
    private:
    double x_;
    double y_;
};

class Rat43AnalyticOptimized : public ceres::SizedCostFunction<1, 1, 1, 1, 1>{
    public:
    Rat43AnalyticOptimized(const double x, const double y): x_(x),y_(y){}
    virtual ~Rat43AnalyticOptimized(){}
    virtual bool Evaluate(double const * const * parameters,
    double* residuals,
    double **jacobians)const{
        const double b1 = parameters[0][0];
        const double b2 = parameters[0][1];
        const double b3 = parameters[0][2];
        const double b4 = parameters[0][3];
        const double t1 = exp(b2 - b3 * x_);
        const double t2 = 1.0 + t1;
        const double t3 = pow(t2, -1.0 / b4);
        residuals[0] = b1 * t3 - y_;
        if (!jacobians) return true;
        double* jacobian = jacobians[0];
        if (!jacobian[0]) return true;
        const double t4 = pow(t2, -1.0 / b4 - 1.0);
        jacobian[0] = t3;
        jacobian[1] = -b1 * t1 * t4 / b4;
        jacobian[2] = -x_ * jacobian[1];
        jacobian[3] = b1 * log(t2) * t3 / (b4 * b4);
        return true;
    }
    private:
    double x_;
    double y_;
};

void verify(double b1, double b2, double b3, double b4, double x, double y){
    double y_ = b1 * pow(1.0 + exp(b2 - b3 * x), (-1.0 / b4));
    std::cout << y_ << "----" << y << std::endl;
}

int main(int argc, char **argv){
  google::InitGoogleLogging(argv[0]);

  // The variable to solve for with its initial value.
  double b1 = 0.0;
  double b2 = 1.0;
  double b3 = 1.0;
  double b4 = 1.0;
  double bs[4] = {b1,b2,b3,b4};

  // Build the problem.
  ceres::Problem problem;

    for (int i = 0; i < kNumObservations; ++i){
        ceres::CostFunction *cost_function = new Rat43AnalyticOptimized(data[2*i],data[2*i+1]);
        // problem.AddResidualBlock(cost_function, NULL, &m, &c);
        problem.AddResidualBlock(cost_function, nullptr, &b1, &b2, &b3, &b4);
    }


  // Run the solver!
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "b1: " << b1 << "\n"
            << "b2: " << b2 << "\n"
            << "b3: " << b3 << "\n"
            << "b4: " << b4 << "\n";

  for (int i = 0; i< 15;++i){
    verify(b1,b2,b3,b4,data[2*i],data[2*i+1]);
  }
  return 0;
}