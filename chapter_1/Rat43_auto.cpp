#include <iostream>
#include <Eigen/Core>
#include <ceres/ceres.h>
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

struct ExponentialModelCost{
    ExponentialModelCost(double x, double y): x_(x), y_(y){}
    template <typename T>
    bool operator()(const T* const b1, const T* const b2, const T* const b3, const T* const b4, T* residuals) const {
        residuals[0] = y_ - (b1[0] * pow(1.0 + exp(b2[0] - b3[0] * T(x_)), (-1.0 / b4[0])));
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
    double b1 = 0.0;
    double b2 = 1.0;
    double b3 = 1.0;
    double b4 = 1.0;
    ceres::Problem problem;
    for (int i = 0; i< 15;++i){
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ExponentialModelCost, 1, 1, 1, 1, 1>(
                new ExponentialModelCost(data[2*i],data[2*i+1])),
                nullptr, &b1, &b2, &b3, &b4);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "b2: " << b2 << "\n";
    std::cout << "b3: " << b3 << "\n";
    std::cout << "b4: " << b4 << "\n";
    std::cout << "b1: " << b1 << "\n";
    for (int i = 0; i< 15;++i){
        verify(b1,b2,b3,b4,data[2*i],data[2*i+1]);
    }
    return 0;
}