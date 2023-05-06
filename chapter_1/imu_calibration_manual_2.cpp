#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

double g = 9.7833;

double G_arr[] = {
    0,0,g,
    0,0,-g,
    0,g,0,
    0,-g,0,
    g,0,0,
    -g,0,0
};

void make_fake_matrix_A(Eigen::Matrix3d &R, Eigen::Matrix3d &T, Eigen::Matrix3d &K, Eigen::Vector3d &b, Eigen::Matrix<double,3,6> &A, Eigen::Matrix<double,3,6> &G){
    Eigen::Vector3d rotation_vector(0.02, 0.03, 0.04);
    Eigen::AngleAxisd angle_axis(rotation_vector.norm(), rotation_vector.normalized());
    R = angle_axis.toRotationMatrix();
    T << 1.0,  0.13, 0.14,
         0.0,  1.0, 0.17,
         0.0,  0.0, 1.0;
    K << 1.003, 0.0, 0.0,
         0.0, 1.004, 0.0,
         0.0, 0.0, 1.005;
    b << 0.01, 0.02, 0.03;
    Eigen::Map<Eigen::Matrix<double, 3,6>, Eigen::RowMajor> G_map(G_arr);
    G = G_map;
    A = (R * T * K).inverse() * G - b * Eigen::MatrixXd::Ones(1,6);
}

class CustomCostFunction : public ceres::SizedCostFunction<3, 3, 3, 3, 3> {
 public:
  CustomCostFunction(const Eigen::Vector3d& g, const Eigen::Vector3d& a)
      : g_(g), a_(a) {}

  virtual ~CustomCostFunction() {}

  virtual bool Evaluate(double const* const* parameters,
                         double* residuals,
                         double** jacobians) const {
        const double* r = parameters[0];
        const double* t = parameters[1];
        const double* k = parameters[2];
        const double* b = parameters[3];

        Eigen::Vector3d r_vec(r[0], r[1], r[2]);
        Sophus::SO3d R = Sophus::SO3d::exp(r_vec); // 用Sophus将旋转向量转换为SO3矩阵

        Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
        T(0, 1) = t[0];
        T(0, 2) = t[1];
        T(1, 2) = t[2];

        Eigen::Matrix3d K = Eigen::Matrix3d::Zero();
        K(0, 0) = k[0];
        K(1, 1) = k[1];
        K(2, 2) = k[2];

        Eigen::Vector3d b_vec(b[0], b[1], b[2]);

        Eigen::Vector3d error = g_ - R.matrix() * T * K * (a_ + b_vec);
        residuals[0] = error[0];
        residuals[1] = error[1];
        residuals[2] = error[2];

        // 如果需要计算雅可比矩阵
        if (jacobians != NULL) {
            // 计算旋转矩阵R关于旋转向量r的雅可比矩阵
            Eigen::Matrix3d dR_dr = R.matrix() * Sophus::SO3d::hat(r_vec).transpose();

            if (jacobians[0] != NULL) {
                // 计算残差关于旋转向量r的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_r(jacobians[0]);
                J_r = -dR_dr * T * K;
            }

            if (jacobians[1] != NULL) {
                // 计算残差关于T的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_t(jacobians[1]);
                J_t = -R.matrix() * K;
            }

            if (jacobians[2] != NULL) {
                // 计算残差关于K的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_k(jacobians[2]);
                J_k = -R.matrix() * T * Eigen::Matrix3d::Identity();
            }

            if (jacobians[3] != NULL) {
                // 计算残差关于b的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J_b(jacobians[3]);
                J_b = -R.matrix() * T * K;
            }
        }
        return true;
    }
    private:
    const Eigen::Vector3d g_;
    const Eigen::Vector3d a_;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    Eigen::Matrix<double,3,3> R;
    Eigen::Matrix<double,3,3> T;
    Eigen::Matrix<double,3,3> K;
    Eigen::Vector3d B;
    Eigen::Matrix<double,3,6> A;
    Eigen::Matrix<double,3,6> G;
    make_fake_matrix_A(R,T,K,B,A,G);
    std::cout << "R:\n" << R << std::endl;
    std::cout << "T:\n" << T << std::endl;
    std::cout << "K:\n" << K << std::endl;
    std::cout << "B:\n" << B << std::endl;
    std::cout << "A:\n" << A << std::endl;
    std::cout << "G:\n" << G << std::endl;

    // 初始化变量（这里使用了示例数据，您可以替换为实际数据）

    double k[3] = {0.99, 0.98, 0.97};
    double b[3] = {0.001, 0.002, 0.003};

    // 初始化变量（这里使用了示例数据，您可以替换为实际数据）
    double r[3] = {0.1, 0.1, 0.1};
    double t[3] = {0.001, 0.002, 0.003};

    // 设置优化问题
    ceres::Problem problem;

    for (int i = 0; i < 6; ++i) {
        ceres::CostFunction* cost_function =
        new CustomCostFunction(G.col(i), A.col(i));
        problem.AddResidualBlock(cost_function, NULL, r, t, k, b);
    }

    // 设置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    // 求解优化问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << "Initial r: " << 0.0 << " " << 0.0 << " " << 0.0 << std::endl;
    std::cout << "Initial t: " << 0.001 << " " << 0.002 << " " << 0.003 << std::endl;
    std::cout << "Initial k: " << 0.99 << " " << 0.98 << " " << 0.97 << std::endl;
    std::cout << "Initial b: " << 0.001 << " " << 0.002 << " " << 0.003 << std::endl;

    std::cout << "Final r: " << r[0] << " " << r[1] << " " << r[2] << std::endl;
    std::cout << "Final t: " << t[0] << " " << t[1] << " " << t[2] << std::endl;
    std::cout << "Final k: " << k[0] << " " << k[1] << " " << k[2] << std::endl;
    std::cout << "Final b: " << b[0] << " " << b[1] << " " << b[2] << std::endl;

    return 0;
}