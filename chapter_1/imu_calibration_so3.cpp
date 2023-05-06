#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

void print_mat(Eigen::MatrixXd mat){
    std::cout << mat.rows() << ", " << mat.cols() << std::endl;
}

// 观察矩阵A
const Eigen::Matrix<double, 3, 6> A = (Eigen::Matrix<double, 3, 6>() <<
    -1.47394, 1.45394, -0.86156, 0.84156, 9.74722, -9.76722,
    -1.47482, 1.43482, 9.7467, -9.7867, -0.456996, 0.416996,
    9.6983, -9.7583, -0.218759, 0.158759, 0.265791, -0.325791).finished();

// 真实矩阵G
const double g = 9.80665;
const Eigen::Matrix<double, 3, 6> G = (Eigen::Matrix<double, 3, 6>() <<
        0, 0, 0, 0, g, -g,
        0, 0, g, -g, 0, 0,
        g, -g, 0, 0, 0, 0
    ).finished();

// 修正代价函数类
class OptimizationProblem : public ceres::SizedCostFunction<18, 3, 6, 3, 3> {
public:
    virtual ~OptimizationProblem() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {

        Eigen::Map<const Sophus::SO3d> R(parameters[0]);
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> T_vec(parameters[1]);
        Eigen::Map<const Eigen::Matrix<double, 3, 1>> K_vec(parameters[2]);
        Eigen::Map<const Eigen::Matrix<double, 3, 1>> b(parameters[3]);

        Eigen::Matrix3d T = Eigen::Matrix3d::Identity();
        T(0, 1) = T_vec(0);
        T(0, 2) = T_vec(1);
        T(1, 2) = T_vec(2);
        T(1, 0) = T_vec(3);
        T(2, 0) = T_vec(4);
        T(2, 1) = T_vec(5);

        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K.diagonal() = K_vec;

        // 计算残差
        Eigen::Matrix<double, 3, 6> est_G = R.matrix() * T * K * (A + b * Eigen::RowVectorXd::Ones(6));
        Eigen::Matrix<double, 18, 1> res;
        for (int col = 0; col < 6; ++col) {
            res.block<3, 1>(col * 3, 0) = est_G.col(col) - G.col(col);
        }
        for (int i = 0; i < 18; ++i){
            residuals[i] = res[i];
        }

        // 计算jacobian矩阵
        if (jacobians != NULL) {
            if (jacobians[0] != NULL) {
                // 计算d(est_G)/d(R)
                Eigen::Map<Eigen::Matrix<double, 18, 3>> dR(jacobians[0]);
                Eigen::Matrix3d TK = T * K;

                for (int col = 0; col < 6; ++col) {
                    Eigen::Vector3d temp = TK * (A.col(col) + b);
                    dR.block<3, 3>(col * 3, 0) = -Sophus::SO3d::hat(temp) * R.matrix();
                }
            }
            if (jacobians[1] != NULL) {
                // 计算d(est_G)/d(T)
                Eigen::Map<Eigen::Matrix<double, 18, 6>> dT(jacobians[1]);
                dT.setZero();

                for (int col = 0; col < 6; ++col) {
                    // std::cout << col << "----------------" << std::endl;
                    // print_mat(R.matrix());
                    // print_mat(K);
                    // print_mat(A.col(col));
                    // print_mat(b);
                    // print_mat(R.matrix() * K * (A.col(col) + b));
                    // std::cout << "----------------" << std::endl;
                    dT.block<3, 1>(col * 3, 0) = R.matrix() * K * (A.col(col) + b);
                    dT.block<3, 1>(col * 3, 1) = R.matrix() * K * (A.col(col) + b);
                    dT.block<3, 1>(col * 3, 2) = R.matrix() * K * (A.col(col) + b);
                }
            }
            // if (jacobians[2] != NULL) {
            //     // 计算d(est_G)/d(K)
            //     Eigen::Map<Eigen::Matrix<double, 18, 3>> dK(jacobians[2]);
            //     dK.setZero();

            //     for (int col = 0; col < 6; ++col) {
            //         dK.block<3, 3>(col * 3, 0) = R.matrix() * T * (A.col(col) + b) * Eigen::Matrix3d::Identity();
            //     }
            // }
            // if (jacobians[3] != NULL) {
            //     // 计算d(est_G)/d(b)
            //     Eigen::Map<Eigen::Matrix<double, 18, 3>> db(jacobians[3]);

            //     for (int col = 0; col < 6; ++col) {
            //         db.block<3, 3>(col * 3, 0) = R.matrix() * T * K;
            //     }
            // }
        }

        return true;
    }
};


// 主函数
int main() {
    // ...设置R、T、K和b的初始值...
    Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Vector3d::Zero()); // 设定R为单位旋转矩阵
    Eigen::Matrix<double, 6, 1> T_vec = Eigen::Matrix<double, 6, 1>::Zero(); // 设定T为单位矩阵，T_vec中的元素全部为0
    Eigen::Matrix<double, 3, 1> K_vec = Eigen::Vector3d::Zero(); // 设定K为单位矩阵，K_vec中的元素全部为0
    Eigen::Matrix<double, 3, 1> b = Eigen::Vector3d::Zero(); // 设定b为零向量

    // 构建优化问题
    ceres::Problem problem;
    ceres::CostFunction* cost_function = new OptimizationProblem();
    problem.AddResidualBlock(cost_function, NULL, R.data(), T_vec.data(), K_vec.data(), b.data());

    // 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-6;

    // 运行优化求解器
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出优化结果
    std::cout << "Final R: " << std::endl << R.matrix() << std::endl;
    // std::cout << "Final T: " << std::endl << Eigen::Matrix3d::Identity() + T_vec.matrix().triangularView<Eigen::StrictlyLower>() << std::endl;
    // std::cout << "Final K: " << std::endl << Eigen::Matrix3d::Identity() + K_vec.matrix().asDiagonal() << std::endl;
    std::cout << "Final b: " << std::endl << b << std::endl;

    return 0;
}
