#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <random>

/*"""
已知A是3*6的观测矩阵，每一列是一个观测值，G是一个3*6的真实矩阵，为[[0,0,g],[0,0,-g],[0,g,0],[0,-g,0],[g,0,0],[-g,0,0]]^T，根据G=R*T*K*(A+b)，需要优化出R、T、K和b，其中R、T、K是3*3的矩阵，b是3*1的向量，并且R是单位正交矩阵，是旋转矩阵，T是对角线元素全是1、其他位置元素接近于0的非正交误差矩阵，K是仅有对角线3个元素不为0（接近于1），其余元素均为0的矩阵，b是3个元素都接近0的向量。用ceres和LM算法进行优化求解
"""*/

double g = 9.7833;
// double A[] = {
//     -0.10056209, -2.86120622, 9.46849853,
//     0.06355833, 2.45145921, -9.4231117,
//     -0.08482739, 9.75056835, 0.90939053,
//     0.2181346, -9.6468714, -2.65919393,
//     9.76441422, 0.1106125, 0.17637564,
//     -9.79061516, -0.57045186, -0.13308613
// };
double G_arr[] = {
    0,0,g,
    0,0,-g,
    0,g,0,
    0,-g,0,
    g,0,0,
    -g,0,0
};

void make_fake_matrix_A(Eigen::Matrix3d &R, Eigen::Matrix3d &T, Eigen::Matrix3d &K, Eigen::Vector3d &b, Eigen::Matrix<double,3,6> &A){
    Eigen::Vector3d rotation_vector(0.02, 0.03, 0.04);
    Eigen::AngleAxisd angle_axis(rotation_vector.norm(), rotation_vector.normalized());
    R = angle_axis.toRotationMatrix();
    T << 1.0,  0.13, 0.14,
         0.15,  1.0, 0.17,
         0.18,  0.19, 1.0;
    K << 1.003, 0.0, 0.0,
         0.0, 1.004, 0.0,
         0.0, 0.0, 1.005;
    b << 0.01, 0.02, 0.03;
    Eigen::Map<Eigen::Matrix<double, 3,6>, Eigen::RowMajor> G(G_arr);
    // std::cout << R * T * K << std::endl;
    // std::cout << (R * T * K).inverse() << std::endl;
    // std::cout << (R * T * K).inverse() * G << std::endl;
    // std::cout << (R * T * K).inverse() * G - b * Eigen::MatrixXd::Ones(1,6) << std::endl;
    // std::cout << b * Eigen::MatrixXd::Ones(1,6) << std::endl;
    A = (R * T * K).inverse() * G - b * Eigen::MatrixXd::Ones(1,6);
}

int main(int argc, char** argv){
    std::mt19937 rng(123);
    Eigen::Matrix<double,3,3> R;
    Eigen::Matrix<double,3,3> T;
    Eigen::Matrix<double,3,3> K;
    Eigen::Vector3d b;
    Eigen::Matrix<double,3,6> A;
    make_fake_matrix_A(R,T,K,b,A);
    std::cout << "R:\n" << R << std::endl;
    std::cout << "T:\n" << T << std::endl;
    std::cout << "K:\n" << K << std::endl;
    std::cout << "b:\n" << b << std::endl;
    std::cout << "A:\n" << A << std::endl;
    return 0;
}
