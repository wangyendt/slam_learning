#include <sophus/so3.hpp>
#include <iostream>
#include <Eigen/Core>

int main(int argc, char** argv){
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
    std::cout << a.inverse() << std::endl;
    std::cout << a * a.inverse() << std::endl;
    Eigen::Vector3d r(1.01, 1.02, 1.03);
    Sophus::SO3d R = Sophus::SO3d::exp(r);
    auto R_inv = R.inverse();
    std::cout << R_inv.log() << std::endl;
    Eigen::Vector3d w(0.001, 0.002, 0.003);
    std::cout << r[0] << "\t" << r[1] << "\t" << r[2] << std::endl;
    std::cout << R.matrix() << std::endl;
    Sophus::SO3d R_ = R * Sophus::SO3d::exp(w);
    std::cout << R_.log() << std::endl;
    Eigen::Quaterniond q = R.unit_quaternion();
    std::cout << q.coeffs() << std::endl;
    Eigen::Quaterniond q_ = (q * Eigen::Quaterniond(1.0, w.x() / 2.0, w.y() / 2.0, w.z() / 2.0));
    std::cout << q_.coeffs() << std::endl;
    q_.normalize();
    std::cout << q_.coeffs() << std::endl;
    std::cout << q_.toRotationMatrix() * R_.matrix().transpose() << std::endl;
}
