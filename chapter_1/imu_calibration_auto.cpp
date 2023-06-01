#include <ceres/ceres.h>
#include <Eigen/Core>
#include <iostream>
#include <sophus/so3.hpp>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

struct ErrorFunction {
    ErrorFunction(const Eigen::Matrix<double, 3, 6>& A,
                  const Eigen::Matrix<double, 3, 6>& G)
        : A_(A), G_(G) {}

    template <typename T>
    bool operator()(const T* const x, T* residual) const {
    Eigen::Matrix<T, 3, 3> rot = (Eigen::AngleAxis<T>(x[0], Eigen::Matrix<T, 3, 1>::UnitX())
        * Eigen::AngleAxis<T>(x[1], Eigen::Matrix<T, 3, 1>::UnitY())
        * Eigen::AngleAxis<T>(x[2], Eigen::Matrix<T, 3, 1>::UnitZ())).toRotationMatrix();
        Eigen::Matrix<T, 3, 3> T_mat;
        T_mat << T(1), x[3], x[4],
                 x[5], T(1), x[6],
                 x[7], x[8], T(1);
        Eigen::Matrix<T, 3, 3> K;
        K << x[9], T(0), T(0),
             T(0), x[10], T(0),
             T(0), T(0), x[11];
        Eigen::Matrix<T, 3, 1> b(x + 12);

        for (int i = 0; i < 6; ++i) {
            Eigen::Matrix<T, 3, 1> G_estimated = rot * T_mat * K * (A_.col(i).cast<T>() + b);
            residual[i * 3] = G_(0, i) - G_estimated(0);
            residual[i * 3 + 1] = G_(1, i) - G_estimated(1);
            residual[i * 3 + 2] = G_(2, i) - G_estimated(2);
        }
        return true;
    }

private:
    const Eigen::Matrix<double, 3, 6> A_;
    const Eigen::Matrix<double, 3, 6> G_;
};


class SO3LocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const {return 3;};
    virtual int LocalSize() const {return 3;};
    // virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    // virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    // virtual int GlobalSize() const { return 6; };
    // virtual int LocalSize() const { return 6; };
};

bool SO3LocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{

    // Eigen::Map<const Eigen::Matrix<double, 3, 1>> lie(x);
    // Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_lie(delta);
    
    // Sophus::SO3d R = Sophus::SO3d::exp(lie);
    // Sophus::SO3d delta_R = Sophus::SO3d::exp(delta_lie);
    // Eigen::Matrix<double, 3, 1> x_plus_delta_lie = (delta_R * R).log();
    // for(int i = 0; i < 3; ++i)
    //     x_plus_delta[i] = x_plus_delta_lie(i, 0);


    Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Map<const Eigen::Vector3d>(x));
    Sophus::SO3d dR = Sophus::SO3d::exp(Eigen::Map<const Eigen::Vector3d>(delta));
    Eigen::Map<Eigen::Vector3d> R_new(x_plus_delta);

    R_new = (dR * R).log();

    return true;


    // Sophus::SO3d R = Sophus::SO3d::exp(Eigen::Map<const Eigen::Vector3d>(x));
    // Eigen::Map<const Eigen::Vector3d> p(x + 3);    // double 指针转为eigen数组
    // Sophus::SO3d dR = Sophus::SO3d::exp(Eigen::Map<const Eigen::Vector3d>(delta));
    // Eigen::Map<const Eigen::Vector3d> dp(delta + 3);
    // Eigen::Map<Eigen::Vector3d> R_new(x_plus_delta);
    // Eigen::Map<Eigen::Vector3d> p_new(x_plus_delta + 3);
    // // 注意这里是右乘更新
    // // [ R' t' ] = [R t][dR dt] 
    // // [ 0  1  ]   [0 1][0  1]
    // // 即 R' = R * dR   -->  R_so3 = (R*dR).log(),  
    // //    t' = t + R*dt
    // R_new = (R * dR).log();
    // p_new = p + R * dp;       // update with right multiply.
    // return true;
}

bool SO3LocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    // ceres::MatrixRef(jacobian, GlobalSize(), LocalSize()).setIdentity();
    ceres::MatrixRef(jacobian, 3, 3) = ceres::Matrix::Identity(3, 3);
    return true;

    // Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > j(jacobian);
    // j.topRows<6>().setIdentity();  // 这里直接设成单位阵就好，雅可比在 cost function 里实现。
    // return true;
}


class MyCostFunction : public ceres::SizedCostFunction<18, 3, 3, 3, 3> {
public:
    MyCostFunction(const Eigen::Matrix<double, 3, 6>& A,
                   const Eigen::Matrix<double, 3, 6>& G)
        : A_(A), G_(G) {}

    virtual ~MyCostFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        // 计算残差
        Eigen::Vector3d omega(parameters[0][0], parameters[0][1], parameters[0][2]);
        Sophus::SO3d R = Sophus::SO3d::exp(omega);
        Eigen::Matrix<double, 3, 3> rot = R.matrix();

        Eigen::Matrix<double, 3, 3> T_mat;
        T_mat << 1.0, parameters[1][0], parameters[1][1],
                0.0, 1.0, parameters[1][2],
                0.0, 0.0, 1.0;
        Eigen::Matrix<double, 3, 3> K;
        K << parameters[2][0], 0.0, 0.0,
            0.0, parameters[2][1], 0.0,
            0.0, 0.0, parameters[2][2];
        Eigen::Matrix<double, 3, 1> b(parameters[3]);

        for (int i = 0; i < 6; ++i) {
            Eigen::Matrix<double, 3, 1> G_estimated = rot * T_mat * K * (A_.col(i) + b);
            residuals[i * 3] = G_(0, i) - G_estimated(0);
            residuals[i * 3 + 1] = G_(1, i) - G_estimated(1);
            residuals[i * 3 + 2] = G_(2, i) - G_estimated(2);
        }

        // 计算雅克比矩阵
        Eigen::Matrix<double, 18, 3> JR;
        Eigen::Matrix<double, 18, 3> JT;
        Eigen::Matrix<double, 18, 3> JK;
        Eigen::Matrix<double, 18, 3> Jb;

        if (jacobians != NULL && jacobians[0] != NULL) {
            for(int i = 0; i < 6; ++i){
                //计算对旋转矩阵雅克比 (Rp)^
                Eigen::Vector3d vec = rot * T_mat * K * (A_.col(i) + b);
                Eigen::Matrix3d vec_hat = Sophus::SO3d::hat(vec);
                // Eigen::Matrix3d vec_hat;
                // vec_hat << 0, -vec.z(), vec.y(),
                //            vec.z(), 0, -vec.x(),
                //             -vec.y(), vec.x(), 0; 
                JR.block<3, 3>(i*3, 0) = vec_hat;
                //计算对T矩阵的雅克比
                // Eigen::Matrix<double, 3, 6> temp;
                // temp << K(1, 1)*(A_(1, i)+b(1)), K(2,2)*(A_(2,i)+b(2)), 0, 0, 0, 0,
                //         0, 0, K(0,0)*(A_(0,i)+b(0)), K(2,2)*(A_(2,i)+b(2)), 0, 0,
                //         0, 0, 0, 0, K(0,0)*(A_(0,i)+b(0)), K(1,1)*(A_(1,i)+b(1));
                Eigen::Matrix<double, 3, 3> temp;
                temp << K(1,1)*(A_(1,i)+b(1)),  K(2,2)*(A_(2,i)+b(2)),      0,
                        0,                      0,                          K(2,2)*(A_(2,i)+b(2)),
                        0,                      0,                          0;
                JT.block<3,3>(i*3, 0) = -rot * temp;
                //计算对K矩阵的雅克比
                Eigen::Matrix3d temp2;
                temp2 << A_(0,i)+b(0), 0, 0,
                         0, A_(1,i)+b(1), 0,
                         0, 0, A_(2,i)+b(2);
                JK.block<3,3>(i*3, 0) = -rot * T_mat * temp2;
                // JK.setZero();

                //计算对b矩阵的雅克比
                Jb.block<3,3>(i*3, 0) = -rot * T_mat * K;
            }

            Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>>(jacobians[0], 18, 3) = JR;
            Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>>(jacobians[1], 18, 3) = JT;
            Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>>(jacobians[2], 18, 3) = JK;
            Eigen::Map<Eigen::Matrix<double, 18, 3, Eigen::RowMajor>>(jacobians[3], 18, 3) = Jb;
        }

        return true;
    }

private:
    const Eigen::Matrix<double, 3, 6> A_;
    const Eigen::Matrix<double, 3, 6> G_;
};

int main(int argc, char** argv) {

    //给定T_,K_,b,R，G,生成仿真测量数据A
    /**************************************/
    Eigen::Matrix<double, 3, 3> T_;
    T_ << 1, 0.12, 0.08,
         0, 1, 0.05,
         0, 0, 1;
    Eigen::Matrix<double, 3, 3> K_;
    K_ << 1.03, 0, 0,
         0, 0.95, 0,
         0, 0, 1.02;
    
    Eigen::Vector3d b_;
    b_ << 0.21, 0.13, 0.05;

    double r[3] = {0.02, 0.03, 0.05};
    auto r_vec = Eigen::Vector3d(r);
    Eigen::Matrix<double, 3, 3> R_ = Sophus::SO3d::exp(r_vec).matrix();
    
    std::cout << "R : "  << std::endl << R_ << std::endl;
     
    // 6面
    double g = 9.7833;
    Eigen::Matrix<double, 3, 6> G;
    G <<   0 ,   0 , 0, 0 ,   g ,-g,
           0 ,   0 , g, -g,  0, 0,
           g ,   -g, 0, 0 ,   0, 0;

    std::cout << "G : " << std::endl << G << std::endl;

    // 根据已知数据生成仿真测量数据A
    Eigen::Matrix<double, 3, 6> A;
    auto temp = R_*T_*K_;
    for(int i = 0; i < 6; ++i){
        A.col(i) = temp.inverse() * G.col(i) - b_;
    }

    std::cout << "A : " << std::endl << A << std::endl;

    // /**************************************/

    // 实际数据
    /****************************************/
    // Eigen::Matrix<double, 3, 6> A;
    // A << -0.10056209, 0.06355833, -0.08482739, 0.2181346, 9.76441422, -9.79061516,
    //      -2.86120622 ,2.45145921, 9.75056835,  -9.6468714, 0.1106125, -0.57045186,
    //      9.46849853, -9.4231117, 0.90939053,  -2.65919393, 0.17637564,-0.13308613;
    
    // std::cout << "A : " << std::endl << A << std::endl;

    // double g = 9.7833;
    // Eigen::Matrix<double, 3, 6> G;
    // G <<   0 ,   0 , 0, 0 ,   g ,-g,
    //        0 ,   0 , g, -g,  0, 0,
    //        g ,   -g, 0, 0 ,   0, 0;

    // std::cout << "G : " << std::endl << G << std::endl;

    /****************************************/

    // Initialize parameters
    double r_param[3] = {0,0,0};
    double T_param[3] = {0,0,0};
    // double K_param[3] = {1,1,1};
    double K_param[3] = {1.03,0.95,1.02};
    double b_param[3] = {0,0,0};

    // Define the optimization problem
    Problem problem;
    //使用自动微分计算
    // CostFunction* cost_function =
    //     new AutoDiffCostFunction<ErrorFunction,18 ,15>(new ErrorFunction(A,G));
    //使用自己推导的雅克比计算
    CostFunction* cost_function = new MyCostFunction(A, G);
    problem.AddResidualBlock(cost_function,nullptr,r_param,T_param,K_param,b_param);
    problem.SetParameterization(r_param , new SO3LocalParameterization());

    // Run the solver
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options,&problem,&summary);

    std::cout << summary.BriefReport() << "\n";
    
    // 输出结果
    std::cout << "Optimized r:\n";
    std::cout << r_param[0] << "," << r_param[1] << "," << r_param[2] << std::endl;

    std::cout << "Optimized R:\n";
    Eigen::Vector3d omega(r_param[0], r_param[1], r_param[2]);
    Sophus::SO3d R = Sophus::SO3d::exp(omega);
    std::cout << R.matrix() << "\n";

    std::cout << "Optimized T:\n";
    Eigen::Matrix3d T;
    T << 1, T_param[0], T_param[1],
        0, 1, T_param[2],
        0, 0, 1;
    std::cout << T << "\n";
    std::cout << "Optimized K:\n";
    Eigen::Matrix3d K;
    K << K_param[0], 0, 0,
         0, K_param[1],0,
         0, 0, K_param[2];
    std::cout << K << std::endl; 

    std::cout << "Optimized b:\n";
    auto b = Eigen::Map<Eigen::Vector3d>(b_param);
    std::cout << b << "\n";

    return 0;
}
