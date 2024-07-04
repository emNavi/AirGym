#ifndef _HELPER_FUNCTION_
#define _HELPER_FUNCTION_
#include <Eigen/Eigen>
class Helper_function
{
public:
    inline void set_dt(float dt)
    {
        diff_step_ = dt;
    }
    inline Eigen::VectorXd rk45_o2(Eigen::VectorXd x_r, Eigen::MatrixXd A, Eigen::VectorXd B)
    {
        Eigen::VectorXd k_1, k_2, k_3, k_4;
        k_1 = A * x_r + B;
        k_2 = A * (x_r + diff_step_ / 2 * k_1) + B;
        k_3 = A * (x_r + diff_step_ / 2 * k_2) + B;
        k_4 = A * (x_r + diff_step_ * k_3) + B;
        Eigen::VectorXd x_r_new = x_r + diff_step_ / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4);
        return x_r_new;
    }

private:
    float diff_step_;
    int dim_;
};
#endif