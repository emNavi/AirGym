#ifndef _QR_CONTROLLER_H_
#define _QR_CONTROLLER_H_

#include <Eigen/Eigen>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include "vechicle_controller/helper_function.hpp"
class uav_controller
{
private:
    float diff_step_;
    float F_param_[3], G_param_[3];
    float P_param_[3], I_param_[3], D_param_[3];

    float err_x_, err_y_, err_z_;
    float u_all_x_, u_all_y_, u_all_z_;
    float urc_x_, urc_y_, urc_z_;

    bool init_x_r_;
    Eigen::VectorXd x_r_, B_u_;
    Eigen::VectorXd B_x_x_, B_x_y_, B_x_z_;
    Eigen::MatrixXd A_x_, A_y_, A_z_;
    Eigen::VectorXd x_r_x_, x_r_y_, x_r_z_;

    geometry_msgs::PoseStamped exp_pos_;
    geometry_msgs::PoseStamped cur_pos_;
    geometry_msgs::TwistStamped exp_vel_;
    geometry_msgs::TwistStamped cur_vel_;
    geometry_msgs::Vector3Stamped exp_acc_;
    geometry_msgs::Vector3Stamped cur_acc_;

    geometry_msgs::TwistStamped ctrl_vel_;
    geometry_msgs::Vector3Stamped ctrl_accel_;

    void initParams_();
    void cal_pid_ctrl_();
    void cal_robust_ctrl_();
    void cal_robust_ctrl_xy_();
    void cal_single_iteration_();

    Helper_function helper_func;

public:
    uav_controller(float diff_step)
        : diff_step_(diff_step),
          F_param_{3, 3, 3},
          G_param_{0.3, 0.3, 0.3},
          P_param_{-2, -2, -2},
          I_param_{-0.2, -0.2, -0.2},
          D_param_{-0.8, -0.8, -0.8},
          init_x_r_(false)

    {
        initParams_();
        helper_func.set_dt(diff_step_);
    }

    void reset_params_PID(float *P_param, float *I_param, float *D_param);
    void reset_params_robust(float *F_Param, float *G_param);

    inline geometry_msgs::TwistStamped getCtrl_single_iteration()
    {
        cal_single_iteration_();
        return ctrl_vel_;
    }
    inline geometry_msgs::Vector3Stamped getCtrl_pid()
    {
        cal_pid_ctrl_();
        return ctrl_accel_;
    }

    inline geometry_msgs::Vector3Stamped getCtrl_robust()
    {
        cal_robust_ctrl_();
        return ctrl_accel_;
    }
    inline geometry_msgs::Vector3Stamped getCtrl_robust_xy()
    {
        cal_robust_ctrl_xy_();
        return ctrl_accel_;
    }

    inline std::vector<float> get_robust_estimate()
    {
        std::vector<float> disturbance = {urc_x_, urc_y_, urc_z_};
        return disturbance;
    }

    inline void setState(geometry_msgs::PoseStamped cur_pos, geometry_msgs::TwistStamped cur_vel,
                         geometry_msgs::PoseStamped exp_pos, geometry_msgs::TwistStamped exp_vel)
    {
        cur_pos_ = cur_pos;
        cur_vel_ = cur_vel;
        exp_pos_ = exp_pos;
        exp_vel_ = exp_vel;
    }
};

#endif