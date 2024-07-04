#include "vechicle_controller/qr_controller.h"

void uav_controller::reset_params_PID(float *P_param, float *I_param, float *D_param)
{
    for (int i = 0; i < 3; i++)
    {
        P_param_[i] = P_param[i];
        I_param_[i] = I_param[i];
        D_param_[i] = D_param[i];
    }
}
void uav_controller::reset_params_robust(float *F_param, float *G_param)
{
    for (int i = 0; i < 3; i++)
    {
        F_param_[i] = F_param[i];
        G_param_[i] = G_param[i];
    }

    A_x_ << -G_param_[0], 0, 1, -F_param_[0];
    A_y_ << -G_param_[1], 0, 1, -F_param_[1];
    A_z_ << -G_param_[2], 0, 1, -F_param_[2];

    B_x_x_ << (-G_param_[0] * G_param_[0]), (F_param_[0] + G_param_[0]);
    B_x_y_ << (-G_param_[1] * G_param_[1]), (F_param_[1] + G_param_[1]);
    B_x_z_ << (-G_param_[2] * G_param_[2]), (F_param_[2] + G_param_[2]);

    // reset rubust integration
    x_r_x_ << 0, err_x_;
    x_r_y_ << 0, err_y_;
    x_r_z_ << 0, err_z_;
}

void uav_controller::initParams_()
{

    x_r_.resize(2);
    B_u_.resize(2);
    B_x_x_.resize(2);
    B_x_y_.resize(2);
    B_x_z_.resize(2);
    A_x_.resize(2, 2);
    A_y_.resize(2, 2);
    A_z_.resize(2, 2);
    x_r_x_.resize(2);
    x_r_y_.resize(2);
    x_r_z_.resize(2);

    B_u_ << 1, 0;

    u_all_x_, u_all_y_, u_all_z_ = 0;

    A_x_ << -G_param_[0], 0, 1, -F_param_[0];
    A_y_ << -G_param_[1], 0, 1, -F_param_[1];
    A_z_ << -G_param_[2], 0, 1, -F_param_[2];

    B_x_x_ << (-G_param_[0] * G_param_[0]), (F_param_[0] + G_param_[0]);
    B_x_y_ << (-G_param_[1] * G_param_[1]), (F_param_[1] + G_param_[1]);
    B_x_z_ << (-G_param_[2] * G_param_[2]), (F_param_[2] + G_param_[2]);
}

void uav_controller::cal_single_iteration_()
{
    float err_x = cur_pos_.pose.position.x - exp_pos_.pose.position.x;
    float err_y = cur_pos_.pose.position.y - exp_pos_.pose.position.y;
    float err_z = cur_pos_.pose.position.z - exp_pos_.pose.position.z;
    float u_ctrl_vel_x = P_param_[0] * err_x;
    float u_ctrl_vel_y = P_param_[1] * err_y;
    float u_ctrl_vel_z = P_param_[2] * err_z;
    ctrl_vel_.twist.linear.x = u_ctrl_vel_x;
    ctrl_vel_.twist.linear.x = u_ctrl_vel_y;
    ctrl_vel_.twist.linear.z = u_ctrl_vel_z;
}

void uav_controller::cal_pid_ctrl_()
{
    float err_x = cur_pos_.pose.position.x - exp_pos_.pose.position.x;
    float err_y = cur_pos_.pose.position.y - exp_pos_.pose.position.y;
    float err_z = cur_pos_.pose.position.z - exp_pos_.pose.position.z;
    float err_vx = cur_vel_.twist.linear.x - exp_vel_.twist.linear.x;
    float err_vy = cur_vel_.twist.linear.y - exp_vel_.twist.linear.y;
    float err_vz = cur_vel_.twist.linear.z - exp_vel_.twist.linear.z;

    float unc_x = P_param_[0] * err_x + D_param_[0] * err_vx;
    float unc_y = P_param_[1] * err_y + D_param_[1] * err_vy;
    float unc_z = P_param_[2] * err_z + D_param_[2] * err_vz; // + i_z * integration_z;

    ctrl_accel_.vector.x = unc_x;
    ctrl_accel_.vector.y = unc_y;
    ctrl_accel_.vector.z = unc_z;
}

void uav_controller::cal_robust_ctrl_()
{
    err_x_ = cur_pos_.pose.position.x - exp_pos_.pose.position.x;
    err_y_ = cur_pos_.pose.position.y - exp_pos_.pose.position.y;
    err_z_ = cur_pos_.pose.position.z - exp_pos_.pose.position.z;
    float err_vx = cur_vel_.twist.linear.x - exp_vel_.twist.linear.x;
    float err_vy = cur_vel_.twist.linear.y - exp_vel_.twist.linear.y;
    float err_vz = cur_vel_.twist.linear.z - exp_vel_.twist.linear.z;

    float unc_x = P_param_[0] * err_x_ + D_param_[0] * err_vx;
    float unc_y = P_param_[1] * err_y_ + D_param_[1] * err_vy;
    float unc_z = P_param_[2] * err_z_ + D_param_[2] * err_vz; // + i_z * integration_z;

    x_r_x_ = helper_func.rk45_o2(x_r_x_, A_x_, B_u_ * u_all_x_ + B_x_x_ * err_x_);
    x_r_y_ = helper_func.rk45_o2(x_r_y_, A_y_, B_u_ * u_all_y_ + B_x_y_ * err_y_);
    x_r_z_ = helper_func.rk45_o2(x_r_z_, A_z_, B_u_ * u_all_z_ + B_x_z_ * err_z_);

    // x_r_x_ = ode45_simulator_o2_(x_r_x_, A_x_, B_x_x_, B_u_, err_x_, u_all_x_);
    // x_r_y_ = ode45_simulator_o2_(x_r_y_, A_y_, B_x_y_, B_u_, err_y_, u_all_y_);
    // x_r_z_ = ode45_simulator_o2_(x_r_z_, A_z_, B_x_z_, B_u_, err_z_, u_all_z_);

    if (!init_x_r_)
    {
        x_r_x_ << 0, err_x_;
        x_r_y_ << 0, err_y_;
        x_r_z_ << 0, err_z_;
        init_x_r_ = true;
        std::cout << "update only once " << std::endl;
    }

    urc_x_ = F_param_[0] * G_param_[0] * (x_r_x_[1] - err_x_);
    urc_y_ = F_param_[1] * G_param_[1] * (x_r_y_[1] - err_y_);
    urc_z_ = F_param_[2] * G_param_[2] * (x_r_z_[1] - err_z_);

    u_all_x_ = unc_x + urc_x_;
    u_all_y_ = unc_y + urc_y_;
    u_all_z_ = unc_z + urc_z_;

    ctrl_accel_.vector.x = u_all_x_;
    ctrl_accel_.vector.y = u_all_y_;
    ctrl_accel_.vector.z = u_all_z_;
}



void uav_controller::cal_robust_ctrl_xy_()
{
    err_x_ = cur_pos_.pose.position.x - exp_pos_.pose.position.x;
    err_y_ = cur_pos_.pose.position.y - exp_pos_.pose.position.y;
    err_z_ = cur_pos_.pose.position.z - exp_pos_.pose.position.z;
    float err_vx = cur_vel_.twist.linear.x - exp_vel_.twist.linear.x;
    float err_vy = cur_vel_.twist.linear.y - exp_vel_.twist.linear.y;
    float err_vz = cur_vel_.twist.linear.z - exp_vel_.twist.linear.z;

    float unc_x = P_param_[0] * err_x_ + D_param_[0] * err_vx;
    float unc_y = P_param_[1] * err_y_ + D_param_[1] * err_vy;
    float unc_z = P_param_[2] * err_z_ + D_param_[2] * err_vz; // + i_z * integration_z;

    x_r_x_ = helper_func.rk45_o2(x_r_x_, A_x_, B_u_ * u_all_x_ + B_x_x_ * err_x_);
    x_r_y_ = helper_func.rk45_o2(x_r_y_, A_y_, B_u_ * u_all_y_ + B_x_y_ * err_y_);

    // x_r_x_ = ode45_simulator_o2_(x_r_x_, A_x_, B_x_x_, B_u_, err_x_, u_all_x_);
    // x_r_y_ = ode45_simulator_o2_(x_r_y_, A_y_, B_x_y_, B_u_, err_y_, u_all_y_);
    // x_r_z_ = ode45_simulator_o2_(x_r_z_, A_z_, B_x_z_, B_u_, err_z_, u_all_z_);

    if (!init_x_r_)
    {
        x_r_x_ << 0, err_x_;
        x_r_y_ << 0, err_y_;
        init_x_r_ = true;
        std::cout << "update only once " << std::endl;
    }

    urc_x_ = F_param_[0] * G_param_[0] * (x_r_x_[1] - err_x_);
    urc_y_ = F_param_[1] * G_param_[1] * (x_r_y_[1] - err_y_);

    u_all_x_ = unc_x + urc_x_;
    u_all_y_ = unc_y + urc_y_;
    u_all_z_ = unc_z;

    ctrl_accel_.vector.x = u_all_x_;
    ctrl_accel_.vector.y = u_all_y_;
    ctrl_accel_.vector.z = u_all_z_;
}