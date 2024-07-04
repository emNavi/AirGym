#ifndef _REGULAR_MOTION_HPP_
#define _REGULAR_MOTION_HPP_
#include <ros/ros.h>
#include <math.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

#include <nav_msgs/Odometry.h>

// 这个hpp中，Moving_along_circle和Capture的get函数返回值为 单步 的 期望
// 其他三个类的 line square curve 的函数返回值为 std::vector 期望的列表
class Moving_along_line
{
public:
    Moving_along_line(double half_length, double moving_velocity, double height, double dt)
        : R_(half_length),
          v_(moving_velocity),
          height_(height),
          dt_(dt) {}

    inline void init_path()
    {
        double p1[2] = {0, R_};
        double p2[2] = {0, -R_};

        generate_line_path_(p1, p2);
        generate_line_path_(p2, p1);

        reshape_();
    }
    inline std::vector<geometry_msgs::PoseStamped> get_pos_list()
    {
        return reshaped_pos_list_;
    }
    inline std::vector<geometry_msgs::TwistStamped> get_vel_list()
    {
        return reshaped_vel_list_;
    }

protected:
    double R_, dt_, v_, height_;

    std::vector<geometry_msgs::PoseStamped> pos_list_;
    std::vector<geometry_msgs::TwistStamped> vel_list_;

    std::vector<geometry_msgs::PoseStamped> reshaped_pos_list_;
    std::vector<geometry_msgs::TwistStamped> reshaped_vel_list_;

    void generate_line_path_(double *start_point, double *end_point);
    // reshape the start point of path to (R, 0)
    void reshape_();
};

class Moving_along_circle
{

public:
    Moving_along_circle(double radius, double omega, double height)
        : radius_(radius),
          omega_(omega),
          height_(height) {}

    inline void set_phi(double phi)
    {
        phi_ = phi;
    };

    geometry_msgs::PoseStamped get_exp_pos();

    geometry_msgs::TwistStamped get_exp_vel();

    geometry_msgs::Vector3Stamped get_exp_acc();

protected:
    double phi_;
    double radius_, height_, omega_;
    geometry_msgs::PoseStamped exp_pos_;
    geometry_msgs::TwistStamped exp_vel_;
    geometry_msgs::Vector3Stamped exp_acc_;
};

class Moving_along_square : public Moving_along_line
{
public:
    Moving_along_square(double half_length, double moving_velocity, double height, double dt)
        : Moving_along_line(half_length, moving_velocity, height, dt) {}
    inline void init_path()
    {
        double p1[2] = {R_, R_};
        double p2[2] = {-R_, R_};
        double p3[2] = {-R_, -R_};
        double p4[2] = {R_, -R_};

        generate_line_path_(p1, p2);
        generate_line_path_(p2, p3);
        generate_line_path_(p3, p4);
        generate_line_path_(p4, p1);

        reshape_();
    }

private:
};

class Moving_along_random_center_circle : public Moving_along_circle
{
public:
    Moving_along_random_center_circle(double radius, double omega, double height, double c_x, double c_y)
        : Moving_along_circle(radius, omega, height),
          c_x_(c_x),
          c_y_(c_y) {}
    geometry_msgs::PoseStamped get_exp_pos();

protected:
    double c_x_, c_y_;
};

class Moving_along_curve
{
public:
    Moving_along_curve(double radius, double omega, double height, double dt, double angle_range, double *c_1, double *c_2)
        : dt_(dt), angle_range_(angle_range), radius_(radius), omega_(omega), height_(height),
          m_r_c_1_(radius, omega, height, c_1[0], c_1[1]),
          m_r_c_2_(radius, omega, height, c_2[0], c_2[1])
    {
        init_path_();
    }

    inline std::vector<geometry_msgs::PoseStamped> get_pos_list()
    {
        return pos_list_;
    }

    inline std::vector<geometry_msgs::TwistStamped> get_vel_list()
    {
        return vel_list_;
    }
    inline std::vector<geometry_msgs::Vector3Stamped> get_acc_list()
    {
        return acc_list_;
    }

private:
    double radius_, omega_, height_, dt_, angle_range_;

    std::vector<geometry_msgs::PoseStamped> pos_list_;
    std::vector<geometry_msgs::TwistStamped> vel_list_;
    std::vector<geometry_msgs::Vector3Stamped> acc_list_;

    Moving_along_random_center_circle m_r_c_1_;
    Moving_along_random_center_circle m_r_c_2_;
    void init_path_();
};

class Capture : public Moving_along_circle
{

public:
    Capture(double radius, double omega, double height)
        : Moving_along_circle(radius, omega, height)
    {
        zero_vel_.twist.linear.x = 0;
        zero_vel_.twist.linear.y = 0;
        zero_vel_.twist.linear.z = 0;
        zero_acc_.vector.x = 0;
        zero_acc_.vector.y = 0;
        zero_acc_.vector.z = 0;
    }
    inline void set_leader_state(nav_msgs::Odometry leader_state)
    {
        leader_state_ = leader_state;
    }
    inline geometry_msgs::TwistStamped get_zero_vel()
    {
        return zero_vel_;
    }
    inline geometry_msgs::Vector3Stamped get_zero_acc()
    {
        return zero_acc_;
    }
    geometry_msgs::PoseStamped get_exp_pos_with_leader();
    geometry_msgs::TwistStamped get_trac_exp_vel_with_leader();
    geometry_msgs::TwistStamped get_cap_exp_vel_with_leader();

private:
    nav_msgs::Odometry leader_state_;
    geometry_msgs::Vector3Stamped leader_acc_;
    geometry_msgs::TwistStamped zero_vel_;
    geometry_msgs::Vector3Stamped zero_acc_;
};

#endif