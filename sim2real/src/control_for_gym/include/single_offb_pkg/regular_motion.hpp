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

    inline std::vector<geometry_msgs::PoseStamped> get_pos_list()
    {
        return reshaped_pos_list_;
    }
    inline std::vector<geometry_msgs::TwistStamped> get_vel_list()
    {
        return reshaped_vel_list_;
    }

    inline void init_path()
    {
        double p1[2] = {R_, 0};
        double p2[2] = {-R_, 0};

        generate_line_path_(p1, p2);
        generate_line_path_(p2, p1);
        reshape_();
    }

protected:
    double R_, dt_, v_, height_;

    std::vector<geometry_msgs::PoseStamped> pos_list_;
    std::vector<geometry_msgs::TwistStamped> vel_list_;

    std::vector<geometry_msgs::PoseStamped> reshaped_pos_list_;
    std::vector<geometry_msgs::TwistStamped> reshaped_vel_list_;

    inline void generate_line_path_(double *start_point, double *end_point)
    {
        geometry_msgs::PoseStamped pos;
        geometry_msgs::TwistStamped vel;

        for (int i = 0; i < static_cast<int>((2 * R_ / v_) / dt_); i++)
        {
            pos.pose.position.x = start_point[0] + i * (end_point[0] - start_point[0]) / (2 * R_) * (dt_ * v_);
            pos.pose.position.y = start_point[1] + i * (end_point[1] - start_point[1]) / (2 * R_) * (dt_ * v_);
            pos.pose.position.z = height_;
            vel.twist.linear.x = (end_point[0] - start_point[0]) / (2 * R_) * v_;
            vel.twist.linear.y = (end_point[1] - start_point[1]) / (2 * R_) * v_;
            vel.twist.linear.z = 0;
            pos_list_.emplace_back(pos);
            vel_list_.emplace_back(vel);
        }
    }
    // reshape the start point of path to (R, 0)
    inline void reshape_()
    {
        int nums = static_cast<int>(R_ / v_ / dt_);
        for (int i = pos_list_.size() - nums; i < pos_list_.size(); i++)
        {
            reshaped_pos_list_.emplace_back(pos_list_[i]);
            reshaped_vel_list_.emplace_back(vel_list_[i]);
        }
        for (int i = 0; i < pos_list_.size() - nums; i++)
        {
            reshaped_pos_list_.emplace_back(pos_list_[i]);
            reshaped_vel_list_.emplace_back(vel_list_[i]);
        }
    }
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
    inline void set_dt(double dt)
    {

        dt_ = dt;
    }
    inline void set_phi_with_k(int k)
    {
        phi_ = omega_ * dt_ * k;
    }
    inline geometry_msgs::PoseStamped get_exp_pos()
    {
        geometry_msgs::PoseStamped exp_pos;
        exp_pos.pose.position.x = radius_ * cos(phi_);
        exp_pos.pose.position.y = radius_ * sin(phi_);
        exp_pos.pose.position.z = height_;
        return exp_pos;
    };

    inline geometry_msgs::TwistStamped get_exp_vel()
    {
        geometry_msgs::TwistStamped exp_vel;
        exp_vel.twist.linear.x = -omega_ * radius_ * sin(phi_);
        exp_vel.twist.linear.y = omega_ * radius_ * cos(phi_);
        exp_vel.twist.linear.z = 0;
        return exp_vel;
    };

    inline geometry_msgs::Vector3Stamped get_exp_acc()
    {
        geometry_msgs::Vector3Stamped exp_acc;
        exp_acc.vector.x = -omega_ * omega_ * radius_ * cos(phi_);
        exp_acc.vector.y = -omega_ * omega_ * radius_ * sin(phi_);
        exp_acc.vector.z = 0;
        return exp_acc;
    }

protected:
    double phi_, dt_;
    double radius_, height_, omega_;
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
};

class Moving_along_random_center_circle : public Moving_along_circle
{
public:
    Moving_along_random_center_circle(double radius, double omega, double height, double c_x, double c_y)
        : Moving_along_circle(radius, omega, height),
          c_x_(c_x),
          c_y_(c_y) {}
    inline geometry_msgs::PoseStamped get_exp_pos()
    {
        geometry_msgs::PoseStamped exp_pos;

        exp_pos.pose.position.x = radius_ * cos(phi_) + c_x_;
        exp_pos.pose.position.y = radius_ * sin(phi_) + c_y_;
        exp_pos.pose.position.z = height_;
        return exp_pos;
    };

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
        return reshaped_pos_list_;
    }

    inline std::vector<geometry_msgs::TwistStamped> get_vel_list()
    {
        return reshaped_vel_list_;
    }
    inline std::vector<geometry_msgs::Vector3Stamped> get_acc_list()
    {
        return reshaped_acc_list_;
    }

private:
    double radius_, omega_, height_, dt_, angle_range_;

    std::vector<geometry_msgs::PoseStamped> pos_list_;
    std::vector<geometry_msgs::TwistStamped> vel_list_;
    std::vector<geometry_msgs::Vector3Stamped> acc_list_;

    std::vector<geometry_msgs::PoseStamped> reshaped_pos_list_;
    std::vector<geometry_msgs::TwistStamped> reshaped_vel_list_;
    std::vector<geometry_msgs::Vector3Stamped> reshaped_acc_list_;

    Moving_along_random_center_circle m_r_c_1_;
    Moving_along_random_center_circle m_r_c_2_;

    inline void init_path_()
    {
        int nums = static_cast<int>(angle_range_ * 2 / (omega_ * dt_));
        geometry_msgs::PoseStamped pos;
        geometry_msgs::TwistStamped vel;
        geometry_msgs::Vector3Stamped acc;
        for (int i = 0; i < nums; i++)
        {
            m_r_c_1_.set_phi(angle_range_ / 2 + omega_ * dt_ * i);
            pos = m_r_c_1_.get_exp_pos();
            vel = m_r_c_1_.get_exp_vel();
            acc = m_r_c_1_.get_exp_acc();

            pos_list_.emplace_back(pos);
            vel_list_.emplace_back(vel);
            acc_list_.emplace_back(acc);
        }
        for (int i = 0; i < nums; i++)
        {
            m_r_c_2_.set_phi(M_PI + angle_range_ / 2 + omega_ * dt_ * i);
            pos = m_r_c_2_.get_exp_pos();
            vel = m_r_c_2_.get_exp_vel();
            acc = m_r_c_2_.get_exp_acc();

            pos_list_.emplace_back(pos);
            vel_list_.emplace_back(vel);
            acc_list_.emplace_back(acc);
        }
        reshape_();
    }

    // reshape but do nothing
    inline void reshape_()
    {
        for (auto col : pos_list_)
        {
            reshaped_pos_list_.emplace_back(col);
        }
        for (auto col : vel_list_)
        {
            reshaped_vel_list_.emplace_back(col);
        }
        for (auto col : acc_list_)
        {
            reshaped_acc_list_.emplace_back(col);
        }
    }
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
    inline geometry_msgs::PoseStamped get_exp_pos_with_leader()
    {
        geometry_msgs::PoseStamped exp_pos;
        exp_pos.pose.position.x = get_exp_pos().pose.position.x + leader_state_.pose.pose.position.x;
        exp_pos.pose.position.y = get_exp_pos().pose.position.y + leader_state_.pose.pose.position.y;
        exp_pos.pose.position.z = get_exp_pos().pose.position.z + leader_state_.pose.pose.position.z;

        return exp_pos;
    }
    inline geometry_msgs::TwistStamped get_trac_exp_vel_with_leader()
    {
        geometry_msgs::TwistStamped exp_vel;
        exp_vel.twist.linear.x = get_zero_vel().twist.linear.x + leader_state_.twist.twist.linear.x;
        exp_vel.twist.linear.y = get_zero_vel().twist.linear.y + leader_state_.twist.twist.linear.y;
        exp_vel.twist.linear.z = get_zero_vel().twist.linear.z;
        return exp_vel;
    };
    inline geometry_msgs::TwistStamped get_cap_exp_vel_with_leader()
    {
        geometry_msgs::TwistStamped exp_vel;
        exp_vel.twist.linear.x = get_exp_vel().twist.linear.x + leader_state_.twist.twist.linear.x;
        exp_vel.twist.linear.y = get_exp_vel().twist.linear.y + leader_state_.twist.twist.linear.y;
        exp_vel.twist.linear.z = get_exp_vel().twist.linear.z;
        return exp_vel;
    };

private:
    nav_msgs::Odometry leader_state_;
    geometry_msgs::TwistStamped zero_vel_;
    geometry_msgs::Vector3Stamped zero_acc_;
};

#endif