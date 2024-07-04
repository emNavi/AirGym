#include "vechicle_controller/regular_motion.h"
void Moving_along_line::generate_line_path_(double *start_point, double *end_point)
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
void Moving_along_line::reshape_()
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

geometry_msgs::PoseStamped Moving_along_circle::get_exp_pos()
{
    geometry_msgs::PoseStamped exp_pos;
    exp_pos.pose.position.x = radius_ * cos(phi_);
    exp_pos.pose.position.y = radius_ * sin(phi_);
    exp_pos.pose.position.z = height_;
    return exp_pos;
}

geometry_msgs::TwistStamped Moving_along_circle::get_exp_vel()
{
    geometry_msgs::TwistStamped exp_vel;
    exp_vel.twist.linear.x = -omega_ * radius_ * sin(phi_);
    exp_vel.twist.linear.y = omega_ * radius_ * cos(phi_);
    exp_vel.twist.linear.z = 0;
    return exp_vel;
}

geometry_msgs::Vector3Stamped Moving_along_circle::get_exp_acc()
{
    geometry_msgs::Vector3Stamped exp_acc;
    exp_acc.vector.x = -omega_ * omega_ * radius_ * cos(phi_);
    exp_acc.vector.y = -omega_ * omega_ * radius_ * sin(phi_);
    exp_acc.vector.z = 0;
    return exp_acc;
}

geometry_msgs::PoseStamped Moving_along_random_center_circle::get_exp_pos()
{
    geometry_msgs::PoseStamped exp_pos;
    exp_pos.pose.position.x = radius_ * cos(phi_) + c_x_;
    exp_pos.pose.position.y = radius_ * sin(phi_) + c_y_;
    exp_pos.pose.position.z = height_;
    return exp_pos;
}

void Moving_along_curve::init_path_()
{
    int nums = static_cast<int>(angle_range_ * 2 / (omega_ * dt_));
    geometry_msgs::PoseStamped pos;
    geometry_msgs::TwistStamped vel;
    geometry_msgs::Vector3Stamped acc;
    for (int i = 0; i < nums; i++)
    {
        m_r_c_1_.set_phi(2 * angle_range_ + omega_ * dt_ * i);
        pos = m_r_c_1_.get_exp_pos();
        vel = m_r_c_1_.get_exp_vel();
        acc = m_r_c_1_.get_exp_acc();

        pos_list_.emplace_back(pos);
        vel_list_.emplace_back(vel);
        acc_list_.emplace_back(acc);
    }
    for (int i = 0; i < nums; i++)
    {
        m_r_c_2_.set_phi(-angle_range_ + omega_ * dt_ * i);
        pos = m_r_c_2_.get_exp_pos();
        vel = m_r_c_2_.get_exp_vel();
        acc = m_r_c_2_.get_exp_acc();

        pos_list_.emplace_back(pos);
        vel_list_.emplace_back(vel);
        acc_list_.emplace_back(acc);
    }
}

geometry_msgs::PoseStamped Capture::get_exp_pos_with_leader()
{
    geometry_msgs::PoseStamped exp_pos;
    exp_pos.pose.position.x = get_exp_pos().pose.position.x + leader_state_.pose.pose.position.x;
    exp_pos.pose.position.y = get_exp_pos().pose.position.y + leader_state_.pose.pose.position.y;
    exp_pos.pose.position.z = get_exp_pos().pose.position.z + leader_state_.pose.pose.position.z;

    return exp_pos;
}
geometry_msgs::TwistStamped Capture::get_trac_exp_vel_with_leader()
{
    geometry_msgs::TwistStamped exp_vel;
    exp_vel.twist.linear.x = get_zero_vel().twist.linear.x + leader_state_.twist.twist.linear.x;
    exp_vel.twist.linear.y = get_zero_vel().twist.linear.y + leader_state_.twist.twist.linear.y;
    exp_vel.twist.linear.z = get_zero_vel().twist.linear.z;
    return exp_vel;
};
geometry_msgs::TwistStamped Capture::get_cap_exp_vel_with_leader()
{
    geometry_msgs::TwistStamped exp_vel;
    exp_vel.twist.linear.x = get_exp_vel().twist.linear.x + leader_state_.twist.twist.linear.x;
    exp_vel.twist.linear.y = get_exp_vel().twist.linear.y + leader_state_.twist.twist.linear.y;
    exp_vel.twist.linear.z = get_exp_vel().twist.linear.z;
    return exp_vel;
};
