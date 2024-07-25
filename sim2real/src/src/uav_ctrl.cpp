/**
 * @file offb_node.cpp
 * @brief Offboard control example node, written with MAVROS version 0.19.x, PX4 Pro Flight
 * Stack and tested in Gazebo Classic SITL
 */
#include <ros/ros.h>

#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/PositionTarget.h>


#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <std_msgs/Float32MultiArray.h>

#include "single_offb_pkg/FSM.hpp"
#include "single_offb_pkg/regular_motion.hpp"

#define ROS_RATE 50.0
#define TAKEOFF_HEIGHT 1.2

mavros_msgs::CommandBool arm_cmd;
ros::ServiceClient set_mode_client, arming_client;
ros::Publisher local_accel_pub, local_pos_pub, err_pub;
CtrlFSM fsm;


mavros_msgs::State current_state;
geometry_msgs::PoseStamped cur_pos;
geometry_msgs::TwistStamped cur_vel;
std_msgs::Float32MultiArray takeoff_cmd;
std_msgs::Float32MultiArray land_cmd;
ros::Publisher vision_pose_pub;

int wait_count = 0;


void publish_ctrl_msg(const mavros_msgs::PositionTarget::ConstPtr &msg)
{

        ROS_INFO("gym go");
        local_accel_pub.publish(msg);

    // mpCtrl.coordinate_frame = mpCtrl.FRAME_LOCAL_NED;

    // geometry_msgs::PoseStamped exp_pos;
    // exp_pos.pose.position = pva_yaw.position;
    // geometry_msgs::TwistStamped exp_vel;
    // exp_vel.twist.linear = pva_yaw.velocity;


    // mpCtrl.type_mask = mpCtrl.IGNORE_YAW_RATE;
    // mpCtrl.position = pva_yaw.position;
    // mpCtrl.velocity = pva_yaw.velocity;
    // mpCtrl.acceleration_or_force = pva_yaw.acceleration;
    // mpCtrl.yaw = pva_yaw.yaw;
}

// void publish_err_msg()
// {
//     std_msgs::Float32MultiArray err_msgs;
//     err_msgs.data = {static_cast<float>(cur_pos.pose.position.x - pva_yaw.position.x),
//                      static_cast<float>(cur_pos.pose.position.y - pva_yaw.position.y),
//                      static_cast<float>(cur_pos.pose.position.z - pva_yaw.position.z),
//                      static_cast<float>(cur_vel.twist.linear.x - pva_yaw.velocity.x),
//                      static_cast<float>(cur_vel.twist.linear.y - pva_yaw.velocity.y),
//                      static_cast<float>(cur_vel.twist.linear.z - pva_yaw.velocity.z)};
//     err_pub.publish(err_msgs);
// }

// callback begin************************
void state_cb(const mavros_msgs::State::ConstPtr &msg)
{
    current_state = *msg;
}

void pos_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    cur_pos = *msg;
}

void vel_cb(const geometry_msgs::TwistStamped::ConstPtr &msg)
{
    cur_vel = *msg;
}

void pva_yaw_cb(const mavros_msgs::PositionTarget::ConstPtr &msg)
{
    fsm.update_cmd_update_time(ros::Time::now());
    if (fsm.now_state == CtrlFSM::RUNNING)
    {
        publish_ctrl_msg(msg);

    }
}
void vrpn_cb(const geometry_msgs::PoseStamped::ConstPtr &msg)
{    // 创建一个新的 PoseStamped 消息
    geometry_msgs::PoseStamped modified_msg;
    modified_msg.header.stamp = ros::Time::now();
    modified_msg.header.frame_id = msg->header.frame_id;  // 保留原来的 frame_id

    // 对位置进行变换（例如，添加一个偏移量）
    modified_msg.pose.position.x = msg->pose.position.x/1000.0;  // 偏移量为 1.0 米
    modified_msg.pose.position.y = msg->pose.position.y/1000.0;
    modified_msg.pose.position.z = msg->pose.position.z/1000.0;

    // 对方向（四元数）进行变换（这里保持不变，仅作为示例）
    modified_msg.pose.orientation = msg->pose.orientation;
    vision_pose_pub.publish(modified_msg);
}

bool takeoff_cmd_flag = false;
void takeoff_cmd_cb(const std_msgs::Float32MultiArray::ConstPtr &msg, int drone_id)
{

    takeoff_cmd = *msg;
    // ROS_INFO_STREAM(takeoff_cmd.data[0] << takeoff_cmd.data[1] << drone_id);

    for (int i = 0; i < takeoff_cmd.data[0] + 1e-2; i++)
    {
        if (abs(takeoff_cmd.data[i + 1] - drone_id) < 1e-3)
        {
            takeoff_cmd_flag = true;
        }
    }
}

bool land_cmd_flag = false;
void land_cmd_cb(const std_msgs::Float32MultiArray::ConstPtr &msg, int drone_id)
{
    land_cmd = *msg;
    for (int i = 0; i < land_cmd.data[0]; i++)
    {
        if (abs(land_cmd.data[i + 1] - drone_id) < 1e-3)
        {
            land_cmd_flag = true;
        }
    }
}

// callback end************************
bool request_offboard()
{
    mavros_msgs::SetMode offb_set_mode;
    arm_cmd.request.value = true;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    if (set_mode_client.call(offb_set_mode) &&
        offb_set_mode.response.mode_sent)
    {
        ROS_INFO("Offboard enabled");
        return true;
    }
    return false;
}

bool request_arm()
{
    if (arming_client.call(arm_cmd) &&
        arm_cmd.response.success)
    {
        ROS_INFO("Vehicle armed");
        return true;
    }
    return false;
}

bool request_land()
{
    mavros_msgs::SetMode auto_land_mode;
    arm_cmd.request.value = true;
    auto_land_mode.request.custom_mode = "AUTO.LAND";

    if (set_mode_client.call(auto_land_mode) &&
        auto_land_mode.response.mode_sent)
    {
        ROS_INFO("Landing");
        return true;
    }
    return false;
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "uav_ctrl_node");
    ros::NodeHandle nh("~");

    std::string ctrl_mode;
    int drone_id;
    nh.param<int>("drone_id", drone_id, 99);

    nh.param<std::string>("ctrl_mode", ctrl_mode, "mpCtrl");
    std::cout << "ctrl_mode " << ctrl_mode << std::endl;
    std::cout << "drone id " << drone_id << std::endl;
    fsm.Init_FSM();
    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>("/mavros/state", 10, state_cb);
    ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", 10, pos_cb);
    ros::Subscriber vel_sub = nh.subscribe<geometry_msgs::TwistStamped>("/mavros/local_position/velocity_local", 10, vel_cb);
    ros::Subscriber pva_yaw_sub = nh.subscribe<mavros_msgs::PositionTarget>("pos_cmd", 10, pva_yaw_cb);
    ros::Subscriber takeoff_cmd_sub = nh.subscribe<std_msgs::Float32MultiArray>("/swarm_takeoff", 10, boost::bind(takeoff_cmd_cb, _1, drone_id));
    ros::Subscriber land_cmd_sub = nh.subscribe<std_msgs::Float32MultiArray>("/swarm_land", 10, boost::bind(land_cmd_cb, _1, drone_id));
    ros::Subscriber vrpn_pose = nh.subscribe<geometry_msgs::PoseStamped>("/vrpn_client_node/drone_7/pose", 10, vrpn_cb);

    arming_client = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    vision_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);

    local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);
    local_accel_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 10);
    err_pub = nh.advertise<std_msgs::Float32MultiArray>("/errors", 10);

    ros::Rate rate(ROS_RATE);

    // wait for FCU connection
    ROS_INFO("FCU connecting");
    while (ros::ok() && !current_state.connected)
    {
        ros::spinOnce();
        rate.sleep();
        std::cout<<">";
    }
    std::cout<<std::endl;
    ROS_INFO("FCU connected");

    // send a few setpoints before starting
    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = TAKEOFF_HEIGHT;
    for (int i = 100; ros::ok() && i > 0; --i)
    {
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    /* ------------------init params --------------------------------------------------*/
    ros::Time last_request = ros::Time::now();
    geometry_msgs::PoseStamped start_pose;
    /* ------------------init params done --------------------------------------------*/
    while (ros::ok())
    {
        fsm.process();
        if (fsm.now_state == CtrlFSM::INIT_PARAM)
        {
            if (!takeoff_cmd_flag)
            {
                ros::spinOnce();
                rate.sleep();
                continue;
            }
            if (current_state.mode != "OFFBOARD" &&
                (ros::Time::now() - fsm.last_try_offboard_time > ros::Duration(5.0)))
            {
                ROS_INFO("REQUEST OFFBOARD");
                if (!request_offboard())
                    ROS_WARN("offboard failed,try again after 5.0 second");
                fsm.last_try_offboard_time = ros::Time::now();
            }
            else if (current_state.mode == "OFFBOARD" && !current_state.armed &&
                     (ros::Time::now() - fsm.last_try_arm_time > ros::Duration(5.0)))
            {
                ROS_INFO("REQUEST ARM");
                if (!request_arm())
                    ROS_WARN("arm failed,try again after 5.0 second");
                fsm.last_try_arm_time = ros::Time::now();
            }
            if (current_state.mode == "OFFBOARD")
            {
                fsm.set_offboard_flag(true);
            }

            if (current_state.mode == "OFFBOARD" && current_state.armed)
            {
                ROS_INFO("SET ARM");
                fsm.set_arm_flag(true);
            }
            local_pos_pub.publish(pose);
        }
        else if (fsm.now_state == CtrlFSM::TAKEOFF)
        {
            if (fsm.last_state != CtrlFSM::TAKEOFF)
            {
                ROS_INFO("MODE: TAKEOFF");
                start_pose.pose.position.x = cur_pos.pose.position.x;
                start_pose.pose.position.y = cur_pos.pose.position.y;
                start_pose.pose.position.z = TAKEOFF_HEIGHT;
            }
            pose.pose = start_pose.pose;
            local_pos_pub.publish(pose);

            if (abs(cur_pos.pose.position.z - TAKEOFF_HEIGHT) < 0.1)
            {
                wait_count ++ ;
                if(wait_count > 300)
                {
                fsm.set_takeoff_over_flag(true);
                ROS_INFO("Take off done");
                }
            }
        }
        else if (fsm.now_state == CtrlFSM::HOVER)
        {
            if (fsm.last_state != CtrlFSM::HOVER)
            {
                ROS_INFO("MODE: HOVER");

                start_pose.pose.position.x = cur_pos.pose.position.x;
                start_pose.pose.position.y = cur_pos.pose.position.y;
                start_pose.pose.position.z = cur_pos.pose.position.z;
                start_pose.pose.orientation = cur_pos.pose.orientation;
            }
            pose.pose = start_pose.pose;
            local_pos_pub.publish(pose);
        }
        else if (fsm.now_state == CtrlFSM::RUNNING)
        {
            if (fsm.last_state != CtrlFSM::RUNNING)
            {
                ROS_INFO("MODE: RUNNING");
                std::cout << "ctrl mode is " << ctrl_mode << std::endl;
            }
            // std::cout << "running ego planner" << std::endl;
            // publish_ctrl_msg(ctrl_mode);
            // publish_err_msg();
        }

        else if (fsm.now_state == CtrlFSM::LANDING)
        {
            if (fsm.last_state != CtrlFSM::LANDING)
            {
                ROS_INFO("MODE: LAND");
            }
            if (current_state.mode != "AUTO.LAND" &&
                (ros::Time::now() - fsm.last_try_offboard_time > ros::Duration(5.0)))
            {
                if (!request_land())
                    ROS_WARN("Try land cmd failed, pls try again in 5 seconds");
                fsm.last_try_offboard_time = ros::Time::now();
            }
        }
        if (land_cmd_flag)
        {
            fsm.set_land_flag(true);
        }
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
