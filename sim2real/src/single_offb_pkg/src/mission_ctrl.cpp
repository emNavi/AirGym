#include <ros/ros.h>
#include <mavros_msgs/PositionTarget.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int8.h>

#define ROS_RATE 50.0

nav_msgs::Odometry target_odom;
void target_odom_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    if (msg->child_frame_id == "car")
    {
        target_odom = *msg;
    }
}

nav_msgs::Odometry cur_odom;
void self_odom_cb(const nav_msgs::Odometry::ConstPtr &msg)
{
    cur_odom = *msg;
}

int mission_num;
void mission_cb(const std_msgs::Int8::ConstPtr &msg)
{
    mission_num = msg->data;
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "track_cap_mission_node");
    ros::NodeHandle nh("~");
    ros::Rate rate(ROS_RATE);

    ros::Subscriber target_state_sub = nh.subscribe<nav_msgs::Odometry>("/car_state", 10, target_odom_cb);
    ros::Subscriber self_state_sub = nh.subscribe<nav_msgs::Odometry>("/mavros/local_position/odom", 10, self_odom_cb);

    ros::Publisher pva_yaw_pub = nh.advertise<quadrotor_msgs::PositionCommand>("pos_cmd", 100);

    int drone_id;
    nh.param("drone_id", drone_id, 255);
    std::cout << "drone_id is " << drone_id << std::endl;
    int num_of_uav;
    nh.param("num_of_uav", num_of_uav, 255);

    double height;
    nh.param("height", height, 1.0);

    double radius;
    nh.param("radius", radius, 1.0);
    std::cout << "r is " << radius << std::endl;

    double phi;
    double omega = M_PI / 16;

    int count = 0;
    int k = 0;
    nav_msgs::Odometry exp_odom;

    quadrotor_msgs::PositionCommand pva_yaw;
    while (true)
    {
        phi = omega * k / ROS_RATE + (drone_id - 1) / num_of_uav;

        exp_odom.pose.pose.position.x = radius * cos(phi);
        exp_odom.pose.pose.position.y = radius * sin(phi);
        exp_odom.pose.pose.position.x = height;

        if (abs(cur_odom.pose.pose.position.z - height) > 0.3 ||
            abs(cur_odom.pose.pose.position.x - (exp_odom.pose.pose.position.x + target_odom.pose.pose.position.x)) > 0.3 ||
            abs(cur_odom.pose.pose.position.y - (exp_odom.pose.pose.position.y + target_odom.pose.pose.position.y)) > 0.1)
        {

            pva_yaw.position.x = target_odom.pose.pose.position.x + exp_odom.pose.pose.position.x;
            pva_yaw.position.y = target_odom.pose.pose.position.y + exp_odom.pose.pose.position.y;
            pva_yaw.position.z = height;

            pva_yaw.velocity.x = 0;
            pva_yaw.velocity.y = 0;
            pva_yaw.velocity.z = 0;

            pva_yaw.acceleration.x = 0;
            pva_yaw.acceleration.y = 0;
            pva_yaw.acceleration.z = 0;
        }
        else
        {
            pva_yaw.position.x = target_odom.pose.pose.position.x + exp_odom.pose.pose.position.x;
            pva_yaw.position.y = target_odom.pose.pose.position.y + exp_odom.pose.pose.position.y;
            pva_yaw.position.z = height;

            pva_yaw.velocity.x = target_odom.pose.pose.position.x;
            pva_yaw.velocity.y = target_odom.pose.pose.position.y;
            pva_yaw.velocity.z = 0;

            pva_yaw.acceleration.x = 0;
            pva_yaw.acceleration.y = 0;
            pva_yaw.acceleration.z = 0;
        } 
        
        pva_yaw_pub.publish(pva_yaw);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}