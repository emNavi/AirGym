#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand

import yaml
import os
import sys

sync_cmd_msg = PositionCommand()
sync_odom_msg = Odometry()

# over_time_flag=False
def cmd_callback(msg):
    global sync_cmd_msg,sync_odom_msg
    print("odom")

    sync_cmd_msg = msg
    current_time = rospy.Time.now()
    sync_cmd_msg.header.stamp = current_time
    sync_odom_msg.header.stamp = current_time
    odom_pub.publish(sync_odom_msg)
    cmd_pub.publish(sync_cmd_msg)

def odom_callback(msg):
    global sync_cmd_msg,sync_odom_msg
    sync_odom_msg = msg




if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('save_traj_node', anonymous=True)
    with open("/home/qyswarm/param_files/real_use/drone_param.yaml", "r") as stream:
        try:
            dictionary = yaml.safe_load(stream)
            drone_id = dictionary["drone_id"]            
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)
    rospy.Subscriber("/quadrotor_control/odom", Odometry, odom_callback)
    rospy.Subscriber("/drone_{}_planning/pos_cmd".format(drone_id), PositionCommand, cmd_callback)
    odom_pub = rospy.Publisher("/record/odom", Odometry, queue_size=50)
    cmd_pub = rospy.Publisher("/record/cmd", PositionCommand, queue_size=50)
    rospy.spin()