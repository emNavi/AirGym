#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
import time

class VrpnProcessor:
    def __init__(self):
        # 初始化节点
        rospy.init_node('vrpn_processor', anonymous=True)

        # 订阅两个话题
        self.twist_sub = rospy.Subscriber('/vrpn_client_node/emnavi_t/twist', TwistStamped, self.twist_callback)
        self.pose_sub = rospy.Subscriber('/vrpn_client_node/emnavi_t/pose', PoseStamped, self.pose_callback)

        # 发布新话题
        self.pub = rospy.Publisher('/processed_data', Float64MultiArray, queue_size=10)

        # 初始化存储数据的变量
        self.pose_data = None
        self.twist_data = None

        # 初始化频率相关变量
        self.last_time = time.time()
        self.message_count = 0
        self.frequency = 0

    def twist_callback(self, data):
        self.twist_data = data
        self.process_and_publish()

    def pose_callback(self, data):
        self.pose_data = data
        self.process_and_publish()

    def process_and_publish(self):
        if self.pose_data is not None and self.twist_data is not None:
            # 提取位置信息
            px = self.pose_data.pose.position.x / 1000
            py = self.pose_data.pose.position.y / 1000
            pz = self.pose_data.pose.position.z / 1000

            # 提取四元数
            x = self.pose_data.pose.orientation.x
            y = self.pose_data.pose.orientation.y
            z = self.pose_data.pose.orientation.z
            w = self.pose_data.pose.orientation.w

            # 提取速度信息
            vx = self.twist_data.twist.linear.x / 1000
            vy = self.twist_data.twist.linear.y / 1000
            vz = self.twist_data.twist.linear.z / 1000

            # 提取角速度信息
            wx = self.twist_data.twist.angular.x
            wy = self.twist_data.twist.angular.y
            wz = self.twist_data.twist.angular.z

            # 生成13维向量
            processed_data = Float64MultiArray()
            processed_data.data = [px, py, pz, x, y, z, w, vx, vy, vz, wx, wy, wz]

            # 发布数据
            self.pub.publish(processed_data)

            # 更新频率检测
            self.message_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:  # 每秒计算一次频率
                self.frequency = self.message_count / (current_time - self.last_time)
                rospy.loginfo(f"VRPN state frequency: {self.frequency} Hz")
                self.message_count = 0
                self.last_time = current_time

if __name__ == '__main__':
    try:
        processor = VrpnProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
