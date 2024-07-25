#!/usr/bin/env python

import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
import time

from scipy.spatial.transform import Rotation as R
import numpy as np

def quaternion2euler(q):
    r = R.from_quat(q)
    euler = r.as_euler('xyz', degrees=False)
    return euler

class LowPassFilter3D:
    def __init__(self, cutoff_frequency=30, initial_value=None):
        self.cutoff_frequency = cutoff_frequency
        if initial_value is None:
            self.filtered_value = np.zeros(3)
        else:
            self.filtered_value = np.array(initial_value)
    
    def __call__(self, x, dt):
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * self.cutoff_frequency))
        print(alpha)
        self.filtered_value = alpha * np.array(x) + (1.0 - alpha) * self.filtered_value
        return self.filtered_value

class VrpnProcessor:
    def __init__(self):
        # 初始化节点
        rospy.init_node('vrpn_processor', anonymous=True)

        # 订阅两个话题
        self.twist_sub = rospy.Subscriber('/vrpn_client_node/drone_7/twist', TwistStamped, self.twist_callback)
        self.pose_sub = rospy.Subscriber('/vrpn_client_node/drone_7/pose', PoseStamped, self.pose_callback)

        # 发布新话题
        self.pub = rospy.Publisher('/processed_data', Float64MultiArray, queue_size=10)

        # 初始化存储数据的变量
        self.pose_data = None
        self.twist_data = None

        # 初始化频率相关变量
        self.stamp = rospy.Time.now()
        self.last_stamp = rospy.Time.now()
        self._last_stamp = rospy.Time.now()

        self.message_count = 0
        self.frequency = 0

        self.previous_euler = np.array([0., 0., 0.])

        # lowpass filter
        self.filter = LowPassFilter3D(cutoff_frequency=2)

    def twist_callback(self, data):
        self.twist_data = data

    def pose_callback(self, data):
        self.pose_data = data
        self.stamp = data.header.stamp
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
            q = [x, y, z, w]
            current_euler = quaternion2euler(q)

            # address yaw jumping
            adjusted_euler = current_euler.copy()
            for i in range(3):
                delta_angle = current_euler[i] - self.previous_euler[i]
                if delta_angle > np.pi:
                    adjusted_euler[i] -= 2 * np.pi
                elif delta_angle < -np.pi:
                    adjusted_euler[i] += 2 * np.pi

            dt = self.stamp.to_sec() - self.last_stamp.to_sec()
            current_euler = self.filter(adjusted_euler, dt)
            wx, wy, wz = (current_euler - self.previous_euler) / 0.01

            self.previous_euler = adjusted_euler
            self.last_stamp = self.stamp
            # print(adjusted_euler)
            # print([wx, wy, wz])

            # 生成13维向量
            processed_data = Float64MultiArray()
            processed_data.data = [px, py, pz, x, y, z, w, vx, vy, vz, current_euler[0], current_euler[1], current_euler[2]]
            # processed_data.data = [px, py, pz, x, y, z, w, vx, vy, vz, wx, wy, wz]

            # 发布数据
            self.pub.publish(processed_data)

            # 更新频率检测
            self.message_count += 1
            if (self.stamp.to_sec() - self._last_stamp.to_sec()) >= 1.0:  # 每秒计算一次频率
                self.frequency = self.message_count / (self.stamp.to_sec() - self._last_stamp.to_sec())
                rospy.loginfo(f"VRPN state frequency: {self.frequency} Hz")
                self.message_count = 0
                self._last_stamp = self.stamp

if __name__ == '__main__':
    try:
        processor = VrpnProcessor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
