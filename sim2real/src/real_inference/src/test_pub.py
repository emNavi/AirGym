#!/home/emnavi/miniconda3/envs/inference/bin/python

import rospy
import random
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance, Point, Quaternion, Vector3

def random_state_gen():
    pose = Pose(Point(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)), 
                Quaternion(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
    twist = Twist(Vector3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)), 
                  Vector3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
    # pose = Pose(Point(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1)), 
    #             Quaternion(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1)))
    # twist = Twist(Vector3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1)), 
    #               Vector3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1)))
    
    odom = Odometry()
    odom.header.stamp = rospy.Time.now()
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"
    odom.pose = PoseWithCovariance(pose, [0]*36)
    odom.twist = TwistWithCovariance(twist, [0]*36)

    return odom

def talker():
    # Set the node name
    rospy.init_node('test_pub', anonymous=True)
    
    # Create the publisher
    pub = rospy.Publisher('/random_states', Odometry, queue_size=10)

    # Set the rate for publishing messages
    rate = rospy.Rate(1)  # 10hz
    
    # Publish the message
    while not rospy.is_shutdown():
        odom = random_state_gen()
        # rospy.loginfo(odom)
        pub.publish(odom)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass