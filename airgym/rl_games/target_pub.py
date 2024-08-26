#!/home/emnavi/miniconda3/envs/inference/bin/python

import rospy
import random
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

def target_pub():
    # Set the node name
    rospy.init_node('target_pub', anonymous=True)
    
    # Create the publisher
    pub = rospy.Publisher('/target_state', Float64MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10Hz

    # Set target state
    state = Float64MultiArray()
    # matrix(0-9);pos(9-12);vel(12-15);rate(15-18)
    state.data = [1,0,0,0,1,0,0,0,1,  0,0,1,  0,0,0,0,0,0]
    
    # Publish the message
    while not rospy.is_shutdown():
        rospy.loginfo(state.data)
        pub.publish(state)
        rate.sleep()

if __name__ == '__main__':
    try:
        target_pub()
    except rospy.ROSInterruptException:
        pass