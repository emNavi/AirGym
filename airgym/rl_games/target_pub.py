#!/home/emnavi/miniconda3/envs/inference/bin/python
import os
import signal
import subprocess
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
    state.data = [1,0,0,0,1,0,0,0,1,  1,0,1,  0,0,0,0,0,0]
    
    # Publish the message
    while not rospy.is_shutdown():
        rospy.loginfo(state.data)
        pub.publish(state)
        rate.sleep()

def kill_roscore():
    try:
        # Get the list of running processes
        output = subprocess.check_output(['ps', 'aux'], universal_newlines=True)
        # Look for roscore and rosmaster processes
        for line in output.splitlines():
            if 'roscore' in line or 'rosmaster' in line:
                # Extract the process ID (PID)
                pid = int(line.split()[1])
                # Kill the process
                os.kill(pid, signal.SIGTERM)
                print(f"Killed process {pid}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    try:
        # kill_roscore()
        # import subprocess
        # cmd = subprocess.Popen(['/bin/bash', '-i', '-c', 'roscore'], start_new_session=True)
        target_pub()
    except rospy.ROSInterruptException:
        pass