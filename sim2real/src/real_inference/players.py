from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.players import PpoPlayerContinuous
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
import time

import rospy
from geometry_msgs.msg import *
from mavros_msgs.msg import *

class CpuPlayerContinuous(PpoPlayerContinuous):
    def __init__(self, params):
        super().__init__(params)

        # initialize
        rospy.init_node('onboard_computing_node', anonymous=True)

        self.obs_sub = rospy.Subscriber('', String, self.callback)
        ctl_mode = self.env_config.get('ctl_mode')
        if ctl_mode == 'pos':
            self.action_pub = rospy.Publisher('/airgym/cmd/pose', PoseStamped, queue_size=2000)
            # # ----- msgs type ----- #
            #     std_msgs/Header header
            #     uint32 seq
            #     time stamp
            #     string frame_id
            #     geometry_msgs/Pose pose
            #     geometry_msgs/Point position
            #         float64 x
            #         float64 y
            #         float64 z
            #     geometry_msgs/Quaternion orientation
            #         float64 x
            #         float64 y
            #         float64 z
            #         float64 w

            # Set the rate for publishing messages
            self.rate = rospy.Rate(50)  # 10hz
        elif ctl_mode == 'vel':
            self.action_pub = rospy.Publisher('/airgym/cmd/vel', Twist, queue_size=2000)
            # # ----- msgs type ----- #
            #     geometry_msgs/Vector3 linear
            #     float64 x
            #     float64 y
            #     float64 z
            #     geometry_msgs/Vector3 angular
            #     float64 x
            #     float64 y
            #     float64 z
            # Set the rate for publishing messages
            self.rate = rospy.Rate(50)  # 10hz
        elif ctl_mode == 'atti':
            self.action_pub = rospy.Publisher('/airgym/cmd/atti', PoseStamped, queue_size=2000)
            # # ----- msgs type ----- #
            #     std_msgs/Header header
            #     uint32 seq
            #     time stamp
            #     string frame_id
            #     geometry_msgs/Pose pose
            #     geometry_msgs/Point position
            #         float64 x
            #         float64 y
            #         float64 z
            #     geometry_msgs/Quaternion orientation
            #         float64 x
            #         float64 y
            #         float64 z
            #         float64 w
            # Set the rate for publishing messages
            self.rate = rospy.Rate(50)  # 10hz
        elif ctl_mode == 'rate':
            self.action_pub = rospy.Publisher('/airgym/cmd/rate', TwistStamped, queue_size=2000)
            # # ----- msgs type ----- #
            #     std_msgs/Header header
            #     uint32 seq
            #     time stamp
            #     string frame_id
            #     geometry_msgs/Twist twist
            #     geometry_msgs/Vector3 linear
            #         float64 x
            #         float64 y
            #         float64 z
            #     geometry_msgs/Vector3 angular
            #         float64 x
            #         float64 y
            #         float64 z
            # Set the rate for publishing messages
            self.rate = rospy.Rate(50)  # 10hz
        else: # prop
            self.action_pub = rospy.Publisher('/airgym/cmd/thrust', Thrust, queue_size=2000)
            # # ----- msgs type ----- #
            #     std_msgs/Header header
            #     uint32 seq
            #     time stamp
            #     string frame_id
            #     float32 thrust
            # Set the rate for publishing messages
            self.rate = rospy.Rate(50)  # 10hz

        # env settings
        self.has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None
        if has_masks_func:
            self.has_masks = self.env.has_action_mask()
        
        need_init_rnn = self.is_rnn
        if need_init_rnn:
            self.init_rnn()
            need_init_rnn = False

    def callback(self, data):
        # Process the incoming message
        rospy.loginfo(f"Obs reveived: {data.data}")
        
        real_obs = data.data
        action = self.inference(real_obs)

        # Create the output message
        output_msg = String()
        output_msg.data = f"Processed: {data.data}"
        
        # Publish the message
        self.publisher.publish(output_msg)

    def inference(self, real_obses):
        if self.has_masks:
            masks = self.env.get_action_mask()
            action = self.get_masked_action(
                real_obses, masks, self.is_deterministic)
        else:
            action = self.get_action(real_obses, self.is_deterministic)
        
        sim_obses, _, done, _ = self.env_step(self.env, action)

        if self.render_env:
            self.env.render(mode='human')
            time.sleep(self.render_sleep)
        
        return action
    
    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()
            self.rate.sleep()