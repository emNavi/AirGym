#!/home/emnavi/miniconda3/envs/inference/bin/python

from rl_games.common.player import BasePlayer
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
from nav_msgs.msg import Odometry

from sim2real.src.real_inference.src.utils import torch_ext

# for PositionTarget
IGNORE_PX=1
IGNORE_PY=2
IGNORE_PZ=4
IGNORE_VX=8
IGNORE_VY=16
IGNORE_VZ=32
IGNORE_AFX=64
IGNORE_AFY=128
IGNORE_AFZ=256
FORCE=512
IGNORE_YAW=1024
IGNORE_YAW_RATE=2048
# for AttitudeTarget
IGNORE_ROLL_RATE=1
IGNORE_PITCH_RATE=2
IGNORE_YAW_RATE_=4
IGNORE_THRUST=64
IGNORE_ATTITUDE=128

class CpuPlayerContinuous(PpoPlayerContinuous):
    def __init__(self, params):
        super().__init__(params)
        print("Running on", self.device)

        # initialize
        rospy.init_node('onboard_computing_node', anonymous=True)

        self.obs_sub = rospy.Subscriber('/random_states', Odometry, self.callback)
        ctl_mode = self.ctl_mode = self.env_config.get('ctl_mode')
        self.action_pub = rospy.Publisher('/airgym/cmd', PositionTarget, queue_size=2000)
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

        self.is_tensor_obses = True

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def callback(self, data):
        # Process the incoming message
        rospy.loginfo(f"Obs reveived...")
        
        pose = data.pose.pose
        pose_tensor = torch.tensor([pose.position.x, pose.position.y, pose.position.z], device=self.device)
        quat_tensor = torch.tensor([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], device=self.device)
        twist = data.twist.twist
        linvel_tensor = torch.tensor([twist.linear.x, twist.linear.y, twist.linear.z], device=self.device)
        angvel_tensor = torch.tensor([twist.angular.x, twist.angular.y, twist.angular.z], device=self.device)
        real_obs = torch.cat((pose_tensor, quat_tensor, linvel_tensor, angvel_tensor)).unsqueeze(0)

        action = self.inference(real_obs)

        # Create the output message
        if self.ctl_mode == "pos":
            output_msg = PositionTarget()
            output_msg.position.x = action[0].cpu().numpy()
            output_msg.position.y = action[1].cpu().numpy()
            output_msg.position.z = action[2].cpu().numpy()
            output_msg.yaw = action[3].cpu().numpy()
            output_msg.type_mask = IGNORE_VX | IGNORE_VY | IGNORE_VZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | FORCE | IGNORE_YAW_RATE
        elif self.ctl_mode == "vel":
            output_msg = PositionTarget()
            output_msg.velocity.x = action[0].cpu().numpy()
            output_msg.velocity.y = action[1].cpu().numpy()
            output_msg.velocity.z = action[2].cpu().numpy()
            output_msg.yaw = action[3].cpu().numpy()
            output_msg.type_mask = IGNORE_PX | IGNORE_PY | IGNORE_PZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | FORCE | IGNORE_YAW_RATE
        elif self.ctl_mode == "atti": # body_rate stores angular
            output_msg = AttitudeTarget()
            output_msg.body_rate.x = action[0].cpu().numpy()
            output_msg.body_rate.y = action[1].cpu().numpy()
            output_msg.body_rate.z = action[2].cpu().numpy()
            output_msg.thrust = action[3].cpu().numpy()
            output_msg.type_mask = IGNORE_ROLL_RATE | IGNORE_PITCH_RATE | IGNORE_YAW_RATE_
        elif self.ctl_mode == "rate":
            output_msg = AttitudeTarget()
            output_msg.body_rate.x = action[0].cpu().numpy()
            output_msg.body_rate.y = action[1].cpu().numpy()
            output_msg.body_rate.z = action[2].cpu().numpy()
            output_msg.thrust = action[3].cpu().numpy()
            output_msg.type_mask = IGNORE_ATTITUDE
        else:
            pass
        
        # Publish the message
        self.action_pub.publish(output_msg)

    def inference(self, real_obses):
        if self.has_masks:
            masks = self.env.get_action_mask()
            action = self.get_masked_action(
                real_obses, masks, self.is_deterministic)
        else:
            action = self.get_action(real_obses, self.is_deterministic)
        
        if self.render_env:
            self.env.render(mode='human')
            time.sleep(self.render_sleep)
        return action
    
    def run(self):
        # get into loop
        while not rospy.is_shutdown():
            rospy.spin()
            self.rate.sleep()

    def env_step(self, env, actions):
        return super().env_step(env, actions)