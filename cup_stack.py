#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Inverse Kinematics Pick and Place Demo
"""
import argparse
import struct
import sys
import copy

import rospy
import rospkg
import ast


from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface



class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True):
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self._iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + self._hover_distance
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)

    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()



def main():
    """RSDK Inverse Kinematics Pick and Place Example
    A Pick and Place example using the Rethink Inverse Kinematics
    Service which returns the joint angles a requested Cartesian Pose.
    This ROS Service client is used to request both pick and place
    poses in the /base frame of the robot.
    Note: This is a highly scripted and tuned demo. The object location
    is "known" and movement is done completely open loop. It is expected
    behavior that Baxter will eventually mis-pick or drop the block. You
    can improve on this demo by adding perception and feedback to close
    the loop.
    """
    rospy.init_node("ik_pick_and_place_demo")
    # Load Gazebo Models via Spawning Services
    # Note that the models reference is the /world frame
    # and the IK operates with respect to the /base frame
    # Remove models from the scene on shutdown


    limb = 'left'
    hover_distance = 0.15 # meters

    gripper = baxter_interface.Gripper('left')
    gripper.calibrate()

    # Starting Joint angles for right arm
    starting_joint_angles = {'right_w0': -0.672650575487754,
                             'right_w1':  0.9598884780192977,
                             'right_w2': 0.4843544337748194,
                             'right_e0': 1.172728312338399,
                             'right_e1': 1.9270633647810511,
                             'right_s0': 0.06787864986392955,
                             'right_s1': -0.9894176081860918}
    hover_joint_angles =    {'right_w0': -0.228946632591898,
                             'right_w1': 1.4365730078546899,
                             'right_w2': 0.15109710760671324,
                             'right_e0': 0.4007524808350643,
                             'right_e1': 1.024315671110485,
                             'right_s0': 0.7205874751091731,
                             'right_s1': -0.95490304045867}
    dab_left_angles = {'left_e0': -0.028378644575880154, 
                        'left_e1': -1.017796252761972, 
                        'left_s0': 0.30871363356193954, 
                        'left_s1': -1.4572817484911431, 
                        'left_w0': -0.3976845192592935,
                        'left_w1': -0.8954612849281103, 
                        'left_w2': 0.68568941218478,}
    dab_right_angles = {
                        'right_e0': 1.171961321944456, 
                        'right_e1': 0.8310340918369229, 
                        'right_s0': 0.3976845192592935, 
                        'right_s1': -0.724805922275858, 
                        'right_w0': -0.6622962051695274, 
                        'right_w1': 0.768140879533621, 
                        'right_w2': -0.03106311095467963}

    pnp = PickAndPlace('right', hover_distance)
    pnpl = PickAndPlace('left', hover_distance)
    
    # An orientation for gripper fingers to be overhead and parallel to the obj
    overhead_orientation = Quaternion(
                             x=-2^(1/2)/2,
                             y=2^(1/2)/2,
                             z=0,
                             w=0)



    cup_stack = list()

    cup_stack.append(Pose(
        position=Point(x=0.5, y=0.10, z=-0.129),
        orientation=overhead_orientation))

    cup_stack.append(Pose(
        position=Point(x=0.5, y=0, z=-.129),
        orientation=overhead_orientation))

    cup_stack.append(Pose(
        position=Point(x=0.5, y=-0.10, z=-.129),
        orientation=overhead_orientation))

    cup_stack.append(Pose(
        position=Point(x=0.5, y=0.05, z=-.009),
        orientation=overhead_orientation))

    cup_stack.append(Pose(
        position=Point(x=0.5, y=-0.05, z=-.009),
        orientation=overhead_orientation))

    cup_stack.append(Pose(
        position=Point(x=0.5, y=0, z=0.111),
        orientation=overhead_orientation))


    
    
    file =  open('/home/ryanbarry/ros_ws/src/baxter_simulator/baxter_sim_examples/scripts/Locations.txt', 'r')
    contents = file.read()
    file.close()
    print(contents)
    loc = contents.split(" | ")

    cup_locations = []

    for i in range(0, len(loc)):
        loc[i] = ast.literal_eval(loc[i])
        cup_locations.append(Pose(
            position=Point(x=-1*float(loc[i][0])/1000. + 0.1, y=float(loc[i][1])/1000. + 0.3, z=-.129),
            orientation=overhead_orientation))

    # print(cup_locations)
        


    # Move to the desired starting angles
    pnp.move_to_start(starting_joint_angles)
    idx = 0
    stacked_flag = 0
    while not rospy.is_shutdown():
        if stacked_flag == 0:
            if idx == 0:
                pnp.move_to_start(starting_joint_angles)

            pnp.pick(cup_locations[idx])
            pnp._guarded_move_to_joint_position(hover_joint_angles)
            pnp.place(cup_stack[idx])
            pnp._guarded_move_to_joint_position(hover_joint_angles)
            if idx < 3:
                idx += 1
            else:
                stacked_flag = 1
                pnp._guarded_move_to_joint_position(dab_right_angles)
                pnpl._guarded_move_to_joint_position(dab_right_angles)
                

        # pnp.gripper_open()
        # rospy.sleep(1)
        # pnp.gripper_close()


    # file =  open('Test.txt', 'r')
    # contents = file.read()
    # print(contents)
    return 0

if __name__ == '__main__':
    sys.exit(main())
