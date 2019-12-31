# Omnidirectional Walking Pattern Generation for Humanoid Robot 
# By : Eko Rudiawan Jamzuri
# 31 December 2019
# This code is an implementation of paper
# Harada, Kensuke, et al. "An analytical method for real-time gait planning for humanoid robots." International Journal of Humanoid Robotics 3.01 (2006): 1-19.

import numpy as np 
import scipy.io 
import itertools
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *
from pytransform3d.trajectories import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from pytransform3d.plot_utils import Trajectory
import matplotlib.animation as animation
from pytransform3d.transformations import transform_from, concat

RIGHT_SUPPORT = 0
LEFT_SUPPORT = 1

class GaitController():
    def __init__(self):
        # Constant distance between hip to center 
        self.hip_offset = 0.035

        # Command for walking pattern
        # Defined as motion vector 
        self.cmd_x = 0.04
        self.cmd_y = 0.02
        self.cmd_a = np.radians(0)

        self.sx = 0.0
        self.sy = 0.0
        self.sa = 0.0
        
        # ZMP trajectory
        self.zmp_x = 0
        self.zmp_y = 0

        # Footsteps FIFO 
        # Use 3 foot pattern for 1 cycle gait
        self.footsteps = [[0.0,-self.hip_offset,0],
                          [0.0,self.hip_offset,0],
                          [0.0,-self.hip_offset,0]]

        self.zmp_x_record = []
        self.zmp_y_record = []

        self.footsteps_record = [[0.0,-self.hip_offset,0],
                                 [0.0,self.hip_offset,0],
                                 [0.0,-self.hip_offset,0]]

        # First support is right leg
        self.support_foot = RIGHT_SUPPORT

        # CoM pose
        self.com = [0,0,0,0,0,0,0]
        self.com_yaw = 0
        # Initial CoM yaw orientation
        self.init_com_yaw = 0.0 
        # Target CoM yaw orientation
        self.target_com_yaw = 0.0
        
        self.com_x_record = []
        self.com_y_record = []
        
        # Initial position and orientation for left swing foot
        self.init_lfoot_pose = np.zeros((7,1), dtype=float)
        self.init_lfoot_position = np.zeros((3,1), dtype=float)
        self.init_lfoot_orientation_yaw = 0.0 
        # Target position and orientation for left swing foot
        self.target_lfoot_pose = np.zeros((7,1), dtype=float)
        self.target_lfoot_position = np.zeros((3,1), dtype=float)
        self.target_lfoot_orientation_yaw = 0.0

        # Initial position and orientation for right swing foot
        self.init_rfoot_pose = np.zeros((7,1), dtype=float)
        self.init_rfoot_position = np.zeros((3,1), dtype=float)
        self.init_rfoot_orientation_yaw = 0.0
        # Target position and orientation for right swing foot
        self.target_rfoot_pose = np.zeros((7,1), dtype=float)
        self.target_rfoot_position = np.zeros((3,1), dtype=float)
        self.target_rfoot_orientation_yaw = 0.0
        
        # Current left foot and right foot pose written from world frame
        self.cur_rfoot = [0,-self.hip_offset,0,0,0,0,0]
        self.cur_lfoot = [0,self.hip_offset,0,0,0,0,0]

        # Current left foot and right foot pose written from CoM frame
        self.left_foot_pose = []
        self.right_foot_pose = []

    # Set default gait parameter
    def get_gait_parameter(self):
        self.zc = 0.33 # CoM constant height
        self.max_swing_height = 0.03 # Maximum swing foot height 

        self.t_step = 0.25 # Timing for 1 cycle gait
        self.dsp_ratio = 0.15 # Percent of DSP phase
        self.dt = 0.01 # Control cycle

        self.t_dsp = self.dsp_ratio * self.t_step 
        self.t_ssp = (1.0 - self.dsp_ratio) * self.t_step
        self.t = 0
        self.dt_bez = 1 / (self.t_ssp / self.dt)
        self.t_bez = 0

    def print_gait_parameter(self):
        print("zc :", self.zc)
        print("dt :", self.dt)

    # Bezier curve function for generating rotation path
    def rot_path(self, init_angle, target_angle, time, t):
        p0 = np.array([[0],[init_angle]])
        p1 = np.array([[0],[target_angle]])
        p2 = np.array([[time],[target_angle]])
        p3 = np.array([[time],[target_angle]])
        path = np.power((1-t), 3)*p0 + 3*np.power((1-t), 2)*t*p1 + 3*(1-t)*np.power(t, 2)*p2 + np.power(t, 3)*p3
        return path

    # Bezier curve function for generating position path
    def swing_foot_path(self, str_pt, end_pt, swing_height, t):
        p0 = str_pt.copy()
        p1 = str_pt.copy()
        p1[2,0] = swing_height+(0.25*swing_height)
        p2 = end_pt.copy()
        p2[2,0] = swing_height+(0.25*swing_height)
        p3 = end_pt.copy()
        path = np.power((1-t), 3)*p0 + 3*np.power((1-t), 2)*t*p1 + 3*(1-t)*np.power(t, 2)*p2 + np.power(t, 3)*p3
        return path

    # Update support foot
    def swap_support_foot(self):
        if self.support_foot == RIGHT_SUPPORT:
            self.support_foot = LEFT_SUPPORT
        else:
            self.support_foot = RIGHT_SUPPORT

    # Function for generating swing foot trajectory
    # Result in foot pose written from world coordinate
    def get_foot_trajectory(self):
        # Get initial position and orientation of swing foot
        if self.t == 0:
            if self.support_foot == LEFT_SUPPORT:
                self.init_rfoot_pose[0,0] = self.cur_rfoot[0]
                self.init_rfoot_pose[1,0] = self.cur_rfoot[1]
                self.init_rfoot_pose[2,0] = 0
                self.init_rfoot_pose[3,0] = self.cur_rfoot[3]
                self.init_rfoot_pose[4,0] = self.cur_rfoot[4]
                self.init_rfoot_pose[5,0] = self.cur_rfoot[5]
                self.init_rfoot_pose[6,0] = self.cur_rfoot[6]
                # Set initial position of swing foot
                self.init_rfoot_position[0,0] = self.init_rfoot_pose[0,0]
                self.init_rfoot_position[1,0] = self.init_rfoot_pose[1,0]
                self.init_rfoot_position[2,0] = self.init_rfoot_pose[2,0]
                euler = euler_from_quaternion([self.init_rfoot_pose[3,0], self.init_rfoot_pose[4,0], self.init_rfoot_pose[5,0], self.init_rfoot_pose[6,0]])
                # Set initial yaw orientation from swing foot
                self.init_rfoot_orientation_yaw = euler[2] 

                # Set target foot pose from next footstep
                self.target_rfoot_pose[0,0] = self.footsteps[1][0]
                self.target_rfoot_pose[1,0] = self.footsteps[1][1]
                self.target_rfoot_pose[2,0] = 0
                q = quaternion_from_euler(0, 0, self.footsteps[1][2])
                self.target_rfoot_pose[3,0] = q[0]
                self.target_rfoot_pose[4,0] = q[1]
                self.target_rfoot_pose[5,0] = q[2]
                self.target_rfoot_pose[6,0] = q[3]
                # Set target position of swing foot
                self.target_rfoot_position[0,0] = self.target_rfoot_pose[0,0]
                self.target_rfoot_position[1,0] = self.target_rfoot_pose[1,0]
                self.target_rfoot_position[2,0] = self.target_rfoot_pose[2,0]
                euler = euler_from_quaternion([self.target_rfoot_pose[3,0], self.target_rfoot_pose[4,0], self.target_rfoot_pose[5,0], self.target_rfoot_pose[6,0]])
                # Set target orientation of swing foot
                self.target_rfoot_orientation_yaw = euler[2]
                euler = euler_from_quaternion([self.cur_lfoot[3], self.cur_lfoot[4], self.cur_lfoot[5], self.cur_lfoot[6]])
                support_foot_yaw = euler[2]
                # Calculate initial CoM yaw orientation and target CoM yaw orientation
                self.init_com_yaw = (support_foot_yaw + self.init_rfoot_orientation_yaw) / 2
                self.target_com_yaw = (support_foot_yaw + self.target_rfoot_orientation_yaw) / 2
            if self.support_foot == RIGHT_SUPPORT:
                self.init_lfoot_pose[0,0] = self.cur_lfoot[0]
                self.init_lfoot_pose[1,0] = self.cur_lfoot[1]
                self.init_lfoot_pose[2,0] = 0
                self.init_lfoot_pose[3,0] = self.cur_lfoot[3]
                self.init_lfoot_pose[4,0] = self.cur_lfoot[4]
                self.init_lfoot_pose[5,0] = self.cur_lfoot[5]
                self.init_lfoot_pose[6,0] = self.cur_lfoot[6]
                self.init_lfoot_position[0,0] = self.init_lfoot_pose[0,0]
                self.init_lfoot_position[1,0] = self.init_lfoot_pose[1,0]
                self.init_lfoot_position[2,0] = self.init_lfoot_pose[2,0]
                euler = euler_from_quaternion([self.init_lfoot_pose[3,0], self.init_lfoot_pose[4,0], self.init_lfoot_pose[5,0], self.init_lfoot_pose[6,0]])
                self.init_lfoot_orientation_yaw = euler[2]
                self.target_lfoot_pose[0,0] = self.footsteps[1][0]
                self.target_lfoot_pose[1,0] = self.footsteps[1][1]
                self.target_lfoot_pose[2,0] = 0
                q = quaternion_from_euler(0, 0, self.footsteps[1][2])
                self.target_lfoot_pose[3,0] = q[0]
                self.target_lfoot_pose[4,0] = q[1]
                self.target_lfoot_pose[5,0] = q[2]
                self.target_lfoot_pose[6,0] = q[3]
                self.target_lfoot_position[0,0] = self.target_lfoot_pose[0,0]
                self.target_lfoot_position[1,0] = self.target_lfoot_pose[1,0]
                self.target_lfoot_position[2,0] = self.target_lfoot_pose[2,0]
                euler = euler_from_quaternion([self.target_lfoot_pose[3,0], self.target_lfoot_pose[4,0], self.target_lfoot_pose[5,0], self.target_lfoot_pose[6,0]])
                self.target_lfoot_orientation_yaw = euler[2]
                euler = euler_from_quaternion([self.cur_rfoot[3], self.cur_rfoot[4], self.cur_rfoot[5], self.cur_rfoot[6]])
                support_foot_yaw = euler[2]
                self.init_com_yaw = (support_foot_yaw + self.init_lfoot_orientation_yaw) / 2
                self.target_com_yaw = (support_foot_yaw + self.target_lfoot_orientation_yaw) / 2

        # Generate foot trajectory 
        if self.t < (self.t_dsp/2.0) or self.t >= (self.t_dsp/2.0 + self.t_ssp):
            self.t_bez = 0
        else:
            if self.support_foot == LEFT_SUPPORT:
                self.cur_lfoot[0] = self.footsteps[0][0]
                self.cur_lfoot[1] = self.footsteps[0][1]
                self.cur_lfoot[2] = 0
                q = quaternion_from_euler(0,0,self.footsteps[0][2])
                self.cur_lfoot[3] = q[0]
                self.cur_lfoot[4] = q[1]
                self.cur_lfoot[5] = q[2]
                self.cur_lfoot[6] = q[3]
                path = self.swing_foot_path(self.init_rfoot_position, self.target_rfoot_position, self.max_swing_height, self.t_bez)
                self.cur_rfoot[0] = path[0,0]
                self.cur_rfoot[1] = path[1,0]
                self.cur_rfoot[2] = path[2,0]
                yaw_path = self.rot_path(self.init_rfoot_orientation_yaw, self.target_rfoot_orientation_yaw, self.t_ssp, self.t_bez)
                q = quaternion_from_euler(0,0,yaw_path[1,0])
                self.cur_rfoot[3] = q[0]
                self.cur_rfoot[4] = q[1]
                self.cur_rfoot[5] = q[2]
                self.cur_rfoot[6] = q[3]
            elif self.support_foot == RIGHT_SUPPORT:
                self.cur_rfoot[0] = self.footsteps[0][0]
                self.cur_rfoot[1] = self.footsteps[0][1]
                self.cur_rfoot[2] = 0
                q = quaternion_from_euler(0,0,self.footsteps[0][2])
                self.cur_rfoot[3] = q[0]
                self.cur_rfoot[4] = q[1]
                self.cur_rfoot[5] = q[2]
                self.cur_rfoot[6] = q[3]
                path = self.swing_foot_path(self.init_lfoot_position, self.target_lfoot_position, self.max_swing_height, self.t_bez)
                self.cur_lfoot[0] = path[0,0]
                self.cur_lfoot[1] = path[1,0]
                self.cur_lfoot[2] = path[2,0]
                yaw_path = self.rot_path(self.init_lfoot_orientation_yaw, self.target_lfoot_orientation_yaw, self.t_ssp, self.t_bez)
                q = quaternion_from_euler(0,0, yaw_path[1,0])
                self.cur_lfoot[3] = q[0]
                self.cur_lfoot[4] = q[1]
                self.cur_lfoot[5] = q[2]
                self.cur_lfoot[6] = q[3]

            # Generate CoM yaw path
            yaw_path = self.rot_path(self.init_com_yaw, self.target_com_yaw, self.t_ssp, self.t_bez)
            self.com_yaw = yaw_path[1,0]
            self.t_bez += self.dt_bez
    
    # Function for generating zmp trajectory
    def get_zmp_trajectory(self):
        epsilon = 0.0001 
        td = self.t % self.t_step 
        if td > -epsilon and td < epsilon:
            self.t0 = self.t
            self.t1 = self.t0 + (self.t_ssp / 2)
            self.t2 = self.t1 + self.t_dsp
            self.tf = self.t_step
            # Initial CoM position
            self.com0_x = self.footsteps[0][0] + (self.footsteps[1][0] - self.footsteps[0][0]) / 2
            self.com0_y = self.footsteps[0][1] + (self.footsteps[1][1] - self.footsteps[0][1]) / 2
            # Final CoM position
            self.com1_x = self.footsteps[1][0] + (self.footsteps[2][0] - self.footsteps[1][0]) / 2
            self.com1_y = self.footsteps[1][1] + (self.footsteps[2][1] - self.footsteps[1][1]) / 2
            # Support foot
            self.sup_x = self.footsteps[1][0]
            self.sup_y = self.footsteps[1][1]

        if self.t >= self.t0 and self.t < self.t1:
            self.zmp_x = self.com0_x+((self.sup_x-self.com0_x)/(self.t1-self.t0))*self.t
            self.zmp_y = self.com0_y+((self.sup_y-self.com0_y)/(self.t1-self.t0))*self.t
        elif self.t >= self.t1 and self.t < self.t2:
            self.zmp_x = self.sup_x
            self.zmp_y = self.sup_y 
        elif self.t >= self.t2 and self.t < self.tf:
            self.zmp_x=self.sup_x+((self.com1_x-self.sup_x)/(self.tf-self.t2))*(self.t-self.t2)
            self.zmp_y=self.sup_y+((self.com1_y-self.sup_y)/(self.tf-self.t2))*(self.t-self.t2)
        self.zmp_x_record.append(self.zmp_x)
        self.zmp_y_record.append(self.zmp_y)
    
    # Add new footstep to FIFO buffer
    def add_new_footstep(self):
        self.footsteps.pop(0)
        if self.support_foot == LEFT_SUPPORT: 
            self.sx = self.cmd_x
            self.sy = -2*self.hip_offset + self.cmd_y
            self.sa += self.cmd_a
            dx = self.footsteps[-1][0] + np.cos(self.sa) * self.sx + (-np.sin(self.sa) * self.sy)
            dy = self.footsteps[-1][1] + np.sin(self.sa) * self.sx + np.cos(self.sa) * self.sy
            self.footsteps.append([dx, dy, self.sa])
            self.footsteps_record .append([dx, dy, self.sa])
        elif self.support_foot == RIGHT_SUPPORT:
            self.sx = self.cmd_x 
            self.sy = 2*self.hip_offset + self.cmd_y
            self.sa += self.cmd_a
            dx = self.footsteps[-1][0] + np.cos(self.sa) * self.sx + (-np.sin(self.sa) * self.sy)
            dy = self.footsteps[-1][1] + np.sin(self.sa) * self.sx + np.cos(self.sa) * self.sy
            self.footsteps.append([dx, dy, self.sa])
            self.footsteps_record .append([dx, dy, self.sa])
        self.swap_support_foot()

    # Function for generating CoM trajectory
    def get_com_trajectory(self):
        self.Tc = np.sqrt(9.81/self.zc)
        cx = np.array([0,
                       (np.sinh(self.Tc*(self.t1 - self.tf))*(self.sup_x - self.com0_x))/(self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - (np.sinh(self.Tc*(self.t2 - self.tf))*(self.sup_x - self.com1_x))/(self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(2*self.t1 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_x - self.com0_x))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.sinh(self.Tc*(2*self.t1 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_x - self.com0_x))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.sinh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_x - self.com0_x))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(2*self.t2 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((self.sup_x - self.com0_x)*(np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) + np.sinh(self.Tc*(self.t1 - self.t2 + self.tf))))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(2*self.t2 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_x - self.com1_x))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf))])

        cy = np.array([0,
                       (np.sinh(self.Tc*(self.t1 - self.tf))*(self.sup_y - self.com0_y))/(self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - (np.sinh(self.Tc*(self.t2 - self.tf))*(self.sup_y - self.com1_y))/(self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(2*self.t1 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_y - self.com0_y))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.sinh(self.Tc*(2*self.t1 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_y - self.com0_y))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.sinh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((np.cosh(self.Tc*(self.t1 + self.t2 - self.tf)) - np.cosh(self.Tc*(self.t1 - self.t2 + self.tf)))*(self.sup_y - self.com0_y))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.cosh(self.Tc*(2*self.t2 - self.tf)) - np.cosh(self.Tc*self.tf))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf)),
                       ((self.sup_y - self.com0_y)*(np.sinh(self.Tc*(self.t1 + self.t2 - self.tf)) + np.sinh(self.Tc*(self.t1 - self.t2 + self.tf))))/(2*self.Tc*self.t1*np.sinh(self.Tc*self.tf)) - ((np.sinh(self.Tc*(2*self.t2 - self.tf)) + np.sinh(self.Tc*self.tf))*(self.sup_y - self.com1_y))/(2*self.Tc*np.sinh(self.Tc*self.tf)*(self.t2 - self.tf))])

        if self.t >= self.t0 and self.t < self.t1:
            self.com[0] = self.com0_x+((self.sup_x-self.com0_x)/(self.t1-self.t0))*(self.t-self.t0)+cx[0]*np.cosh(self.Tc*self.t)+cx[1]*np.sinh(self.Tc*self.t)
            self.com[1] = self.com0_y+((self.sup_y-self.com0_y)/(self.t1-self.t0))*(self.t-self.t0)+cy[0]*np.cosh(self.Tc*self.t)+cy[1]*np.sinh(self.Tc*self.t)
        elif self.t >= self.t1 and self.t < self.t2:
            self.com[0] = self.sup_x+cx[2]*np.cosh(self.Tc*(self.t-self.t1))+cx[3]*np.sinh(self.Tc*(self.t-self.t1))
            self.com[1] = self.sup_y+cy[2]*np.cosh(self.Tc*(self.t-self.t1))+cy[3]*np.sinh(self.Tc*(self.t-self.t1))
        elif self.t >= self.t2 and self.t < self.tf:
            self.com[0] = self.sup_x+((self.com1_x-self.sup_x)/(self.tf-self.t2))*(self.t-self.t2)+cx[4]*np.cosh(self.Tc*(self.t-self.t2))+cx[5]*np.sinh(self.Tc*(self.t-self.t2))
            self.com[1] = self.sup_y+((self.com1_y-self.sup_y)/(self.tf-self.t2))*(self.t-self.t2)+cy[4]*np.cosh(self.Tc*(self.t-self.t2))+cy[5]*np.sinh(self.Tc*(self.t-self.t2))
        # CoM height is constant
        self.com[2] = self.zc 
        # CoM orientation 
        q = quaternion_from_euler(0, 0, self.com_yaw)
        self.com[3] = q[0]
        self.com[4] = q[1]
        self.com[5] = q[2]
        self.com[6] = q[3]
        self.com_x_record.append(self.com[0])
        self.com_y_record.append(self.com[1])

    # Create transformation matrix
    def create_tf_matrix(self, list_xyz_qxyzw):
        T_mat = np.eye(4)
        T_mat[0,3] = list_xyz_qxyzw[0]
        T_mat[1,3] = list_xyz_qxyzw[1]
        T_mat[2,3] = list_xyz_qxyzw[2]
        R_mat = matrix_from_quaternion([list_xyz_qxyzw[6], list_xyz_qxyzw[3], list_xyz_qxyzw[4], list_xyz_qxyzw[5]])
        T_mat[:3,:3] = R_mat
        return T_mat

    # Function for tranform left foot and right foot pose into CoM frame
    def get_foot_pose(self):
        world_to_com = self.create_tf_matrix(self.com)
        world_to_lfoot = self.create_tf_matrix(self.cur_lfoot)
        world_to_rfoot = self.create_tf_matrix(self.cur_rfoot)
        world_to_com_inv = np.linalg.pinv(world_to_com)
        com_to_lfoot = world_to_com_inv.dot(world_to_lfoot)
        com_to_rfoot = world_to_com_inv.dot(world_to_rfoot)
        q_lfoot = quaternion_from_matrix(com_to_lfoot[:3,:3])
        q_rfoot = quaternion_from_matrix(com_to_rfoot[:3,:3])
        self.left_foot_pose = [com_to_lfoot[0,3], com_to_lfoot[1,3], com_to_lfoot[2,3], q_lfoot[1], q_lfoot[2], q_lfoot[3], q_lfoot[0]]
        self.right_foot_pose = [com_to_rfoot[0,3], com_to_rfoot[1,3], com_to_rfoot[2,3], q_rfoot[1], q_rfoot[2], q_rfoot[3], q_rfoot[0]]
    
    # Function for getting walking pattern
    def get_walking_pattern(self):
        self.get_zmp_trajectory()
        self.get_com_trajectory()
        self.get_foot_trajectory()
        self.get_foot_pose()
        self.t += self.dt 
        if self.t > self.t_step:
            self.t = 0
            self.add_new_footstep()

    def initialize(self):
        self.get_gait_parameter()
        self.print_gait_parameter()
        
    def end(self):
        pass

    def run(self):
        print("===========================")
        print("Barelang Gait Controller   ")
        print("===========================")
        self.initialize()
        t_sim = 5
        t = 0
        com_trajectory = []
        lfoot_trajectory = []
        rfoot_trajectory = []
        while t < t_sim:
            self.get_walking_pattern()
            com_trajectory.append([self.com[0], self.com[1], self.com[2], self.com[6], self.com[3], self.com[4], self.com[5]])
            rfoot_trajectory.append([self.cur_rfoot[0], self.cur_rfoot[1], self.cur_rfoot[2], self.cur_rfoot[6], self.cur_rfoot[3], self.cur_rfoot[4], self.cur_rfoot[5]])
            lfoot_trajectory.append([self.cur_lfoot[0], self.cur_lfoot[1], self.cur_lfoot[2], self.cur_lfoot[6], self.cur_lfoot[3], self.cur_lfoot[4], self.cur_lfoot[5]])
            t += self.dt

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        com_trajectory = np.array(com_trajectory)
        rfoot_trajectory = np.array(rfoot_trajectory)
        lfoot_trajectory = np.array(lfoot_trajectory)
        plot_trajectory(ax=ax, P=com_trajectory, s=0.01, show_direction=False)
        plot_trajectory(ax=ax, P=rfoot_trajectory, s=0.01, show_direction=False)
        plot_trajectory(ax=ax, P=lfoot_trajectory, s=0.01, show_direction=False)
        ax.set_ylim(-0.2,0.7)
        ax.set_zlim(0.0,0.5)
        plt.show()
        self.end()

def main():
    gc = GaitController()
    gc.run()

if __name__ == "__main__":
    main()