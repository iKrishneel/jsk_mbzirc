#/usr/bin/env python

import roslib
roslib.load_manifest("jsk_mbzirc_tasks")

import rospy
from cv_bridge import CvBridge
import message_filters
import tf

from sklearn.neighbors import NearestNeighbors, KDTree, DistanceMetric
from scipy.spatial import distance

from sensor_msgs.msg import Image, PointCloud2, Imu
from jsk_mbzirc_msgs.msg import ProjectionMatrix
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

import numpy as np
from  collections import Counter
import sys
import math
import cv2
import random
import time

sub_mask_ = '/track_region_mapping/output/track_mask'
sub_odo_ = '/ground_truth/state'
sub_imu_ = '/raw_imu'
sub_matrix_ = '/projection_matrix'

sub_image_ = '/downward_cam/camera/image'
sub_point3d_ = '/uav_landing_region/output/point'
sub_pose_ = '/uav_landing_region/output/pose'

pub_image_ = None
pub_topic_ = '/track_region_segmentation/output/track_mask'

ALTITUDE_THRESH_ = 5.0  ## for building map
DISTANCE_THRESH_ = 4.0  ## incase of FP
VEHICLE_SPEED_ = 3.0  ## assume fast seed of 15km/h

class MapInfo:
    image = None
    indices = []
    point3d = []
    odometry = None
    imu = None

class HeliportAlignmentAndPredictor:
    def __init__(self):
        self.pub_image_ = rospy.Publisher(pub_topic_, Image, queue_size=10) 

        self.subscribe()
        self.map_info = MapInfo()
        self.proj_matrix = None
        self.is_initalized = False
        self.kdtree = None
        self.position_list = []

    def subscribe(self):
        mask_sub = message_filters.Subscriber(sub_mask_, Image)
        odom_sub = message_filters.Subscriber(sub_odo_, Odometry)
        imu_sub = message_filters.Subscriber(sub_imu_, Imu)
        pmat_sub = message_filters.Subscriber(sub_matrix_, ProjectionMatrix)
        init_ats = message_filters.ApproximateTimeSynchronizer((mask_sub, odom_sub, imu_sub, pmat_sub), 10, 1)
        init_ats.registerCallback(self.init_callback)
        
        sub_image = message_filters.Subscriber(sub_image_, Image)
        sub_point3d = message_filters.Subscriber(sub_point3d_, PointStamped)
        sub_pose = message_filters.Subscriber(sub_pose_, PoseStamped)

        ats = message_filters.ApproximateTimeSynchronizer((sub_image, sub_point3d, sub_pose), 10, 10)
        ats.registerCallback(self.callback)
        
    def callback(self, image_msg, point_msg, pose_msg):
    
        if not self.is_initalized:
            rospy.logerr("-- vehicle track map info is not availible")
            return        
        
        current_point = np.array((point_msg.point.x, point_msg.point.y, point_msg.point.z))
        current_point = current_point.reshape(1, -1)
        distances, indices = self.kdtree.kneighbors(current_point)
        
        ##? add condition to limit neigbors
        if distances > DISTANCE_THRESH_:
            rospy.logerr("point is to far")
            return
        
        ## debug view
        im_color = cv2.cvtColor(self.map_info.image, cv2.COLOR_GRAY2BGR)
        x, y = self.map_info.indices[indices]
        cv2.circle(im_color, (x, y), 10, (0, 255, 0), -1)

        # get the direction
        self.position_list.append([current_point, point_msg.header.stamp])
        if len(self.position_list) < 2:
            rospy.logwarn("direction is unknow... another detection is required")
            return

        time_diff = point_msg.header.stamp - self.position_list[0][1]
        #print "difference in time: ", time_diff

        prev_index = len(self.position_list) - 2
        vehicle_direction = current_point - self.position_list[prev_index][0]
        print "DIRECTION IS: ", vehicle_direction, "\n", current_point, "\n", self.position_list[prev_index][0]
            
        # TEST - to plot predicted motion
        itr = 0
        prev_pt = current_point
        dir_x = (vehicle_direction[0][0]/math.fabs(vehicle_direction[0][0])) * VEHICLE_SPEED_
        dir_y = (vehicle_direction[0][1]/math.fabs(vehicle_direction[0][1])) * VEHICLE_SPEED_
        while (itr < 200):
            next_pt = prev_pt

            print dir_x , "\t", dir_y 
            
            next_pt[0][0] = prev_pt[0][0] + dir_x
            next_pt[0][1] = prev_pt[0][1] + dir_y
            dist, index = self.kdtree.kneighbors(next_pt) #, radius = VEHICLE_SPEED_, return_distance = True)
            prev_pt[0][0] = self.map_info.point3d[index][0]
            prev_pt[0][1] = self.map_info.point3d[index][1]

            print prev_pt
            print next_pt
            
            diff = prev_pt - next_pt
            dx = (diff[0][0]/math.fabs(diff[0][0])) * VEHICLE_SPEED_
            dy = (diff[0][1]/math.fabs(diff[0][1])) * VEHICLE_SPEED_

            print dx, "\t", dy , "\n"
            
            if not math.isnan(dx) and not math.isnan(dy):
                dir_x = dx
                dir_y = dy
            
            x1, y1 = self.map_info.indices[index]
            cv2.circle(im_color, (x1, y1), 5, (0, 0, 255), -1)
            self.plot_image("plot", im_color)
            cv2.waitKey(30)

            print "iterator: ", itr
            rospy.sleep(1)
            
            itr += 1
            
        # self.plot_image("input", self.map_info.image)
        self.plot_image("plot", im_color)
        cv2.waitKey(3)

    def init_callback(self, mask_msg, odometry_msgs, imu_msg, projection_msg):
        self.proj_matrix = np.reshape(projection_msg.data, (3, 4))
        if (self.is_initalized):
            return

        altit = odometry_msgs.pose.pose.position.z
        if altit < ALTITUDE_THRESH_:
            rospy.logwarn("cannot build the map at this altitude: "+ str(altit))
            return

        ## TODO: skeletonize the mask
        
        
        image = self.convert_image(mask_msg, "mono8")        
        world_points = []
        indices = []
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y, x] == 255:
                    point3d = self.projection_to_world_coords(x, y, 0.0)
                    world_points.append(point3d)
                    indices.append([x, y])
                    
        self.map_info.indices = indices
        self.map_info.point3d = world_points
        self.map_info.odometry = odometry_msgs
        self.map_info.imu = imu_msg
        self.map_info.image = image
        self.is_initalized = True
        
        self.kdtree = NearestNeighbors(n_neighbors = 1, radius = VEHICLE_SPEED_, algorithm = "kd_tree", leaf_size = 30, \
                                       metric='euclidean').fit(np.array(world_points))
        
        rospy.loginfo("-- map initialized")

        del image
        del world_points
        del indices
        del altit
        
    def convert_image(self, image_msg, encoding):
        bridge = CvBridge()
        cv_img = None
        try:
            cv_img = bridge.imgmsg_to_cv2(image_msg, str(encoding))
        except Exception as e:
            print (e)
        return cv_img

    def plot_image(self, name, image):
        cv2.namedWindow(str(name), cv2.WINDOW_NORMAL)
        cv2.imshow(str(name), image)

    def projection_to_world_coords(self, x, y, ground_z = 0.0):    
        a00 = x * self.proj_matrix[2, 0] - self.proj_matrix[0, 0]
        a01 = x * self.proj_matrix[2, 1] - self.proj_matrix[0, 1]
        a10 = y * self.proj_matrix[2, 0] - self.proj_matrix[1, 0]
        a11 = y * self.proj_matrix[2, 1] - self.proj_matrix[1, 1]
        bv0 = self.proj_matrix[0, 2] * ground_z + self.proj_matrix[0, 3] -  \
              x * self.proj_matrix[2, 2] * ground_z - x * self.proj_matrix[2, 3]
        bv1 = self.proj_matrix[1, 2] * ground_z + self.proj_matrix[1, 3] -  \
              y * self.proj_matrix[2, 2] * ground_z - y * self.proj_matrix[2, 3]
        denom = a11 * a00 - a01 * a10
        pos_x = (a11 * bv0 - a01 * bv1) / denom
        pos_y = (a00 * bv1 - a10 * bv0) / denom
        return (pos_x, pos_y, ground_z)


def main():
    rospy.init_node('uav_heliport_alignment_prediction', anonymous=True)
    hpp = HeliportAlignmentAndPredictor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logwarn("SHUT DOWN COMMAND RECEIVED")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

