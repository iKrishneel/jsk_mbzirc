#/usr/bin/env python

import roslib
roslib.load_manifest("jsk_mbzirc_tasks")

import rospy
from cv_bridge import CvBridge
import message_filters

from sklearn.neighbors import NearestNeighbors, KDTree, DistanceMetric
from scipy.spatial import distance

from sensor_msgs.msg import Image, PointCloud2, Imu
from jsk_mbzirc_msgs.msg import ProjectionMatrix
from geometry_msgs.msg import Point, PointStamped
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
sub_point3d_ = '/track_region_mapping/output/point'

pub_image_ = None
pub_topic_ = '/track_region_segmentation/output/track_mask'

ALTITUDE_THRESH_ = 5.0  ## for building map

class MapInfo:
    image = None
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
        
        self.msg_info = MapInfo()

    def subscribe(self):
        mask_sub = message_filters.Subscriber(sub_mask_, Image)
        odom_sub = message_filters.Subscriber(sub_odo_, Odometry)
        imu_sub = message_filters.Subscriber(sub_imu_, Imu)
        pmat_sub = message_filters.Subscriber(sub_matrix_, ProjectionMatrix)
        init_ats = message_filters.ApproximateTimeSynchronizer((mask_sub, odom_sub, imu_sub, pmat_sub), 10, 1)
        init_ats.registerCallback(self.init_callback)
        
        sub_image = message_filters.Subscriber(sub_image_, Image)
        sub_point3d = message_filters.Subscriber(sub_point3d_, PointStamped)

        ats = message_filters.ApproximateTimeSynchronizer((sub_image, sub_point3d), 100, 10)
        ats.registerCallback(self.callback)
        
    def callback(self, image_msg, point_msg):
        print "RUNNING...."
        if not self.is_initalized:
            rospy.logerr("THE MAP IS NOT YET CREATED")
            return

        self.plot_image("input", self.map_info.image)
        cv2.waitKey(3)
        

    def init_callback(self, mask_msg, odometry_msgs, imu_msg, projection_msg):
        self.proj_matrix = np.reshape(projection_msg.data, (3, 4))
        if (self.is_initalized):
            self.msg_info.imu = imu_msg
            self.msg_info.odometry = odometry_msgs
            return

        altit = odometry_msgs.pose.pose.position.z
        if altit < ALTITUDE_THRESH_:
            rospy.logwarn("cannot build the map at this altitude: "+ str(altit))
            return

        image = self.convert_image(mask_msg, "mono8")        
        world_points = []
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if image[y, x] == 255:
                    point3d = self.projection_to_world_coords(x, y, 0.0)
                    world_points.append(point3d)

        self.map_info.point3d = world_points
        self.map_info.odometry = odometry_msgs
        self.map_info.imu = imu_msg
        self.map_info.image = image
        self.is_initalized = True
        
        rospy.loginfo("-- map initialized")
        # self.plot_image("input", image)
        # cv2.waitKey(3)
        
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

